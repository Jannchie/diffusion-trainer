"""Shared reader -> processor -> writer threaded pipeline for dataset processors.

`LatentsGenerateProcessor` and `TaggingProcessor` both stream a list of images
through three stages on separate thread pools, connected by bounded queues, and
report progress while watching for stalls. That orchestration lived twice; this
base class owns it and exposes the three per-item stages as hooks.

Progress contract: every input item is counted exactly once, at whichever stage
it finishes (skipped/failed in read, failed in process, or written). The monitor
loop ends when the count reaches the item total, so subclasses must let every
item reach a terminal stage.
"""

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from queue import Queue
from typing import Generic, TypeVar

from diffusion_trainer.shared import get_progress, logger

TItem = TypeVar("TItem")
TLoaded = TypeVar("TLoaded")
TPayload = TypeVar("TPayload")


class ThreadedPipelineProcessor(ABC, Generic[TItem, TLoaded, TPayload]):
    """Three-stage (read/process/write) threaded pipeline over a list of items."""

    def __init__(
        self,
        *,
        num_reader: int,
        num_writer: int,
        num_process_workers: int,
        description: str = "Processing...",
        poll_interval: float = 0.05,
        stall_ticks: int = 200,
        initial_completed: int = 0,
        enqueue_batch_size: int | None = None,
        enqueue_batch_pause: float = 0.0,
    ) -> None:
        self.num_reader = num_reader
        self.num_writer = num_writer
        self.num_process_workers = num_process_workers
        self.description = description
        self.poll_interval = poll_interval
        self.stall_ticks = stall_ticks
        self.initial_completed = initial_completed
        self.enqueue_batch_size = enqueue_batch_size
        self.enqueue_batch_pause = enqueue_batch_pause

        self.read_queue: Queue[TItem | None] = Queue(maxsize=max(num_reader, 1) * 2)
        self.process_queue: Queue[TLoaded | None] = Queue(maxsize=max(num_process_workers, 1))
        self.write_queue: Queue[TPayload | None] = Queue(maxsize=max(num_writer, 1))

        self.progress_counter = 0
        self.progress_lock = threading.Lock()
        self.progress_callback: Callable[[int], None] | None = None

    # ----- per-item stages implemented by subclasses -----

    @abstractmethod
    def get_items(self) -> list[TItem]:
        """Return the full list of items to process."""

    @abstractmethod
    def read_item(self, item: TItem) -> TLoaded | None:
        """Load an item for processing. Return None to finish it now (skip/handled error)."""

    @abstractmethod
    def process_item(self, worker: object, loaded: TLoaded) -> TPayload | None:
        """Transform a loaded item into a write payload. Return None on a handled error."""

    @abstractmethod
    def write_item(self, payload: TPayload) -> None:
        """Persist a payload. Should handle/log its own errors; the item is counted regardless."""

    def make_process_worker(self, index: int) -> object:  # noqa: ARG002
        """Return the worker object passed to ``process_item`` for process thread ``index``.

        Defaults to ``None``; override when each process thread needs its own model
        (e.g. a per-GPU VAE or tagger).
        """
        return None

    # ----- shared orchestration -----

    def _bump(self) -> None:
        with self.progress_lock:
            self.progress_counter += 1
            if self.progress_callback:
                self.progress_callback(self.progress_counter)

    def _reader_loop(self) -> None:
        while True:
            item = self.read_queue.get()
            if item is None:
                break
            try:
                loaded = self.read_item(item)
            except Exception:
                logger.exception("Reader failed for %s", item)
                self._bump()
                continue
            if loaded is None:
                self._bump()
            else:
                self.process_queue.put(loaded)

    def _process_loop(self, worker: object) -> None:
        while True:
            loaded = self.process_queue.get()
            if loaded is None:
                break
            try:
                payload = self.process_item(worker, loaded)
            except Exception:
                logger.exception("Processor failed")
                self._bump()
                continue
            if payload is None:
                self._bump()
            else:
                self.write_queue.put(payload)

    def _writer_loop(self) -> None:
        while True:
            payload = self.write_queue.get()
            if payload is None:
                break
            try:
                self.write_item(payload)
            except Exception:
                logger.exception("Writer failed")
            finally:
                self._bump()

    def run(self) -> None:
        items = self.get_items()
        total = len(items)
        if total == 0:
            logger.info("No items to process")
            return
        logger.info("Found %d items to process", total)

        workers = [self.make_process_worker(i) for i in range(self.num_process_workers)]
        reader_threads = [threading.Thread(target=self._reader_loop, daemon=True) for _ in range(self.num_reader)]
        process_threads = [threading.Thread(target=self._process_loop, args=(worker,), daemon=True) for worker in workers]
        writer_threads = [threading.Thread(target=self._writer_loop, daemon=True) for _ in range(self.num_writer)]
        all_threads = reader_threads + process_threads + writer_threads
        for thread in all_threads:
            thread.start()

        with get_progress() as progress:
            task = progress.add_task(self.description, total=total, completed=self.initial_completed)
            self.progress_callback = lambda count: progress.update(task, completed=count)

            self._enqueue_items(items)

            last_count = self.progress_counter
            stall_count = 0
            while self.progress_counter < total:
                current_count = self.progress_counter
                if current_count != last_count:
                    progress.update(task, completed=current_count)
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count > self.stall_ticks:
                        logger.warning("Processing seems stalled at %d/%d", current_count, total)
                        stall_count = 0
                last_count = current_count
                time.sleep(self.poll_interval)

            self._shutdown(reader_threads, process_threads, writer_threads)
            progress.update(task, completed=total)

        logger.info("Successfully processed %d items", total)

    def _enqueue_items(self, items: list[TItem]) -> None:
        if self.enqueue_batch_size:
            for i in range(0, len(items), self.enqueue_batch_size):
                for item in items[i : i + self.enqueue_batch_size]:
                    self.read_queue.put(item)
                if i > 0 and self.enqueue_batch_pause > 0:
                    time.sleep(self.enqueue_batch_pause)
        else:
            for item in items:
                self.read_queue.put(item)

    def _shutdown(
        self,
        reader_threads: list[threading.Thread],
        process_threads: list[threading.Thread],
        writer_threads: list[threading.Thread],
    ) -> None:
        for _ in reader_threads:
            self.read_queue.put(None)
        for _ in process_threads:
            self.process_queue.put(None)
        for _ in writer_threads:
            self.write_queue.put(None)
        for thread in reader_threads + process_threads + writer_threads:
            thread.join()
