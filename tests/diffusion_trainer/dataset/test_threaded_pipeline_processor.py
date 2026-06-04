"""Orchestration tests for ThreadedPipelineProcessor.

These exercise the threading skeleton (counting, skip/error paths, sentinel
shutdown, termination) with fake stages, independent of any GPU work.
"""

import threading

from diffusion_trainer.dataset.processors.base import ThreadedPipelineProcessor


class _FakeProcessor(ThreadedPipelineProcessor[int, int, int]):
    """Doubles each item; items divisible by `skip_mod` skip, by `err_mod` error."""

    def __init__(self, items: list[int], *, skip_mod: int = 0, err_mod: int = 0, **kwargs: object) -> None:
        super().__init__(num_reader=2, num_writer=2, num_process_workers=2, poll_interval=0.001, **kwargs)  # type: ignore[arg-type]
        self._items = items
        self._skip_mod = skip_mod
        self._err_mod = err_mod
        self.written: list[int] = []
        self._write_lock = threading.Lock()

    def get_items(self) -> list[int]:
        return self._items

    def read_item(self, item: int) -> int | None:
        if self._skip_mod and item % self._skip_mod == 0:
            return None  # skip, counted by base
        return item

    def process_item(self, worker: object, loaded: int) -> int | None:  # noqa: ARG002  # override must keep the base signature
        if self._err_mod and loaded % self._err_mod == 0:
            return None  # handled error, counted by base
        return loaded * 2

    def write_item(self, payload: int) -> None:
        with self._write_lock:
            self.written.append(payload)


def test_all_items_written_and_counted() -> None:
    proc = _FakeProcessor(list(range(1, 51)))
    proc.run()
    assert proc.progress_counter == 50
    assert sorted(proc.written) == [i * 2 for i in range(1, 51)]


def test_skip_path_still_counts_but_not_written() -> None:
    proc = _FakeProcessor(list(range(1, 21)), skip_mod=5)  # 5,10,15,20 skipped
    proc.run()
    assert proc.progress_counter == 20  # every item counted once
    assert all(v % 10 != 0 for v in proc.written)  # skipped items never written
    assert len(proc.written) == 16


def test_error_path_still_counts() -> None:
    proc = _FakeProcessor(list(range(1, 21)), err_mod=4)  # 4,8,... error in process
    proc.run()
    assert proc.progress_counter == 20
    assert len(proc.written) == 15  # 5 errored (4,8,12,16,20)


def test_empty_items_returns_immediately() -> None:
    proc = _FakeProcessor([])
    proc.run()
    assert proc.progress_counter == 0
    assert proc.written == []


def test_no_thread_leak_after_run() -> None:
    before = threading.active_count()
    proc = _FakeProcessor(list(range(1, 31)))
    proc.run()
    # sentinels must drain every worker thread
    assert threading.active_count() == before


def test_per_worker_object_used() -> None:
    class _WorkerProcessor(_FakeProcessor):
        def make_process_worker(self, index: int) -> object:
            return f"worker-{index}"

        def process_item(self, worker: object, loaded: int) -> int | None:
            assert isinstance(worker, str)
            assert worker.startswith("worker-")
            return loaded * 2

    proc = _WorkerProcessor(list(range(1, 11)))
    proc.run()
    assert proc.progress_counter == 10
