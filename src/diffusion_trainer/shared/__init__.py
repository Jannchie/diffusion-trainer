"""Module contains shared code for the diffusion_trainer package."""

import logging

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

logger = logging.getLogger("diffusion_trainer")


def get_progress() -> Progress:
    """Return a Progress object."""
    return Progress(
        SpinnerColumn(style="yellow"),
        TextColumn(" {task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TaskProgressColumn(),
    )
