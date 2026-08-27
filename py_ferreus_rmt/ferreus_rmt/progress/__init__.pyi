from typing import Callable, Optional, Union

class IsosurfaceProgress:
    """
    Event indicating progress for isosurface extraction.

    Attributes
    ----------
    isovalue : float
        Isovalue currently being surfaced.
    stage : str
        Human-readable extraction stage.
    progress : float
        Fraction in ``[0, 1]`` indicating overall progress.
    """
    isovalue: float
    stage: str
    progress: float

class Message:
    """
    Arbitrary informational message.

    Attributes
    ----------
    message : str
        The message text.
    """
    message: str

ProgressEvent = Union[IsosurfaceProgress, Message]
"""Union of all progress event payloads passed to :class:`Progress` callbacks."""

ProgressCallback = Callable[[ProgressEvent], None]
"""Callable accepting one :data:`ProgressEvent` and returning ``None``."""

class Progress:
    """
    Wrapper for progress event reporting.

    Parameters
    ----------
    callback : ProgressCallback, optional
        Function invoked with each :data:`ProgressEvent`.
    """
    def __init__(
        self,
        callback: Optional[ProgressCallback] = None,
    ) -> None: ...
