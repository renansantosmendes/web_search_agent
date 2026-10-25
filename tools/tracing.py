"""Langfuse tracing of the model calls, the agent runs and the Tavily searches.

Tracing is on when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (and
LANGFUSE_HOST when the server is not the cloud default). Without the keys the
application still works, but nothing is recorded and a warning is logged.

Required environment variables:
    LANGFUSE_PUBLIC_KEY   public key of the Langfuse project
    LANGFUSE_SECRET_KEY   secret key of the Langfuse project
    LANGFUSE_HOST         optional; base URL of the Langfuse server
"""

import logging
import os
from typing import Any

from langfuse import get_client
from langfuse.langchain import CallbackHandler

REQUIRED_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")

logger = logging.getLogger(__name__)


def tracing_enabled() -> bool:
    """Tell whether the Langfuse keys are configured.

    Returns
    -------
    bool
        True when every key in ``REQUIRED_KEYS`` is set and not empty.

    Example
    -------
    >>> tracing_enabled()  # doctest: +SKIP
    True
    """
    enabled = all(os.environ.get(name) for name in REQUIRED_KEYS)
    if not enabled:
        logger.warning("Langfuse keys are missing; calls are not traced.")
    return enabled


def tracing_callbacks() -> list[Any]:
    """Build the callbacks that record a LangChain run in Langfuse.

    Returns
    -------
    list[Any]
        A list with one Langfuse callback handler, or an empty list when the
        tracing is not configured.

    Example
    -------
    >>> config = {"callbacks": tracing_callbacks()}  # doctest: +SKIP
    """
    if not tracing_enabled():
        return []
    return [CallbackHandler()]


def flush_tracing() -> None:
    """Send the traces still in memory to Langfuse.

    Serverless instances can stop right after the response, so the traces
    must be sent before the request returns. Does nothing without the keys.

    Example
    -------
    >>> flush_tracing()  # doctest: +SKIP
    """
    if tracing_enabled():
        get_client().flush()
