"""Logging helpers shared across OpenEvolve.

Provides :class:`LoggerPrefixFilter`, a handler-level filter that rewrites
``record.name`` so every downstream module logger appears under a common
prefix. This lets users tell apart log output coming from multiple concurrent
``OpenEvolve`` runs (GitHub issue #290).
"""

import logging
from typing import Optional


class LoggerPrefixFilter(logging.Filter):
    """Rewrite ``record.name`` to live under a configured prefix.

    When installed on a handler, every record emitted through that handler is
    renamed from e.g. ``openevolve.controller`` to
    ``<prefix>.openevolve.controller``. Formatters that include ``%(name)s``
    then show the prefixed name, which makes concurrent runs distinguishable
    and greppable in shared log output.

    The filter is idempotent: names that already carry the prefix are left
    untouched, so a record flowing through several handlers that share the
    same filter instance is only prefixed once.
    """

    def __init__(self, prefix: Optional[str] = None) -> None:
        super().__init__()
        self.prefix = prefix or ""

    def filter(self, record: logging.LogRecord) -> bool:
        if self.prefix and not record.name.startswith(self.prefix + "."):
            record.name = f"{self.prefix}.{record.name}"
        return True


def console_handler(logger_prefix: Optional[str]) -> logging.Handler:
    """The console (stderr) handler of an OpenEvolve run's processes.

    The format names the logger (with ``logger_prefix``, through
    :class:`LoggerPrefixFilter`) only when a prefix is set; the default format is
    unchanged otherwise.
    """
    handler = logging.StreamHandler()
    console_format = "%(asctime)s - %(levelname)s - %(message)s"
    if logger_prefix:
        # Include the (prefixed) logger name so console output is
        # distinguishable across runs as well.
        console_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        handler.addFilter(LoggerPrefixFilter(logger_prefix))
    handler.setFormatter(logging.Formatter(console_format))
    return handler
