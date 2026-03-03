"""Logging helpers for application, token usage, database activity, and RAG retrieval."""

import contextvars
import logging
import logging.config
from pathlib import Path
from typing import Any

import litellm

from src.core import settings

litellm.set_verbose = False

logging.getLogger("LiteLLM").setLevel(logging.WARNING)

_current_user_id = contextvars.ContextVar("logger_user_id", default="__unknown__")
_current_session_id = contextvars.ContextVar("logger_session_id", default="__unknown__")


class ContextFilter(logging.Filter):
    """Inject request context fields into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach request context fields to the log record."""
        if not hasattr(record, "user_id"):
            record.user_id = _current_user_id.get()
        if not hasattr(record, "session_id"):
            record.session_id = _current_session_id.get()
        return True


class NewlineSanitizerFilter(logging.Filter):
    """Escape newline characters to keep logs single-line for Loki/Grafana."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Replace literal newlines in message and args with escaped versions."""
        if isinstance(record.msg, str):
            record.msg = record.msg.replace("\r\n", "\\n").replace("\n", "\\n")
        if record.args:
            record.args = tuple(
                (
                    arg.replace("\r\n", "\\n").replace("\n", "\\n")
                    if isinstance(arg, str)
                    else arg
                )
                for arg in record.args
            )
        return True


class Logger:
    """Centralised logger factory and convenience wrappers for structured logging."""

    _logger: logging.Logger | None = None
    _supervisor_logger: logging.Logger | None = None
    _rag_logger: logging.Logger | None = None
    _configured: bool = False

    @classmethod
    def _ensure_configured(cls) -> None:
        """Configure logging once using dictConfig from settings."""
        if cls._configured:
            return
        config_dict = settings.logging_config
        for handler in config_dict.get("handlers", {}).values():
            filename = handler.get("filename")
            if not filename:
                continue
            log_path = Path(filename)
            if not log_path.is_absolute():
                log_path = Path(__file__).resolve().parents[3] / log_path
                handler["filename"] = str(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.config.dictConfig(config_dict)
        context_filter = ContextFilter()
        newline_filter = NewlineSanitizerFilter()

        def _attach_filters(logger: logging.Logger) -> None:
            logger.addFilter(context_filter)
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.addFilter(newline_filter)

        for logger_name in (
            "",
            "supervisor",
            "rag_retriever",
        ):
            _attach_filters(logging.getLogger(logger_name))
        cls._configured = True

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """Return the active main logger, creating it if needed."""
        if cls._logger is None:
            cls._ensure_configured()
            cls._logger = logging.getLogger()
        return cls._logger

    @classmethod
    def get_supervisor_logger(cls) -> logging.Logger:
        """Return the active supervisor logger, creating it if needed."""
        if cls._supervisor_logger is None:
            cls._ensure_configured()
            cls._supervisor_logger = logging.getLogger("supervisor")
        return cls._supervisor_logger

    @classmethod
    def get_rag_logger(cls) -> logging.Logger:
        """Return the active RAG retriever logger, creating it if needed."""
        if cls._rag_logger is None:
            cls._ensure_configured()
            cls._rag_logger = logging.getLogger("rag_retriever")
        return cls._rag_logger

    @classmethod
    def info(cls, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an informational message via the main application logger."""
        cls.get_logger().info(message, *args, **kwargs)

    @classmethod
    def warn(cls, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message via the main application logger."""
        cls.get_logger().warning(message, *args, **kwargs)

    @classmethod
    def error(cls, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message via the main application logger."""
        cls.get_logger().error(message, *args, **kwargs)

    @classmethod
    def debug(cls, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message via the main application logger."""
        cls.get_logger().debug(message, *args, **kwargs)

    @classmethod
    def set_logger(cls, logger: logging.Logger) -> None:
        """Replace the managed logger instance (useful for testing)."""
        cls._logger = logger

    @classmethod
    def bind_context(cls, *, user_id: str, session_id: str) -> None:
        """Bind request context for structured logging."""
        _current_user_id.set(user_id or "__unknown__")
        _current_session_id.set(session_id or "__unknown__")

    @classmethod
    def clear_context(cls) -> None:
        """Clear request context back to defaults."""
        _current_user_id.set("__unknown__")
        _current_session_id.set("__unknown__")


def get_logger() -> logging.Logger:
    """Facilitate existing call sites expecting a module-level helper."""
    return Logger.get_logger()
