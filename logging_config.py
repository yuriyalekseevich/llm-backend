import logging
import json


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Base log record
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "pathname": record.pathname,
            "funcName": record.funcName,
            "lineno": record.lineno,
        }

        # Include extra fields passed via `extra` kwarg
        for key, value in record.__dict__.items():
            if key in ("name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated", "thread", "threadName", "processName", "process"):
                continue
            try:
                json.dumps(value)  # ensure serializable
                log_record[key] = value
            except Exception:
                log_record[key] = str(value)

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger to emit JSON to stdout."""
    root = logging.getLogger()
    root.setLevel(level)

    # Avoid adding duplicate handlers on reload by clearing existing ones
    if root.handlers:
        root.handlers = []

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)


def get_logger(name: str = __name__) -> logging.Logger:
    return logging.getLogger(name)
