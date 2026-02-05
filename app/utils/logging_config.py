import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "pathname": record.pathname,
            "funcName": record.funcName,
            "lineno": record.lineno,
        }

        # extra fields
        for key, value in record.__dict__.items():
            if key in ("name","msg","args","levelname","levelno","pathname","filename",
                       "module","exc_info","exc_text","stack_info","lineno","funcName",
                       "created","msecs","relativeCreated","thread","threadName","processName","process"):
                continue
            try:
                json.dumps(value)
                log_record[key] = value
            except Exception:
                log_record[key] = str(value)

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record, ensure_ascii=False)


def setup_logging(level: int = logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:
        root.handlers = []

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)

logger = logging.getLogger(__name__)
