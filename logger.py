"""
logger.py — настройка логирования

Двойной вывод: файл + очередь для GUI-виджета.
Импортировать этот модуль первым, до всех остальных модулей проекта.
"""

import queue
import logging

# Очередь для передачи строк лога в GUI-текстовый виджет
log_queue: queue.Queue = queue.Queue()


class _QueueHandler(logging.Handler):
    """Перенаправляет записи лога в очередь для GUI."""
    def emit(self, record: logging.LogRecord) -> None:
        log_queue.put(self.format(record))


def setup_logging(log_file: str = "rc_optimiser.log") -> logging.Logger:
    """
    Настроить логгер 'rc_opt' с тремя обработчиками:
      1. FileHandler  — запись в файл
      2. StreamHandler — вывод в stderr (консоль)
      3. _QueueHandler — передача строк в GUI

    Параметры
    ---------
    log_file : str
        Начальный путь лог-файла. Можно переопределить позже из worker.py.

    Возвращает
    ----------
    logging.Logger
    """
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logger = logging.getLogger("rc_opt")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(sh)

    qh = _QueueHandler()
    qh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(qh)

    return logger
