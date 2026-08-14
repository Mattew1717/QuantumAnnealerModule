import logging
import sys
from pathlib import Path
from datetime import datetime

class Logger:
    """Custom logger"""
    
    def __init__(self, log_dir=None):

        # 'IsingComparison' is a global singleton: never clear existing handlers
        # or a previously-attached file handler is silently lost.
        self.logger = logging.getLogger('IsingComparison')
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(message)s')

        has_stream = any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            for h in self.logger.handlers
        )
        if not has_stream:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.file_handler = next(
            (h for h in self.logger.handlers if isinstance(h, logging.FileHandler)),
            None,
        )
        if log_dir and self.file_handler is None:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)

            log_filename = log_dir_path / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            self.file_handler = logging.FileHandler(log_filename, mode='w')
            self.file_handler.setLevel(logging.INFO)
            self.file_handler.setFormatter(formatter)
            self.logger.addHandler(self.file_handler)
        
    
    def info(self, message):
        self.logger.info(message)
        self._flush()
    
    def warning(self, message):
        self.logger.warning(message)
        self._flush()
    
    def error(self, message):
        self.logger.error(message)
        self._flush()
    
    def _flush(self):
        """Flush all handlers"""
        for handler in self.logger.handlers:
            handler.flush()
        sys.stdout.flush()