import logging
from pathlib import Path
from datetime import datetime

class Logger:
    """Custom logger that writes to both console and file."""
    
    def __init__(self, log_dir='logs', run_timestamp=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        if run_timestamp is None:
            run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_timestamp = run_timestamp
        
        log_file = self.log_dir / f'run_{run_timestamp}.log'
        
        # Configure logging
        self.logger = logging.getLogger('IsingComparison')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # File handler
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)