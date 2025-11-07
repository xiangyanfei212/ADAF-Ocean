import os
import sys
import logging


def log_to_file_and_screen(logger_name=None, log_level=logging.INFO, log_file_path="train.log"):

    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))

    # Create logger and set level
    if logger_name is not None:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Define the format of log
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)  

    return logger
