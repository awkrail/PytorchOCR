import os
import sys
import logging


def print_dict(d, logger, delimiter=0):
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            print_dict(v, logger, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            for value in v:
                print_dict(value, logger, delimiter + 4)
        else:
            logger.info("{}{} : {}".format(delimiter * " ", k, v))


def get_logger(
    name="pytorchocr",
    log_file=None,
    log_level=logging.DEBUG,
    log_ranks="0"):
    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt = "%Y/%m/%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file_folder = os.path.split(log_file)[0]
        os.makedirs(log_file_folder, exist_ok=True)
        file_handler = logging.FileHandler(log_file, "a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)
    return logger
