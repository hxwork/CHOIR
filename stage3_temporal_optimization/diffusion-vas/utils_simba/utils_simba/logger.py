import logging
def fetch_logger(logger_name="", logger_file=""):
    # Create a custom logger
    if logger_name != "":
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()

    # Set the logging level
    logger.setLevel(logging.DEBUG)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s')
    console_handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(console_handler)    


    if logger_file != "":
            # Create a file handler
        file_handler = logging.FileHandler(logger_file)
        file_handler.setLevel(logging.DEBUG)    
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger