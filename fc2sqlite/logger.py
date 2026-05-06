import logging
logger = logging.getLogger("fc2sqlite")
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)
formatter = logging.Formatter(
    "{levelname} - {message}",
    style="{",
)
console_handler.setFormatter(formatter)
logger.propagate = False # this stops RootLogger from duplicating everything
logger.setLevel("WARNING")  # actually, this is the default anyway

