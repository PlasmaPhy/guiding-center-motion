from loguru import logger


# Setup logger
logger.remove()

# Format templates
fmt = "{time:HH:mm:ss:SSS} | {function: <20} |  {level: ^7} | {message}"
# fmt = "{time:HH:mm:ss:SSS} | {name: <18} |  {level: >6} | {message}"
level = "DEBUG"

logger.add("usr/log.txt", delay=True, level=level, format=fmt, mode="w")
logger.info(f"Logger added on {level} level.\n")
logger.info("---------------------------------------------------")
