import logging
import sys

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))