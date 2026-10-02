import logging
from typing import Optional

from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
access_log = logging.getLogger("Walkplan.access")



if __name__ == "__main__":
    