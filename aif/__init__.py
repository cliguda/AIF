import logging

from aif.common.config import settings

# Config root logger TODO: Convert to UTC by time.gmtime?
filename = f'{settings.common.project_path}{settings.common.log_root_filename}'
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO, filename=filename)
