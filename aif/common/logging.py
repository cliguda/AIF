"""
AIF - Artificial Intelligence for Finance
Copyright (C) 2022 Christian Liguda

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import logging
from pushsafer import Client

from aif.common.config import settings

__logger_initialized = False

FORMAT_MSG = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
FORMAT_DATE = '%Y-%m-%d %H:%M:%S'


class PushsaferHandler(logging.StreamHandler):

    def __init__(self):
        super().__init__()
        self.client = Client(settings.pushsafer.api_key)

    def emit(self, record):
        msg = self.format(record)
        self.client.send_message(message=msg, title='Alert from AIF', icon=4)


def get_aif_logger(name: str):
    """ Create a custom logger """

    # Add logging level ALERT if called the first time.
    global __logger_initialized
    if not __logger_initialized:
        # Set root logger
        filename = f'{settings.common.project_path}{settings.common.log_root_filename}'
        logging.basicConfig(format=FORMAT_MSG, datefmt=FORMAT_DATE, level=logging.INFO, filename=filename)

        # Adding new level TRACE for very detailed debugging messages (e.g. individual trades in the evaluation process)
        logging.addLevelName(5, 'TRACE')
        setattr(logging, 'TRACE', 5)

        # Adding a logging method for the TRACE level
        def log_trace(self, message, *args, **kws):
            if self.isEnabledFor(5):
                # Yes, logger takes its '*args' as 'args'.
                self._log(5, message, args, **kws)

        logging.Logger.trace = log_trace

        # Adding a new level ACTION for actions that are initiated by the program.
        logging.addLevelName(100, 'ACTION')
        setattr(logging, 'ACTION', 100)

        # Adding a logging method for the ACTION level
        def log_action(self, message, *args, **kws):
            if self.isEnabledFor(100):
                # Yes, logger takes its '*args' as 'args'.
                self._log(100, message, args, **kws)

        logging.Logger.action = log_action

        __logger_initialized = True

    logger = logging.getLogger(name)
    if len(logger.handlers) == 0:
        logger.propagate = False
        logger.setLevel(logging.TRACE)

        # Create a formatter that is used by all handlers TODO: Convert to UTC?
        formatter = logging.Formatter(FORMAT_MSG, datefmt=FORMAT_DATE)

        # Create handlers
        # Log everything from level DEBUG to file
        debug_path = f'{settings.common.project_path}{settings.common.log_debug_filename}'
        debug_handler = logging.FileHandler(debug_path)
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(formatter)
        logger.addHandler(debug_handler)

        # Everything from level ALERT is also logged separately
        alert_path = f'{settings.common.project_path}{settings.common.log_alert_filename}'
        alert_handler = logging.FileHandler(alert_path)
        alert_handler.setLevel(logging.WARNING)
        alert_handler.setFormatter(formatter)
        logger.addHandler(alert_handler)

        if len(settings.pushsafer.api_key) > 0:
            # Add a handler to send all warnings, errors and actions via push-notification
            push_notification_handler = PushsaferHandler()
            push_notification_handler.setLevel(logging.WARNING)
            push_notification_handler.setFormatter(formatter)
            logger.addHandler(push_notification_handler)

        # Adding a handler for the logging on the console
        terminal_output_handler = logging.StreamHandler()
        if settings.common.log_console_level == 'debug':
            terminal_output_handler.setLevel(logging.DEBUG)
        elif settings.common.log_console_level == 'info':
            terminal_output_handler.setLevel(logging.INFO)

        terminal_output_handler.setFormatter(formatter)
        logger.addHandler(terminal_output_handler)

    return logger
