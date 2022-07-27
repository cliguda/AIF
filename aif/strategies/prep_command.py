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

from dataclasses import dataclass
from enum import Enum, auto

import pandas as pd

import aif.common.logging as logging


class Command(Enum):
    """
    Some strategies need some preparation, before the actual rule for entry or exit signals can be applied.
    Commands:
    SHIFT:  The most common command is SHIFT, that can be used to SHIFT previous entries (e.g. SHIFT on COLUMN EMA_20
            creates a new column EMA_20_SHIFT_1 were each row contains the previous value of the EMA_20. Shifting
            the EMA_20 and the column Close allows to check, if the previous EMA_20 was below the closing price and
            the current EMA_20 is above the closing price. See the strategies in library for more examples)
    MARK:   The MARK command can be used, to mark certain signals, before the strategy is applied. E.g. entries
            with and RSI < 30 can be marked with 1 and entries with RSI > 70 can be marked with -1. Afterwards the
            FFILL COMMAND can be used, to propagate the marks. Thereby the strategy can check, if a signal is appearing
            after the RSI was below 30.
    FFILL:  Forward propagate for all NA values in the given column. Normally used after marking signals with MARK.
    """
    SHIFT = "SHIFT"
    MARK = "MARK"
    FFILL = "FFILL"


@dataclass
class CommandDescription:
    """The class describes one command with its arguments. Furthermore the command can be applied by using the
    apply_command method."""
    command: Command
    args: dict[str, str]

    def apply_command(self, df: pd.DataFrame) -> list[str]:
        """This method applies a command to a Dataframe and returns the name of the new columns"""
        new_columns = []

        if self.command == Command.SHIFT:
            column_name = self.args.get('COLUMN')
            intervals = self.args.get('INTERVALS', 1)
            new_column_name = f'{column_name}_Shift_{intervals}'

            logging.get_aif_logger(__name__).debug(f'Shifted column {column_name} on {intervals} (-> {new_column_name}).')

            df.loc[:, new_column_name] = df[column_name].shift(int(intervals))
            new_columns.append(new_column_name)
        elif self.command == Command.MARK:
            expression = self.args.get('EXPR')
            new_column_name = self.args.get('NEW_COLUMN')
            value = self.args.get('VALUE')

            logging.get_aif_logger(__name__).debug(f'Mark {expression} with {value} (-> {new_column_name}).')

            df.loc[df.eval(expression), new_column_name] = value
            new_columns.append(new_column_name)
        elif self.command == Command.FFILL:
            column = self.args.get('COLUMN')

            logging.get_aif_logger(__name__).debug(f'FFILL {column}')

            df.loc[:, column] = df[column].ffill()
        else:
            raise ValueError(f'Cannot apply invalid command: {self.command.name}')

        return new_columns
