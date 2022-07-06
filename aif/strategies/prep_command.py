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
    """
    SHIFT = "SHIFT"


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
        else:
            raise ValueError(f'Cannot apply invalid command: {self.command.name}')

        return new_columns
