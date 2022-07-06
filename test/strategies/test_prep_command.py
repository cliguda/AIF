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

import numpy as np
import pandas as pd

from aif.strategies.prep_command import Command, CommandDescription


def test_command_shift():
    """Testing the Shift command and the apply command method."""
    # Create data
    df = pd.DataFrame({'X': [1, 2, 3, 4, 5, 6], 'Y': [2, 3, 2.5, 3, 6, 7]})

    # Define rule
    preprocessor = [
        CommandDescription(Command.SHIFT, {'COLUMN': 'X', 'INTERVALS': 1}),
        CommandDescription(Command.SHIFT, {'COLUMN': 'X', 'INTERVALS': -1}),
        CommandDescription(Command.SHIFT, {'COLUMN': 'Y', 'INTERVALS': 2})
    ]

    new_columns = []
    for cmd in preprocessor:
        new_columns.extend(cmd.apply_command(df))

    assert new_columns == ['X_Shift_1', 'X_Shift_-1', 'Y_Shift_2']

    assert all(df['X_Shift_1'].dropna() == [1, 2, 3, 4, 5])
    assert len(df['X_Shift_1']) == len(df['X_Shift_1'].dropna()) + 1
    assert np.isnan(df.iloc[0]['X_Shift_1'])

    assert all(df['X_Shift_-1'].dropna() == [2, 3, 4, 5, 6])
    assert len(df['X_Shift_-1']) == len(df['X_Shift_-1'].dropna()) + 1
    assert np.isnan(df.iloc[-1]['X_Shift_-1'])

    assert all(df['Y_Shift_2'].dropna() == [2, 3, 2.5, 3])
    assert len(df['Y_Shift_2']) == len(df['Y_Shift_2'].dropna()) + 2
    assert np.isnan(df.iloc[0]['Y_Shift_2'])
    assert np.isnan(df.iloc[1]['Y_Shift_2'])
