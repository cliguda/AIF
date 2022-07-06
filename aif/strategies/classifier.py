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

from typing import Protocol, runtime_checkable
from numpy import ndarray


@runtime_checkable
class Classifier(Protocol):
    """This protocol is used to give ML-classifiers and other custom classifiers a common predict interface. It can
    be used as return type for different classifier builder etc."""

    def fit(self, X, y, **fit_params) -> None:
        ...

    def predict(self, X, **predict_params) -> ndarray:
        ...

    def set_params(self, **params: dict) -> None:
        ...
