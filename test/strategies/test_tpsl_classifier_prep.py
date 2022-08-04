import numpy as np

import aif.strategies.library.tpsl_classifier_preperation as prep


def test__evaluate_long_group():
    # Long no SL Hit
    data = np.array([
               [0, 0, 0, 0, 0, 0],                      # Open
               [1000, 1021, 1035, 1045, 1070, 1109],    # High
               [990, 979, 915, 950, 951, 940],          # Low
               [1000, 1018, 1030, 1020, 1050, 1060],    # Close
            ], dtype=float).transpose()

    x = prep._evaluate_long_group(m=data, sl_threshold=0.1, tp_threshold=0.15)
    assert x == 0

    x = prep._evaluate_long_group(m=data, sl_threshold=0.1, tp_threshold=0.1)
    assert x == 1

    # Long SL Hit
    data = np.array([
               [0, 0, 0, 0, 0, 0],                      # Open
               [1000, 1121, 1035, 1045, 1070, 1109],    # High
               [990, 979, 915, 950, 951, 940],          # Low
               [1000, 1018, 1030, 1020, 1050, 1060],    # Close
            ], dtype=float).transpose()

    x = prep._evaluate_long_group(m=data, sl_threshold=0.08, tp_threshold=0.15)
    assert x == 0

    x = prep._evaluate_long_group(m=data, sl_threshold=0.08, tp_threshold=0.12)
    assert x == 1


def test__evaluate_short_group():
    # Long no SL Hit
    data = np.array([
               [0, 0, 0, 0, 0, 0],                      # Open
               [1000, 1021, 1035, 1045, 1070, 1098],    # High
               [990, 979, 915, 879, 951, 940],          # Low
               [1000, 1018, 1030, 1020, 1050, 1060],    # Close
            ], dtype=float).transpose()

    x = prep._evaluate_short_group(m=data, sl_threshold=0.1, tp_threshold=0.15)
    assert x == 0

    x = prep._evaluate_short_group(m=data, sl_threshold=0.1, tp_threshold=0.12)
    assert x == 1

    # Long SL Hit
    data = np.array([
               [0, 0, 0, 0, 0, 0],                      # Open
               [1000, 1021, 1101, 1045, 1070, 1098],    # High
               [990, 879, 915, 879, 951, 940],          # Low
               [1000, 1018, 1030, 1020, 1050, 1060],    # Close
            ], dtype=float).transpose()

    x = prep._evaluate_short_group(m=data, sl_threshold=0.1, tp_threshold=0.15)
    assert x == 0

    x = prep._evaluate_short_group(m=data, sl_threshold=0.1, tp_threshold=0.12)
    assert x == 1