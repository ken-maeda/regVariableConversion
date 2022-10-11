import pytest
import pandas as pd
import numpy as np
from scipy.special import comb

import rvc
"""
pytest --cov --cov-report html ./
"""
def createData():
    df_test = pd.read_csv("rvc/tests/data_test.csv", index_col=0)
    return df_test

@pytest.fixture
def i_app():
    return rvc.RVC()

@pytest.fixture
def i_app_data():
    model = rvc.RVC()
    df_test = createData()
    model.df = df_test
    model.col_target = "EP"
    model.col_ex = list(model.df.columns)
    model.col_ex.remove(model.col_target)
    return model

def test_key2smf(i_app):
    i_app.col_target = "Y"
    i_app.col_ex0 = "X"
    form = i_app._key2smf(("1", "2", "3"))
    assert form == "Y ~ + X + np.square(X) + np.power(X, 3) "

    form, label = i_app._key2smf(("1", "2", "3"), returnLabel=True)
    assert label == "123"

def test_key2ind(i_app):
    res = i_app._key2ind("123")
    assert res == ['X', 'X2', 'X3']

def test_slidingFilter(i_app_data):
    i_app_data.df.iloc[:, 0] = i_app_data.df.iloc[:, 0] - 10
    _ = i_app_data._slidingFilter(i_app_data.df)
    assert len(i_app_data.GAP) == 4

def test_fit(i_app):
    df_test = createData()
    # i_app.COMB = 3  # fast caluculation
    i_app.fit(df_test.iloc[:, [0, -1]], "EP")

    # _cost
    SUM = 0
    for i in range(1, 1 + i_app.COMB):
        SUM += comb(len(i_app.KEY), i)
    i_app.df_cost.shape[1] == SUM
    assert i_app.df_cost.shape[1] == SUM

    # _evaluateOnLayer
    assert i_app.df_bestCosts.shape[1] == 1
    assert i_app.df_mse_bestCosts.shape[1] == 1

    # _decision
    assert len(i_app.res_key) == 1
    assert len(i_app.res_ind) == 1
    assert len(i_app.res_formula) == 1

    # _publish
    assert i_app.df_output.shape[1] == len(i_app.res_ind[0]) + 1

    # _combine
    i_app._combine()
    assert i_app.df_output.shape[1] == 2
