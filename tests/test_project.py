"""Regression, chronological-split and model smoke tests using synthetic data."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from utils import FEATURES, TARGET, chronological_split, load_data, scores
from historical_average import fit_averages, clock_columns


def sample_data(n=160):
    times = pd.date_range('2024-05-01', periods=n, freq='15min', tz='UTC')
    x = np.arange(n)
    return pd.DataFrame({
        'collecteddate': times,
        'target_time': times + pd.Timedelta(minutes=15),
        'current_occupancy': x % 8,
        'previous_15min': (x-1) % 8,
        'previous_30min': (x-2) % 8,
        'hour': times.hour,
        'minute': times.minute,
        'day_of_week': times.dayofweek,
        TARGET: (x+1) % 8,
        'target_is_measured': True,
    })


def test_missing_columns_rejected(tmp_path):
    path=tmp_path/'bad.csv'
    pd.DataFrame({'headcount':[1]}).to_csv(path,index=False)
    with pytest.raises(ValueError,match='Missing required columns'):
        load_data(path)


def test_chronological_boundaries():
    train,val,test=chronological_split(sample_data())
    assert train['target_time'].max() < val['collecteddate'].min()
    assert val['target_time'].max() < test['collecteddate'].min()
    assert test['collecteddate'].is_monotonic_increasing


def test_mae_rmse_known_values():
    s=scores([0,2,4],[0,4,4])
    assert np.isclose(s['MAE'],2/3)
    assert np.isclose(s['RMSE'],np.sqrt(4/3))
    assert s['R2'] <= 1


@pytest.mark.parametrize('estimator',[
    DecisionTreeRegressor(max_depth=3,random_state=42),
    LinearRegression(),
    make_pipeline(StandardScaler(),SVR(C=1)),
])
def test_regression_models_fit_predict(estimator):
    train,val,test=chronological_split(sample_data())
    estimator.fit(train[FEATURES],train[TARGET])
    pred=estimator.predict(test[FEATURES])
    assert len(pred)==len(test)
    assert np.isfinite(pred).all()


def test_historical_average_uses_training_rows():
    train,val,test=chronological_split(sample_data())
    means, hourly, overall=fit_averages(train)
    assert np.isclose(overall,train[TARGET].mean())
    assert 0 <= overall <= 8
    assert len(means)>0


def test_load_data_csv(tmp_path):
    path=tmp_path/'processed.csv'
    sample_data().to_csv(path,index=False)
    loaded=load_data(path)
    assert len(loaded)==160
    assert 'target_is_measured' in loaded
