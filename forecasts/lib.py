from pathlib import Path

import pandas as pd
from statsmodels.tsa import stattools
from scipy.stats import rv_continuous

datadir = Path('..') / 'data'


def adf_test(timeseries):
    """
    https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
    """
    print("Results of Dickey-Fuller Test:")
    dftest = stattools.adfuller(timeseries.dropna(), autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)
    
def kpss_test(timeseries):
    """
    https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
    """
    print("Results of KPSS Test:")
    kpsstest = stattools.kpss(timeseries.dropna(), regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)

def boundary_probs(dist: rv_continuous, boundaries: list[int]):
    forecast = {f"Less than ${boundaries[0]}": dist.cdf(boundaries[0])}
    for low, high in zip(boundaries, boundaries[1:]):
        forecast[f"More than or equal to ${low} but less than ${high}"] = dist.cdf(high) - dist.cdf(low)
    forecast[f"More than or equal to ${boundaries[-1]}"] = 1 - dist.cdf(boundaries[-1])

    return {k: f'{round(100*v)}%' for k, v in forecast.items()}