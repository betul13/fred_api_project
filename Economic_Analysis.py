import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from decouple import config
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import seaborn as sns
from fredapi import Fred
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
import itertools
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings('ignore')

pd.set_option("display.max_columns", None)


fred_key = "6855d309ff62795129ab0d8417f76ce6"

secret_key = os.environ.get('fred-api')

fred = Fred(api_key = fred_key)

rsfs = fred.search("Retail Sales", order_by="popularity", sort_order="desc")


print(rsfs.head())

totalsa = fred.get_series(series_id="TOTALSA")



