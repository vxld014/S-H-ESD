import pandas as pd
import numpy as np
from scipy.stats import t
from matplotlib import pyplot

def seasonal_mean(x, freq):
  """
  Returns the mean of the timeseries for each period in x.
  
  x    : List, array, or series containing the time series
  freq : Int that gives the number of periods per cycle (7 for week, 12 for monthly, etc)
  """
  return np.array([pd.Series(x[i::freq]).mean(skipna=True) for i in range(freq)])


def ts_S_Md_decomposition(x, freq):
  """
  Decomposes the timeseries using a modified STL method:Rx = X - Sx - X^~.
    Rx : Residuals
    X  : Original time series
    Sx : Seasonality component (average value per period)
    X^~: Median of original timeseries
    
  x    : List, array, or series containing the time series 
  freq : Int that gives the number of periods per cycle (7 for week, 12 for monthly, etc)  
  """
  nobs = len(x)
  
  # Seasonality
  period_averages = seasonal_mean(x, freq)
  seasonal = np.tile(period_averages, nobs // freq + 1)[:nobs]
  
  # Median
  med = np.tile(pd.Series(x).median(skipna=True), nobs)
  
  # Residuals
  res = np.array(x) - seasonal - med
  
  return {"observed": np.array(x), "seasonal": seasonal, "median":med, "residual":res}


def ts_decomposition_plot(x):
  """
  Plots the timeseries decomposition for x.
  """
  fig, ax = pyplot.subplots(4, 1, sharex=True) # 4 graphs arranged in rows
  ax[0].plot(x["observed"])
  ax[0].set_ylabel("Observed")
  ax[1].plot(x["median"])
  ax[1].set_ylabel("Median")
  ax[2].plot(x["seasonal"])
  ax[2].set_ylabel("Seasonal")
  ax[3].plot(x["residual"])
  ax[3].set_ylabel("Residual")
  display(pyplot.show())
  pyplot.gcf().clear()  
  
  
def esd_test_statistics(x, hybrid=True):
  """
  Compute the location and dispersion sample statistics used to carry out the ESD test.
  """
  if hybrid:
    location = pd.Series(x).median(skipna=True) # Median
    dispersion = np.median(np.abs(x - np.median(x))) # Median Absolute Deviation
  else:  
    location = pd.Series(x).mean(skipna=True) # Mean
    dispersion = pd.Series(x).std(skipna=True) # Standard Deviation
    
  return location, dispersion    


def esd_test(x, freq, alpha=0.95, ub=0.499, hybrid=True):
  """
  Carries out the Extreme Studentized Deviate(ESD) test which can be used to detect one or more outliers present in the timeseries
  
  x      : List, array, or series containing the time series
  freq   : Int that gives the number of periods per cycle (7 for week, 12 for monthly, etc)
  alpha  : Confidence level in detecting outliers
  ub     : Upper bound on the fraction of datapoints which can be labeled as outliers (<=0.499)
  hybrid : Whether to use the robust statistics (median, median absolute error) or the non-robust versions (mean, standard deviation) to test for anomalies
  """
  nobs = len(x)
  if ub > 0.4999:
    ub = 0.499
  k = max(int(np.floor(ub * nobs)), 1) # Maximum number of anomalies. At least 1 anomaly must be tested.
  res_tmp = ts_S_Md_decomposition(x, freq)["residual"] # Residuals from time series decomposition
    
  # Carry out the esd test k times  
  res = np.ma.array(res_tmp, mask=False) # The "ma" structure allows masking of values to exclude the elements from any calculation
  anomalies = [] # returns the indices of the found anomalies
  for i in range(1, k+1):
    location, dispersion = esd_test_statistics(res, hybrid) # Sample statistics
    tmp = np.abs(res - location) / dispersion
    idx = np.argmax(tmp) # Index of the test statistic
    test_statistic = tmp[idx] 
    n = nobs - res.mask.sum() # sums non masked values
    critical_value = (n - i) * t.ppf(alpha, n - i - 1) / np.sqrt((n - i - 1 + np.power(t.ppf(alpha, n - i - 1), 2)) * (n - i - 1)) 
    if test_statistic > critical_value:
      anomalies.append(idx)
    res.mask[idx] = True  
    
  return anomalies