# S-H-ESD
Timeseries Anomaly Detection using S-ESD and S-H-ESD

## Credit
All credit belongs to Jordan Hochenbaum, Owen S. Vallis, and Arun Kejariwal. This repo contains my implementation of the algorithm outlined in https://arxiv.org/pdf/1704.07706.pdf

## Usage
```python
import ts_anomaly_detection.py

# Synthetic "weekly" seasonality time-series
x = np.array([10, 20, 30, 40, 50, 60, 70, 10, 20, 30, 40, 50, 60, 70, 10, 20, 30, 40, 50, 60, 70, 10, 20, 30, 40, 50, 60, 70, 10, 20, 30, 40, 50, 60, 70])
# Synthetic anomalies
x[[0, 12, 17, 25, 30]] = 0

# Use the algorithm to find anomalies
anomalies = esd_test(x, 7, alpha=0.95, ub=0.499, hybrid=True)

# Plot the anomalies
fig, ax = pyplot.subplots()
ax.plot(pd.Series(x).index, x, color="blue", label = "Original")
ax.scatter(anomalies, x[anomalies], color='red', label='Anomaly')
pyplot.legend(loc="best")
pyplot.show()
```
