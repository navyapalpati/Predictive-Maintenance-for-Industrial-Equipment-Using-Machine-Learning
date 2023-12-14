import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load your dataset
df = pd.read_csv("predictive_maintenance.csv")

# Set the default index and convert it to datetime
df.index = pd.to_datetime(df.index, format='%Y-%m-%d')  # Adjust the format accordingly

# Assuming 'Toolwear' is the variable to predict
time_series_data = df['Toolwear[min]']

# Plot the time series data
time_series_data.plot(figsize=(12, 6))
plt.title('Toolwear Over Time')
plt.xlabel('Time')
plt.ylabel('Toolwear[min]')
plt.show()

# Fit ARIMA model
order = (5, 1, 0)  # You may need to tune these parameters based on your data
model = ARIMA(time_series_data, order=order)
results = model.fit()

# Forecast future values
forecast_steps = 10  # You can adjust this based on your needs
forecast = results.get_forecast(steps=forecast_steps)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(time_series_data, label='Observed')
plt.plot(forecast.predicted_mean, color='red', label='Forecast')
plt.fill_between(forecast.index,
                 forecast.conf_int()[:, 0],
                 forecast.conf_int()[:, 1], color='red', alpha=0.2)

plt.title('ARIMA Forecast for Toolwear')
plt.xlabel('Time')
plt.ylabel('Toolwear[min]')
plt.legend()
plt.show()
