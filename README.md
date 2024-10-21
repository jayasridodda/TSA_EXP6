## Reg no: 212222240028
## Developed By: DODDA JAYASRI
## Date: 

# Ex.No: 6               HOLT WINTERS METHOD

### AIM:
To create and implement Holt Winter's Method Model using python for Salesforcehistory dataset.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset and parse publication_date as datetime
data = pd.read_csv("Salesforcehistory.csv", parse_dates=['Date'])

# Set publication_date as the index
data.set_index('Date', inplace=True)

# Resample ratings_count to monthly frequency (sum of ratings per month)
monthly_data = data['Open'].resample('MS').sum()

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(monthly_data.values.reshape(-1, 1)).flatten(), 
                        index=monthly_data.index)

# Split into training and testing sets (80% train, 20% test)
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Fit the Holt-Winters additive model on training data
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12).fit()

# Forecast for the test data length
test_predictions_add = model_add.forecast(steps=len(test_data))

# Evaluate model performance on test data
mae = mean_absolute_error(test_data, test_predictions_add)
rmse = mean_squared_error(test_data, test_predictions_add, squared=False)
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Plot 1: Train, Test, and Test Predictions
plt.figure(figsize=(12, 8))
plt.plot(train_data, label='Train', color='black')
plt.plot(test_data, label='Test', color='green')
plt.plot(test_predictions_add, label='Prediction', color='red')
plt.title('Holt-Winters Additive Forecast - Train vs. Test Predictions')
plt.legend(loc='best')
plt.grid('True')
plt.show()

# Fit the final model on the entire dataset (additive trend & seasonality)
final_model = ExponentialSmoothing(monthly_data, trend='add', seasonal='add', seasonal_periods=12).fit()

# Forecast next 12 months
forecast = final_model.forecast(steps=12)

# Plot Historical Data with 12-Month Forecast
plt.figure(figsize=(12, 8))
monthly_data.plot(label='Observed', legend=True)
forecast.plot(label='Forecast', legend=True)
plt.title('Holt-Winters Additive Forecast - Next 12 Months')
plt.xlabel('Date')
plt.ylabel('Ratings Count')
plt.grid('True')
plt.show()

# Output final predictions
print("Final Predictions for the next 12 months:")
print(final_prediction)


```

### OUTPUT:

## TEST_PREDICTION:

![image](https://github.com/user-attachments/assets/e93c0707-6e55-4a46-9156-76b09a801077)


## FINAL_PREDICTION:

![image](https://github.com/user-attachments/assets/811f0c4e-eb67-48f0-bef7-b654670d2491)


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
