## Reg no: 212222240028
## Developed By: DODDA JAYASRI
## Date: 

# Ex.No: 6               HOLT WINTERS METHOD

### AIM:
To create and implement Holt Winter's Method Model using python for goodreadsbooks dataset.

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
# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from math import sqrt

# 2. Load dataset
df = pd.read_csv('Salesforcehistory.csv')

# 3. Convert 'publication_date' to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Handle invalid dates
df = df.dropna(subset=['Date'])  # Remove rows with missing dates
df.set_index('Date', inplace=True)

# 4. Resample the 'ratings_count' data to monthly frequency (sum)
monthly_data = df['Open'].resample('M').sum()

# 5. Plot the resampled data
plt.figure(figsize=(10, 6))
plt.plot(monthly_data, label='Monthly Ratings Count')
plt.title('Monthly Resampled Ratings Count')
plt.xlabel('Date')
plt.ylabel('Ratings Count')
plt.legend()
plt.grid(True)
plt.show()

# 5. Split the data into training and testing sets
train_size = int(len(monthly_data) * 0.8)
train, test = monthly_data[:train_size], monthly_data[train_size:]

# 6. Fit the Holt-Winters model
model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
model_fit = model.fit()

# 7. Make predictions for the test set
test_predictions = model_fit.forecast(len(test))

# 8. Calculate RMSE for test predictions
rmse = sqrt(mean_squared_error(test, test_predictions))
print(f'Root Mean Squared Error for Test Predictions: {rmse}')

# 9. Final prediction for the next 12 months
final_prediction = model_fit.forecast(12)


# 10. Create two subplots for graphical representation
fig, axs = plt.subplots(2, 1, figsize=(12, 12))

# Plot 1: Training and Test Data with Test Predictions
axs[0].plot(train, label='Training Data', color='blue', linewidth=2)
axs[0].plot(test, label='Test Data', color='orange', linewidth=2)
axs[0].plot(test_predictions, label='Test Predictions', color='green', linestyle='--', marker='o')
axs[0].set_title('Holt-Winters Forecasting: Training and Test Data', fontsize=16)
axs[0].set_xlabel('Date', fontsize=14)
axs[0].set_ylabel('Ratings Count', fontsize=14)
axs[0].legend()
axs[0].grid()

# Plot 2: Final Predictions for the next 12 months
axs[1].plot(monthly_data, label='Historical Data', color='blue', linewidth=2)
axs[1].plot(final_prediction.index, final_prediction, label='Final Predictions', color='red', linestyle='--', marker='x')
axs[1].set_title('Holt-Winters Forecasting: Final Predictions for Next 12 Months', fontsize=16)
axs[1].set_xlabel('Date', fontsize=14)
axs[1].set_ylabel('Ratings Count', fontsize=14)
axs[1].legend()
axs[1].grid()

# Adjust layout for better fit
plt.tight_layout()
plt.show()


```

### OUTPUT:

![image](https://github.com/user-attachments/assets/838464fe-d905-4f2e-a233-2b410da30faa)



## TEST_PREDICTION:

![image](https://github.com/user-attachments/assets/c2526ac6-67ef-4893-9526-d0110a7459b3)


## FINAL_PREDICTION:

![image](https://github.com/user-attachments/assets/cbac2a5e-3f51-4b13-858e-f919aba9c276)




### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
