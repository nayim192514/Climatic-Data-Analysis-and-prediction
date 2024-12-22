import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"E:\Machine learning\Ml class_cse\Climate data_jashore_2019-2021.csv"
climate_data = pd.read_csv(file_path)

# Selecting relevant columns and ensuring no missing values
data = climate_data[['Temp_MAX', 'Temp_MIN', 'RH']].dropna()

# Splitting the data into features and target
X = data[['Temp_MIN', 'RH']]
y = data['Temp_MAX']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the Linear Regression model
linear_model = LinearRegression()

# Fitting the model
linear_model.fit(X_train, y_train)

# Predicting on the test set
y_pred = linear_model.predict(X_test)

# Calculating performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Printing the metrics
print(f"Linear Regression MSE: {mse:.4f}")
print(f"Linear Regression R2: {r2:.4f}")

# Visualizing the results: Predicted vs. Actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='Green', alpha=0.7, edgecolor='k', label='Predicted vs. Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Fit')
plt.title('Linear Regression: Actual vs. Predicted Temp_MAX')
plt.xlabel('Actual Temp_MAX')
plt.ylabel('Predicted Temp_MAX')
plt.legend()
plt.grid(True)
plt.show()
