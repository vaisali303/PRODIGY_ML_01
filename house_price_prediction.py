# House Price Prediction using Linear Regression

# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# 2. Load the dataset
data = pd.read_csv("train.csv")


# 3. Select required columns
data = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]


# 4. Remove rows with missing values
data = data.dropna()


# 5. Separate features and target
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data['SalePrice']


# 6. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 7. Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# 8. Make predictions
y_pred = model.predict(X_test)


# 9. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)


# 10. Visualize Actual vs Predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
