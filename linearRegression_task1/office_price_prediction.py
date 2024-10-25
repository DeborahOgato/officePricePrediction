import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


file_path = r'E:\AI assignment\linearRegression_task1\Nairobi Office Price Ex (1).csv'
data = pd.read_csv(file_path)


x = data['SIZE'].values
y = data['PRICE'].values

# Define the mean squared error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient Descent function
def gradient_descent(x, y, m, c, learning_rate, epochs):
    n = len(y)
    for epoch in range(epochs):
        # Compute predictions
        y_pred = m * x + c
        
        # Compute gradients
        dm = (-2 / n) * np.sum(x * (y - y_pred))
        dc = (-2 / n) * np.sum(y - y_pred)
        
        # Update weights
        m -= learning_rate * dm
        c -= learning_rate * dc
        
        # Print error every 100 epochs
        error = mean_squared_error(y, y_pred)
        print(f"Epoch {epoch+1}: Mean Squared Error = {error:.4f}")
    
    return m, c


# Set initial values and train the model
np.random.seed(0)
m, c = np.random.rand(2)  # Random initial values for m and c
learning_rate = 0.0001  # Learning rate
epochs = 10

# Train the model
m_trained, c_trained = gradient_descent(x, y, m, c, learning_rate, epochs)

# Predict the price for a 100 sq. ft office
size_100_price = m_trained * 100 + c_trained
print(f"The price for 100 sq. ft is: {size_100_price:.2f}")

# Plot the data points and the line of best fit
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, m_trained * x + c_trained, color='red', label='Line of Best Fit')
plt.xlabel('Office Size (sq. ft)')
plt.ylabel('Office Price')
plt.title('Office Price vs Size with Linear Regression Fit')
plt.legend()
plt.show()
