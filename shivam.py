import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st
import random

# Set up the data
n_samples = 250
X = np.sort(random.sample(range(1, 50), 25))
y = X**2+7
 
 
test = np.sort(random.sample(range(1, 50), 20))
y_test = test**2+7
# test = test[:, np.newaxis]
n_neighbors = range(1, 25)
# bias = np.zeros(len(n_neighbors))
# variance = np.zeros(len(n_neighbors))
mse = np.zeros(len(n_neighbors))

k= st.sidebar.slider("K Value", min_value=1,max_value=14,step=1)
# Create a KNN regressor
knn = KNeighborsRegressor(n_neighbors=k)

# Train the regressor
knn.fit(X[:, np.newaxis], y)

# Predict on the training data
y_pred = knn.predict(test[:, np.newaxis])

# Calculate the bias, variance, and MSE
bias = np.mean((y_test - np.mean(y_pred)) ** 2)
variance = np.mean((y_pred - np.mean(y_pred)) ** 2)
# mse= mean_squared_error(y, y_pred)

# Define the data to plot
 
a = ['bais', 'variance']
b = [bias, variance]

# # # Create a bar chart
# plt.bar(x, y)

# # # Set the title and axis labels
# plt.title('Bar Chart Example')
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')





    
    
    
plt.plot(test,y_pred)    
    
plt.title("Line graph")    
plt.ylabel('Y axis')    
plt.xlabel('X axis')    
#plt.show() 
# # Show the plot
plt.show()
fig = plt.gcf()
st.pyplot(fig)
#st.pyplot()

a = ['bais', 'variance']
b = [bias, variance]

# # # Create a bar chart
plt.bar(a, b)

# # # Set the title and axis labels
plt.title('Bar Chart Example')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()
fig = plt.gcf()
st.pyplot(fig)