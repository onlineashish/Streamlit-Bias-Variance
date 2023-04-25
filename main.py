import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import bias_variance_decomp
import matplotlib.pyplot as plt
import streamlit as st

np.random.seed(42)
#default dataset
x = [1,2,3,4,5,6,7,8,9,10]
y = [1,4,9,16,25,36,49,64,81,100]

option = st.selectbox('Select Dataset', ('Simple', 'Complex'))

# Generate toy data
if(option == 'Complex'):
    x = np.random.rand(100) * 20
    y = 50 * x + x**2 + np.random.randn(100) * 100

# x = np.arange(1,51)
# y = x**2+3
if(option == 'Simple'):
    x = [1,2,3,4,5,6,7,8,9,10]
    y = [1,4,9,16,25,36,49,64,81,100]

# Create DataFrame
df = pd.DataFrame({'x': x, 'y': y})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['x']], df['y'], test_size=0.2 ,random_state=42)

k= st.sidebar.slider("K Value", min_value=1,max_value=len(X_train)//2,step=1)

knn = KNeighborsRegressor(n_neighbors=k,weights='distance')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# Calculate the bias, variance, and MSE
error, bias, variance = bias_variance_decomp(knn, X_train.values, y_train.values, X_test.values, y_test.values, loss='mse', num_rounds=100, random_seed=42)


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')



st.write("Mean Squared Error: ", error)
st.write("Bias: ", bias)
st.write("Variance: ", variance)

# Plot the data
plt.scatter(X_train, y_train, label='Trained values')
plt.scatter(X_test, y_test, label='True values')
plt.scatter(X_test, y_pred, label='Predicted Values')


plt.xlabel('House Area')
plt.ylabel('House Price')
plt.title('KNN Regression')
plt.legend()
plt.show()
fig = plt.gcf()
st.pyplot(fig)



#streamlit plot for bias and variance on varying k
nam = ['bias', 'variance']
val = [bias, variance]
plt.bar(nam, val )

plt.title('Bias and Variance')
plt.xlabel('Bias and Variance')
plt.ylabel('Value')
addlabels(nam, val)

plt.show()
fig = plt.gcf()
st.pyplot(fig)

