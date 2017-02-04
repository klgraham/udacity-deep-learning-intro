import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

data = pd.read_fwf('brain_body.txt')
x = data[['Brain']]
y = data[['Body']]

lr = linear_model.LinearRegression()
lr.fit(x, y)

plt.scatter(x, y)
plt.plot(x, lr.predict(x))
plt.show()

challenge_data = pd.read_csv('challenge_dataset.txt', names=['Brain', 'Body'])
x1 = challenge_data[['Brain']]
y1 = challenge_data[['Body']]

plt.scatter(x1, y1)
plt.plot(x1, lr.predict(x1))
plt.show()