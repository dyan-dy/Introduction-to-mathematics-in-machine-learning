#!/usr/bin/env python
# coding: utf-8

# In[22]:


import matplotlib.pyplot as plt
import numpy as np

train = np.loadtxt(r'C:\Users\Gaodongyu\Documents\gaodongyu\10 math of ML\sourcecode-cn\sourcecode-cn\data3.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:, 2]

plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
plt.plot(train_x[train_y == 0, 0], train_x[train_y == 0, 1], 'x')
# plt.axis('scaled')
plt.show()


# In[23]:


theta = np.random.rand(4)

accuracies = []

mu = train_x.mean(axis = 0)
sigma = train_x.std(axis = 0)

def standardize(x):
    return (x - mu)/sigma

train_z = standardize(train_x)


# In[24]:


def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    x3 = x[:, 0, np.newaxis] ** 2
    return np.hstack([x0, x, x3])

X = to_matrix(train_z)


# In[25]:


def f(x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))

def classify(x):
    return (f(x) >= 0.5).astype(np.int)


# In[26]:


ETA = 1e-3
epoch = 5000
count = 0
for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    result = classify(X) == train_y
    accuracy = len(result[result == True]) / len(result)
    accuracies.append(accuracy)
    
    count += 1
    print('第{}次: w = {}'.format(count, theta))


# In[27]:


x1 = np.linspace(-2, 2, 100)
x2 = -(theta[0] + theta[1] * x1 + theta[3] * x1 ** 2) / theta[2]


# In[28]:


plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.plot(x1, x2, linestyle='dashed')
plt.show()


# In[29]:


x = np.arange(len(accuracies))

plt.plot(x, accuracies)
plt.show()


# 随机梯度下降法

# In[33]:


theta = np.random.rand(4)

for _ in range(epoch):
    p = np.random.permutation(X.shape[0])
    for x,y in zip(X[p, :], train_y[p]):
        theta = theta - ETA * (f(x) - y) * x


# In[34]:


x1 = np.linspace(-2, 2, 100)
x2 = -(theta[0] + theta[1] * x1 + theta[3] * x1 ** 2) / theta[2]

plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.plot(x1, x2, linestyle='dashed')
plt.show()


# In[ ]:




