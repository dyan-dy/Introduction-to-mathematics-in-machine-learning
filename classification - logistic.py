#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np

train = np.loadtxt(r'C:\Users\Gaodongyu\Documents\gaodongyu\10 math of ML\sourcecode-cn\sourcecode-cn\images2.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:, 2]

plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
plt.plot(train_x[train_y == 0, 0], train_x[train_y == 0, 1], 'x')
plt.axis('scaled')
plt.show()


# In[15]:


theta = np.random.rand(3)

mu = train_x.mean(axis = 0)
sigma = train_x.std(axis = 0)

def standardize(x):
    return (x - mu)/sigma

train_z = standardize(train_x)
# print(train_z)


# In[20]:


def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    return np.hstack([x0, x])

X = to_matrix(train_z)


# In[23]:


plt.plot(train_z[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_x[train_y == 0, 1], 'x')
# plt.axis('scaled')
plt.show()


# In[24]:


def f(x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))


# In[27]:


def classify(x):
    return (f(x) >= 0.5).astype(np.int)


# In[29]:


ETA = 1e-3
epoch = 5000
count = 0
for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    count += 1
    print('第{}次: w = {}'.format(count, theta))


# In[31]:


x0 = np.linspace(-2, 2, 100)
plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.plot(x0, -(theta[0] + theta[1] * x0) / theta[2], linestyle='dashed')
plt.show()

