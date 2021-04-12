#!/usr/bin/env python
# coding: utf-8

# In[25]:


import matplotlib.pyplot as plt
import numpy as np


# In[4]:


train = np.loadtxt(r'C:\Users\Gaodongyu\Documents\gaodongyu\10 math of ML\sourcecode-cn\sourcecode-cn\click.csv', delimiter=',', skiprows=1)
train_x = train[:, 0]
train_y = train[:, 1]


# In[6]:


plt.plot(train_x, train_y, 'o')
plt.show()


# 线性回归

# In[8]:


theta0 = np.random.rand()
theta1 = np.random.rand()


# In[10]:


def f(x):
    return theta0 + theta1 * x


# In[11]:


def E(x,y):
    return 0.5 * (np.sum(y - f(x)) ** 2)


# In[13]:


mu = train_x.mean()
sigma = train_x.std()
def standardize(x):
    return (x - mu)/sigma
train_z = standardize(train_x)


# In[16]:


plt.plot(train_z, train_y, 'o')
plt.show()


# In[18]:


ETA = 1e-3
diff = 1
count = 0

error = E(train_z, train_y)
while diff > 1e-2:
    tmp0 = theta0 - ETA * np.sum(f(train_z) - train_y)
    tmp1 = theta1 - ETA * np.sum((f(train_z) - train_y) * train_z)
    
    theta0 = tmp0
    theta1 = tmp1
    
    current_error = E(train_z, train_y)
    diff = error - current_error
    error = current_error
    
    count += 1
    log = '第{}次: theta0 = {:.3f}, theta1 = {:.3f}, 差值 = {:.4f}'
    print(log.format(count, theta0, theta1, diff))


# In[21]:


x = np.linspace(-3, 3, 100)
plt.plot(train_z, train_y, 'o')
plt.plot(x, f(x))
plt.show


# In[26]:


print(f(standardize(100)))
print(f(standardize(200)))
print(f(standardize(300)))


# 多项式回归

# In[28]:


theta = np.random.rand(3)


# In[29]:


def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x**2]).T


# In[31]:


X = to_matrix(train_z)


# In[32]:


def f(x):
    return np.dot(x, theta)


# In[34]:


diff = 1

error = E(X, train_y)
while diff > 1e-2:
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error


# In[36]:


x = np.linspace(-3, 3, 100)

plt.plot(train_z, train_y, 'o')
plt.plot(x, f(to_matrix(x)))
plt.show()


# In[38]:


def MSE(x, y):
    return (1 / x.shape[0]) * np.sum((y - f(x)) ** 2)

theta = np.random.rand(3)

errors = []
diff = 1
errors.append(MSE(X, train_y))
while  diff > 1e-2:
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    errors.append(MSE(X, train_y))
    diff = errors[-2] - errors[-1]

x = np.arange(len(errors))
plt.plot(x, errors)
plt.show()


# 随机梯度下降法

# In[41]:


theta = np.random.rand(3)

errors = []

diff = 1

#重复学习
errors.append(MSE(X, train_y))
while diff > 1e-2:
    p = np.random.permutation(X.shape[0])
    for x,y in zip(X[p,:], train_y[p]):
        theta = theta - ETA * (f(x) - y) * x
    
    errors.append(MSE(X, train_y))
    diff = errors[-2] - errors[-1]


# In[42]:


x = np.linspace(-3, 3, 100)

plt.plot(train_z, train_y, 'o')
plt.plot(x, f(to_matrix(x)))
plt.show()

