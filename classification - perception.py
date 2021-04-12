#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np


# In[6]:


train = np.loadtxt(r'C:\Users\Gaodongyu\Documents\gaodongyu\10 math of ML\sourcecode-cn\sourcecode-cn\images1.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:, 2]


# In[10]:


m


# In[12]:


w = np.random.rand(2)


# In[13]:


def f(x):
    if np.dot(w,x) >= 0:
        return 1
    else:
        return -1


# In[19]:


epoch = 10

count = 0
for _ in range(epoch):
    for x,y in zip(train_x, train_y):
        if f(x) != y:
            w = w + y * x
            count += 1
            print('第{}次: w = {}'.format(count, w))


# In[21]:


x1 = np.arange(0, 500)

plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
plt.plot(train_x[train_y == -1, 0], train_x[train_y == -1, 1], 'x')
plt.plot(x1, -w[0]/w[1] * x1, linestyle = 'dashed')
plt.show()


# In[24]:


print(f([200, 100]))
print(f([100, 200]))


# In[ ]:




