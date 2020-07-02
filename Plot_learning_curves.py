#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_context('talk')


# In[4]:


get_ipython().system('ls runs/')


# In[7]:


df1 = pd.read_csv('runs/run-baseline_mnist_20200427-205525_test-tag-accuracy.csv')
df1.head()


# In[8]:


df2 = pd.read_csv('runs/run-contrast_loss_model_mnist_20200427-205809_test-tag-accuracy.csv')


# In[14]:


fig, ax = plt.subplots()
ax.plot(df1['Step'], df1['Value'], label='MLP baseline')
ax.plot(df2['Step'], df2['Value'], label='Contrastive')

ax.set(xlabel='Epoch', ylabel='Accuracy (Test set)', title='MNIST dataset');
ax.legend();
fig.savefig('figs/mnist_test_acc_curves.png')


# In[15]:


df1 = pd.read_csv('runs/run-baseline_fashion_mnist_20200427-210140_test-tag-accuracy.csv')
df2 = pd.read_csv('runs/run-contrast_loss_model_fashion_mnist_20200427-210420_test-tag-accuracy.csv')


# In[16]:


fig, ax = plt.subplots()
ax.plot(df1['Step'], df1['Value'], label='MLP baseline')
ax.plot(df2['Step'], df2['Value'], label='Contrastive')

ax.set(xlabel='Epoch', ylabel='Accuracy (Test set)', title='Fashion MNIST dataset');
ax.legend();
fig.savefig('figs/fashion_mnist_test_acc_curves.png')


# In[ ]:




