#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go


# In[2]:


from plotly.figure_factory import create_table
import plotly.express as px
gapminder = pd.read_excel("fmnist.xlsx")


# In[3]:


table = create_table(gapminder.head(5))


# In[4]:


py.iplot(table)


# In[5]:


type(gapminder)


# In[6]:


fig=px.bar(gapminder, x='Models', y='Accuracy', height=400)


# In[7]:


fig.show()


# In[8]:


fig=px.bar(gapminder, x='Time taken', y='Accuracy',
           hover_data= ['Models'], color='Models', height=400)


# In[9]:


fig.show()


# In[ ]:




