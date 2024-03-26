#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import keras
from keras.layers import Dense, Dropout
from tensorflow.keras import Sequential
image_data = tf.keras.datasets.fashion_mnist 


# In[2]:


(train_images, train_labels), (test_images, test_labels) = image_data.load_data()


# In[3]:


print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)


# In[4]:


train_images=train_images/255.0
test_images = test_images/255.0


# In[5]:


n_samples = len(train_images)
n_samples


# In[6]:


n_samples = len(test_images)
n_samples


# In[7]:


import matplotlib.pyplot as plt
plt.gray() 
plt.matshow(train_images[0])
plt.matshow(train_images[10])
plt.show()


# In[8]:


for i in range(0,10):
    plt.matshow(train_images[i]) 
    
plt.show()


# In[9]:


test_labels[16]


# In[10]:


test_labels


# In[11]:


train_images[0]


# In[12]:


import matplotlib.pyplot as plt


# In[13]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(50, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])
model.summary()


# In[14]:


history = model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))
model.evaluate(test_images, test_labels)


# In[15]:


test_loss, test_acc=model.evaluate(test_images, test_labels, verbose=0)
print('/nTest accuracy:', test_acc)


# In[16]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')  # title
plt.ylabel('Accuracy')  #  ylabel
plt.xlabel('Epoch')  #  xlabel
plt.legend(['Train', 'Test'], loc='upper left')  #  'Test'
plt.show() 


# In[17]:


probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])


# In[18]:


predictions = probability_model.predict(test_images)


# In[19]:


predictions[0]


# In[20]:


np.argmax(predictions[0])


# In[21]:


test_labels[0]

