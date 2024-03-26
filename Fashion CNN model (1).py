#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
np.random.seed(0)
import tensorflow as tf


# ## Importing libraries

# In[2]:


from tensorflow.keras import datasets, layers, models


# In[3]:


import matplotlib.pyplot as plt


# ## Load and preprocess data

# In[4]:


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[5]:


type(train_images)


# In[6]:


type(train_labels)


# In[7]:


print(len(train_images))


# In[8]:


print(len(train_images))


# In[9]:


print(train_images[0].shape)


# In[10]:


train_images[0]


# In[11]:


plt.imshow(train_images[0], cmap='gray')
plt.imshow


# In[12]:


print(train_labels[0])


# In[13]:


#Scale these values to a range of 0 to 1 by divide the values by 255. 

train_images = train_images / 255.0

test_images = test_images / 255.0


# In[14]:


train_images=train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images=test_images.reshape((test_images.shape[0], 28, 28, 1))


# In[15]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))


# In[16]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[17]:


history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))  # Corrected parentheses and added a comma after epochs=5


# In[18]:


test_loss, test_acc=model.evaluate(test_images, test_labels, verbose=0)
print('/nTest accuracy:', test_acc)


# In[19]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')  # C title
plt.ylabel('Accuracy')  #  ylabel
plt.xlabel('Epoch')  # xlabel
plt.legend(['Train', 'Test'], loc='upper left')  #  'Test'
plt.show()  


# In[ ]:




