#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


# In[2]:


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[3]:


train_images = train_images / 255.0

test_images = test_images / 255.0


# In[4]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[5]:


plt.figure(figsize=(10,10))
for i in range(5):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[6]:


train_images.shape, test_images.shape


# In[7]:


train_images=train_images.reshape((train_images.shape[0], 28, 28))
test_images=test_images.reshape((test_images.shape[0], 28, 28))


# In[ ]:


train_images = train_images.astype('float32')
test_images = test_images.astype('float32')


# In[8]:


# convert data to 3 channels
train_images = np.stack((train_images,)*3, axis=-1)
test_images = np.stack((test_images,)*3, axis=-1)


# In[9]:


from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[10]:


train_images.shape, test_images.shape


# In[11]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[12]:


# data augmentation with generator
train_generator = ImageDataGenerator(
    rescale = 1./255,  # normalization of images
    rotation_range = 40, # augmention of images to avoid overfitting
    shear_range = 0.2,
    zoom_range = 0.2,
    fill_mode = 'nearest'
)

val_generator = ImageDataGenerator(rescale = 1./255)

train_iterator = train_generator.flow(train_images, train_labels, batch_size=512, shuffle=True)

val_iterator = val_generator.flow(test_images, test_labels, batch_size=512, shuffle=False)


# In[13]:


from tensorflow.keras.applications.resnet50 import ResNet50


# In[14]:


model = Sequential()
# add the pretrained model
model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
# add fully connected layer with output
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

# set resnet layers not trainable
model.layers[0].trainable=False
model.summary()


# ## Training the model

# In[15]:


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[16]:


history = model.fit(train_iterator, epochs=5, validation_data=val_iterator)


# In[17]:


test_loss, test_acc = model.evaluate(train_iterator,, verbose=0)

print('\nTest accuracy:', test_acc)


# In[18]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')  # title
plt.ylabel('Accuracy')  #  ylabel
plt.xlabel('Epoch')  # xlabel
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()  


# In[19]:


probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])


# In[20]:


predictions = probability_model.predict(test_images)


# In[21]:


predictions[0]


# In[22]:


np.argmax(predictions[0])


# In[23]:


test_labels[0]


# In[ ]:




