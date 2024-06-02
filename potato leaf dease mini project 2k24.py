#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk(r'C:\Users\hp\Desktop\Potato'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

import os
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

# Your code for data loading and preprocessing


# In[5]:


SIZE= 256
SEED_TRAINING=121
SEED_TESTING=197
SEED_VALIDATION=164
CHANNELS=3
n_classes=3
EPOCHS=20
BATCH_SIZE=16
input_shape=(SIZE, SIZE, CHANNELS)


# In[6]:


train_datagen= ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest')


# In[7]:


validation_datagen= ImageDataGenerator(rescale=1./255)
test_datagen= ImageDataGenerator(rescale=1./255)


# In[9]:


train_generator = train_datagen.flow_from_directory(
        directory= r'C:\Users\hp\Desktop\Potato\Train',
        target_size= (255,256), # all images will be resized to 64*64
        batch_size= BATCH_SIZE,
        class_mode= 'categorical',
        color_mode="rgb")


# In[10]:


validation_generator= validation_datagen.flow_from_directory(
            directory= r'C:\Users\hp\Desktop\Potato\Valid',
            target_size= (256,256),
            batch_size= BATCH_SIZE,
            class_mode= 'categorical',
            color_mode="rgb")


# In[13]:


test_generator = test_datagen.flow_from_directory(
        directory =r'C:\Users\hp\Desktop\Potato\Test',
        target_size = (256, 256),
        batch_size = BATCH_SIZE,
        class_mode = 'categorical',
        color_mode = "rgb")


# In[14]:


from tensorflow.keras import models, layers

model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])


# In[15]:


model.summary()


# In[16]:


model.compile(
    optimizer='adam',
    loss= tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)


# In[18]:


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // validation_generator.batch_size
)


# In[19]:


score= model.evaluate(test_generator)
print('Test loss:',score[0])
print('Test accuracy:',score[1])


# In[20]:


acc= history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss= history.history['val_loss']


# In[21]:


plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[22]:


train_generator.class_indices




class_names = list(train_generator.class_indices.keys())
class_names





from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image





model.save('my_model.keras')




def prediction(img):
    class_names = ['Early_blight', 'Healthy', 'Late_blight']

    # Load the image and convert it to RGB if it's grayscale
    my_image = image.load_img(img, target_size=(256, 256))
    my_image = image.img_to_array(my_image)
    if my_image.shape[-1] == 1:  # If the image has only one channel (grayscale)
        my_image = np.concatenate([my_image] * 3, axis=-1)  # Convert to RGB

    my_image = np.expand_dims(my_image, 0)

    out = np.round(model.predict(my_image)[0], 2)
    fig = plt.figure(figsize=(7, 4))
    plt.barh(class_names, out, color='lightgray', edgecolor='red', linewidth=1, height=0.5)

    for index, value in enumerate(out):
        plt.text(value/2 + 0.1, index, f"{100*value:.2f}%", fontweight='bold')
    plt.xticks([])
    plt.yticks([0, 1, 2], labels=class_names, fontweight='bold', fontsize=14)
    fig.savefig('pred_img.png', bbox_inches='tight')
    return plt.show()





img=r'C:\Users\hp\Desktop/Potato/Test/Potato___healthy/Potato_healthy-26-_0_4635.jpg'
prediction(img)





img=r'C:\Users\hp\Desktop/Potato/Test/Potato___Late_blight/00695906-210d-4a9d-822e-986a17384115___RS_LB 4026.JPG'
prediction(img)




img=r'C:\Users\hp\Desktop/Potato/Test/Potato___Early_blight/08194ca3-f0b2-4aaa-8df8-5ec5ddc6696a___RS_Early.B 8151.JPG'
prediction(img)













