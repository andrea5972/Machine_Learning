
# coding: utf-8

# Example adapted from [this online post](https://nextjournal.com/gkoehler/digit-recognition-with-keras).

# In[1]:


from keras.datasets import mnist
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[2]:


num_training = X_train.shape[0]
num_test = X_test.shape[0]
width = X_train.shape[1]
height = X_train.shape[2]
num_pixels = width * height
X_flat_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_flat_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
X_flat_train /= 255
X_flat_test /= 255


# In[3]:


X_flat_train.shape


# In[4]:


X_flat_test.shape


# In[5]:


y_encoded_train = np_utils.to_categorical(y_train)
y_encoded_test = np_utils.to_categorical(y_test)


# building a linear stack of densely connected layers with the sequential model from keras

# ![](nn_example.png)

# In[6]:


from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation

model = Sequential()

model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))                            

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))


# In[7]:


# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


# In[8]:


model.fit(X_flat_train, y_encoded_train,
          batch_size=128, epochs=2,
          verbose=2,
          validation_data=(X_flat_test, y_encoded_test))


# Compute model accuracy on the 10,000 testing examples 

# In[9]:


loss_and_metrics = model.evaluate(X_flat_test, y_encoded_test, verbose=2)

print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])


# To save a trained model: we save its structure and its weights.

# In[10]:


open('mnist_simple_cnn_model_structure.json', 'w').write(model.to_json())
model.save_weights('mnist_simple_cnn_model_weights.h5')


# Load saved model

# In[11]:


#from keras.models import model_from_json
#model = model_from_json(open('mnist_simple_cnn_model_structure.json').read())
#model.load_weights('mnist_simple_cnn_model_weights.h5')

