# Databricks notebook source
# MAGIC %md <img src="https://www.mlflow.org/docs/latest/_static/MLflow-logo-final-black.png" width="200">
# MAGIC #### MLflow Lab Part 1: Tracking Module
# MAGIC          
# MAGIC The goal of this example is to show how easy and straightforward it is to implement MLflow to a standard ML model.
# MAGIC 
# MAGIC **Simple steps to convert a ML model to MLflow:**
# MAGIC 1. Import the required MLflow packages
# MAGIC 2. Use MLflow Automatic Logging to log every metric and parameter of the model.
# MAGIC 3. Iniciate the training model with `mlflow.start_run` to start logging.

# COMMAND ----------

# MAGIC %md #### Classification Problem using CNNs with Tensorflow
# MAGIC 
# MAGIC This is a very simple classification problem where we will be using the Fashion MNIST dataset from Keras.
# MAGIC 
# MAGIC <img src="https://timesofdatascience.com/wp-content/uploads/2019/02/fashion-846x515.jpg"
# MAGIC          alt="Fashion MNIST dataset " width="400">
# MAGIC          
# MAGIC **The Fasion MNIST dataset includes:**
# MAGIC 
# MAGIC * 60,000 training examples
# MAGIC * 10,000 testing examples
# MAGIC * 10 classes 
# MAGIC * 28x28 grayscale/single channel images
# MAGIC 
# MAGIC **The ten fashin labels include:**
# MAGIC 
# MAGIC 1. T-shirt/top
# MAGIC 2. Trouser/pants
# MAGIC 3. Pullover shirt
# MAGIC 4. Dress
# MAGIC 5. Coat
# MAGIC 6. Sandal
# MAGIC 7. Shirt
# MAGIC 8. Sneaker
# MAGIC 9. Bag
# MAGIC 10. Ankle boot

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D
from tensorflow.keras.models import Model

import mlflow
from mlflow import pyfunc
import mlflow.tensorflow

# COMMAND ----------

print(tf.__version__)

# COMMAND ----------

# MAGIC %md ### Load the Dataset

# COMMAND ----------

# Load in the data
fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Increase one dimension so it can be used by the 2D convolutional keras layer
x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)
print("x_train.shape:", x_train.shape)


# COMMAND ----------

# MAGIC %md ### Run the Model

# COMMAND ----------

def run_model(params):
  
  # number of classes
  K = len(set(y_train))
  print("number of classes:", K)
  # Build the model using the functional API
  i = Input(shape=x_train[0].shape)
  x = Conv2D(32, params['convSize'], strides=2, activation='relu')(i)
  x = Conv2D(64, params['convSize'], strides=2, activation='relu')(x)
  x = Conv2D(128, params['convSize'], strides=2, activation='relu')(x)
  x = Flatten()(x)
  x = Dropout(0.2)(x)
  x = Dense(512, activation='relu')(x)
  x = Dropout(0.2)(x)
  x = Dense(K, activation='softmax')(x)

  model = Model(i, x)

  # Compile and fit
  # Note: make sure you are using the GPU for this!
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=params['epochs'])


# COMMAND ----------

for epochs, convSize in [[3,2], [15,3]]:
  params = {'epochs': epochs,
            'convSize': convSize}
  run_model(params)
