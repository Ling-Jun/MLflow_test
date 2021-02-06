# Databricks notebook source
# MAGIC %md ### Classes and Utility functions

# COMMAND ----------

import pandas as pd
import tensorflow as tf
import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

class Utils:
  @staticmethod
  def load_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Increase one dimension so it can be used by the 2D convolutional keras layer
    x_train = np.expand_dims(x_train,-1)
    x_test = np.expand_dims(x_test,-1)
    print("x_train.shape:", x_train.shape)
    return (x_train, y_train), (x_test, y_test)

# COMMAND ----------

# displayHTML("""
# <div> Declared Utils class with utility methods:</div> 
#   <li> Declared <b style="color:green">load_data(path, index_col=0)</b> returns numpy arrays of training and test data</li><br/>
# """)

# COMMAND ----------

import pandas as pd
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

# COMMAND ----------

class PlotUtils:
    @staticmethod
    def confusionMatrix(y_test, preditcion, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
      """
      This function prints and plots the confusion matrix.
      Normalization can be applied by setting `normalize=True`.
      """
      cm = confusion_matrix(y_test, preditcion)
      if normalize:
          cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
          print("Normalized confusion matrix")
      else:
          print('Confusion matrix, without normalization')

      print(cm)

      plt.imshow(cm, interpolation='nearest', cmap=cmap)
      plt.title(title)
      plt.colorbar()
      tick_marks = np.arange(len(classes))
      plt.xticks(tick_marks, classes, rotation=45)
      plt.yticks(tick_marks, classes)

      fmt = '.2f' if normalize else 'd'
      thresh = cm.max() / 2.
      for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
          plt.text(j, i, format(cm[i, j], fmt),
                   horizontalalignment="center",
                   color="white" if cm[i, j] > thresh else "black")

      plt.tight_layout()
      plt.ylabel('True label')
      plt.xlabel('Predicted label')
      plt.show()
      
    @staticmethod
    def predticClassification (model_uri, x_test):
      '''
      Function that loads a pyfunc flavor of the model and predicts with unseen data
      '''
      print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_uri))
      model = mlflow.pyfunc.load_model(model_uri)
      predictions = pd.DataFrame(model.predict(x_test))
      return predictions

# COMMAND ----------

# displayHTML("""
# <div> Declared PlotUtils class with utility methods:</div> 
#   <li> Declared <b style="color:green">confusionMatrix(model_uri, power_predictions, past_power_output)</b> Plots a confusion matrix </li>
#   <li> Declared <b style="color:green">predticClassification(model_uri, x_test)</b> Returns the result of the model's prediction</b></li>
#    <br/>
# """)

# COMMAND ----------

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D
from tensorflow.keras.models import Model
import mlflow
import mlflow.tensorflow

print("Using mlflow version {}".format(mlflow.__version__))

# COMMAND ----------

class TensorFlowModel:
  def __init__(self, x_train, y_train, x_test, y_test, params, activation="softmax"):
    self.params= params
    self.x_train = np.expand_dims(x_train,-1)
    self.x_test = np.expand_dims(x_test,-1)
    self.y_train= y_train
    self.y_test= y_test
    self.K = len(set(y_train))
    self.i = Input(shape=self.x_train[0].shape)
    self.x = Conv2D(32, params['convSize'], strides=2, activation='relu')(self.i)
    self.x = Conv2D(64, params['convSize'], strides=2, activation='relu')(self.x)
    self.x = Conv2D(128, params['convSize'], strides=2, activation='relu')(self.x)
    self.x = Flatten()(self.x)
    self.x = Dropout(0.2)(self.x)
    self.x = Dense(512, activation='relu')(self.x)
    self.x = Dropout(0.2)(self.x)
    self.x = Dense(self.K, activation=activation)(self.x)
    self._model= Model(self.i,self.x)
    self._model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
  
  def mlflow_run(self, run_name="TensorFlow: MNIST Model"):
    with mlflow.start_run(run_name=run_name) as run:
      # Automatically capture the model's parameters, metrics, artifacts,
      # and source code with the autolog() function
      mlflow.tensorflow.autolog()
      self._model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test),epochs=self.params['epochs'])
    return run.info.run_id
    

# COMMAND ----------

# displayHTML("""
# <div> Declared TensorFlowModel class with public methods:</div> 
#   <li> Declared <b style="color:green"> mlflow_run(model, X_train, y_train, **kwargs)</b> returns MLflow run_id </li>
#   <br/>
# """)
