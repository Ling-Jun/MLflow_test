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

# MAGIC %run ./classes_init

# COMMAND ----------

print(tf.__version__)

# COMMAND ----------

# MAGIC %md ### Load the Dataset

# COMMAND ----------

# Load in the data
(x_train, y_train), (val_x, val_y) = Utils.load_data()

# COMMAND ----------

# MAGIC %md ### Run the Model

# COMMAND ----------

params = [{'epochs': 1, 'convSize': 2},
         {'epochs': 10, 'convSize': 3}]
for params in params_list:
  TensorFlow_obj = TensorFlowModel(x_train, y_train, val_x, val_y, params, activation="softmax")
  print("Using paramerts={}".format(params))
  runID = TensorFlow_obj.mlflow_run()
  print("MLflow run_id={}".format(runID))

# COMMAND ----------

# MAGIC %md ### Let's now explore the MLflow  UI
# MAGIC 
# MAGIC * Add Notes & Tags
# MAGIC * Compare Runs pick two best runs
# MAGIC * Annotate with descriptions and tags
# MAGIC * Evaluate the best run
