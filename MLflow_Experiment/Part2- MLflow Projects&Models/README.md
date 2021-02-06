<!-- <img src="https://www.mlflow.org/docs/latest/_static/MLflow-logo-final-black.png" width="200"> -->

#### MLflow Lab Part 2: Projects and Models
         
This second lab is aimed to explain how to containerize/package models and projects using MLflow. 
All of that in the most simple and right to the point way!

&nbsp;

**MLflow Projects**

An MLflow Project is a format for packaging data science code in a reusable and reproducible way, based primarily on conventions. In addition, the Projects component includes an API and command-line
tools for running projects, making it possible to chain together projects into workflows.
  
###### What's does a MLflow Project Look Like?

[MLflow Project Example](./MLflow_project)


- the __conda.yaml__ file in __MLflow_project__ folder needs to include all the package name & versions
 under pip:, otherwise runing __run_project.py__ from root will not work. 
- The root treats the MLflow_project folder as an independent environment/package. So even if the conda env
 in my local computer's terminal has matplotlib, as long as matplotlib isn't in conda.yaml, 
 __python run_project.py__ won't work!!!
- __MLProject__ is another config file that needs to include the right params and commands.
 The command is the same command as when we run the script __Train_TensorFlow.py__ directly.

[run_project.py file](./run_project.py)

- run_id automatically logs all aspects of the ML model

- we can intentionally log artifacts using __mlflow.get_artifact_uri()__ , but log the artifact_URI doesn't mean we actually log the artifact in the artifact folder, we need to add what we want to log into the artifact, otherwise it's just an empty folder. 
That's why in the run_id folder, there are files in the run_id folder

&nbsp;

**MLflow Models**

An MLflow Model is a standard format for packaging machine learning models that can be used in a variety of downstream tools—for example, real-time serving through a REST API or batch 
inference on Apache Spark. The format defines a convention that lets you save a model in different “flavors” that can be understood by different downstream tools.

###### What's does a MLflow Model Look Like?

<img src="https://raw.githubusercontent.com/Isaac4real/MLflow_Experiment/master/Part2-%20MLflow%20Projects%26Models/Images/ModelFolderStructure.png" height="200">

&nbsp;

&nbsp;
## Content
1. Introduction MLflow Projects module: API (command line and programmatically)
2. Configure Databricks CLI to be able to run MLproject from GitHub
3. Load and run the project form GitHub using `MLflow Fluent API` and `MLflow Project API`
4. Explore MLflow UI: check for conda.yaml, MLproject file and metrics
5. Replicate everything in your machine (locally)
