Introduction
============

MLToolKit supports all stages of the machine learning application development process. 

.. image:: https://raw.githubusercontent.com/mltoolkit/MLToolkit/master/MLToolkit/image/MLTKProcess.png
   :height: 200px

Functions
---------
  * Data Extraction (SQL, Flatfiles, Binary Files, Images, etc.)
  * Exploratory Data Analysis (statistical summary, univariate analysis, visulize distributions, etc.)
  * Feature Engineering (Supports numeric, text, date/time. Image data support will integrate in later releases of v0.1)
  * Model Building (Currently supported for binary classification and regression only)
  * Hyper Parameter Tuning [in development for v0.2]
  * Cross Validation (will integrate in later releases of v0.1)
  * Model Performance Analysis, Explain Predictions (LIME and SHAP) and Performance Comparison Between Models.
  * JSON input script for executing model building and scoring tasks.
  * Model Building UI [in development for v0.2]
  * ML Model Building Project [in development for v0.2]
  * Auto ML (automated machine learning) [in development for v0.2]
  * Model Deploymet and Serving [included, will be imporved for v0.2]

Supported Machine Learning Algorithms/Packages
----------------------------------------------
  * RandomForestClassifier: scikit-learn
  * LogisticRegression: statsmodels
  * Deep Feed Forward Neural Network (DFF): tensorflow
  * Convlutional Neural Network (CNN): tensorflow
  * Gradient Boost : catboost, xgboost, lightgbm
  * Linear Regression: statsmodels
  * RandomForestRegressor: scikit-learn
  ... More models will be added in the future releases ...
