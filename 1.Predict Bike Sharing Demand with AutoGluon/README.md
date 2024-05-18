
# Predict Bike Sharing Demand with AutoGluon

Building ML models for the Bike Sharing Demand competition in Kaggle.using AutoGluon library ,Tabular Prediction and using AWS Sagemaker Studio,The target value is the "count"

Bike sharing systems are a means of renting bicycles where the process of obtaining membership, rental, and bike return is automated throughout a city. Using these systems, people are able rent a bike from a one location and return it to a different place on an as-needed basis,In this project.



## AutoGluon:

AutoGluon is a new open source AutoML library that automates deep learning (DL) and machine learning (ML) for real world applications involving image, text and tabular datasets. Whether you are new to ML or an experienced practitioner, AutoGluon simplify workflow.AutoGluon can develop DL models using just a few lines of Python code.

- With AutoGluon, there is no need to specify validation data. AutoGluon will optimally allocate a validation set using the training data provided.
- AutoGluon works with tabular data using TabularPrediction
- AutoGluon processes the data and trains an ensemble of ML models called a “predictor” which is able to predict the target variable .
- It uses the other columns as predictive features
- we don’t need to do any data processing, feature engineering, or even declare the type of prediction problem
- AutoGluon automatically prepares the data and infers whether our problem is regression or classification (including whether it is binary or multiclass)

##  Model one 
Pure AutoGluon’s Tabular Prediction
- Loading the Dataset using kaggle cli
- Create the train dataset in pandas by reading the csv
- drop casual and registered columns as they are not in the test dataset.
- Train a model using AutoGluon’s Tabular Prediction
- Use the root_mean_squared_error as the metric to use for evaluation.
- Set a time limit of 10 minutes(600 seconds),how long fit() should run for
- Use the preset best_quality to focus on creating the best model.
- Output summary of information about models produced during fit() 
- WeightedEnsemble_L3: is the best model

## Model two 
Using Exploratory Data Analysis (EDA) and Creating an additional feature
- Create a histogram of all features
- Create a new features,by separate the datetime into year,month,day,hours
- Make category types for these so models know they are not just numbers
- Rerun the model with the same settings as before, but with new features
- WeightedEnsemble_L3: is the best model

##  Model three
Hyper parameter optimization
- num_bag_folds
  - Number of folds used for bagging of models. When num_bag_folds = k, training time is roughly increased by a factor of k
  - Increasing num_bag_folds will result in models with lower bias but that are more prone to overfitting
  - To improve predictions, avoid increasing num_bag_folds much beyond 10
- num_bag_sets
   - Number of repeats of kfold bagging to perform
   - Values greater than 1 will result in superior predictive performance
- num_stack_levels
   - Number of stacking levels to use in stack ensemble
   - Recommend values between 1-3 to maximize predictive performance.
- WeightedEnsemble_L2: is the best model
## Reference

[AutoGluon TabularPrediction](https://towardsdatascience.com/autogluon-deep-learning-automl-5cdb4e2388ec#:~:text=AutoGluon%20enables%20you%20to%20automatically,supervised%20learning%20with%20tabular%20datasets.)

[AutoGluon](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html.)
