# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The related data is located at this url "https://archive.ics.uci.edu/dataset/222/bank+marketing". The zip file is downloaded and extracted to be able to upload the file "bankmarketing_traing.csv" as a tabular dataset to the Azure ML Studio.
The classification goal is to predict if the client will subscribe a term deposit (variable y).

After running the experiments, the best performing model was a VotingEnsemble, achieving an accuracy of 0.90799. Below are the top results:
VotingEnsemble: 0.9079870314518954
StackEnsemble: 0.9067262660515146
LightGBM: 0.9059963517641572
XGBoostClassifier: 0.9054433738232561
XGBoostClassifier: 0.9040278165165234
XGBoostClassifier: 0.9032757832289446
XGBoostClassifier: 0.9032536612933524
XGBoostClassifier: 0.9032315256593612
XGBoostClassifier: 0.903120962947338
LightGBM: 0.9016832308683319
LogisticRegression: 0.90157267109168
LogisticRegression: 0.9013736020466034
XGBoostClassifier: 0.9009311809469832
LogisticRegression: 0.8994492813164428
LightGBM: 0.899427169165421
XGBoostClassifier: 0.8992944336380386
XGBoostClassifier: 0.8988078440084271
XGBoostClassifier: 0.8977682598041445
XGBoostClassifier: 0.8965738592320838
LightGBM: 0.8926367662323041
RandomForest: 0.8843644538840763
ExtremeRandomTrees: 0.8830152144397247
LightGBM: 0.8830152144397247
LightGBM: 0.8830152144397247
LightGBM: 0.8830152144397247
...
RandomForest: 0.8103117368096112
RandomForest: 0.7761609642482945
RandomForest: 0.7604344147769089
ExtremeRandomTrees: 0.6899873824048726

## Scikit-learn Pipeline
Before any training or compute configuration, the Azure environment was initialized by loading the workspace and creating an experiment using the Azure ML SDK.

Compute Cluster Setup: The pipeline starts by creating an Azure compute cluster to run experiments. The cluster uses Standard_D2_V2 VMs with a maximum of 4 nodes, providing sufficient resources for parallel training runs.

Data Preparation: Since the dataset could not be directly loaded using TabularDatasetFactory.from_delimited_files(url), it was uploaded to Azure ML Studio as a tabular dataset. The data was then cleaned by converting categorical variables to numerical features and mapping binary values. The dataset was split into training and test sets with a 2/3 to 1/3 ratio.

Model Training and Hyperparameter Tuning: A Logistic Regression model was trained with hyperparameters C (0.1 or 1) and max_iter (10, 20, 50). Hyperparameter optimization was performed using Random Parameter Sampling which efficiently explores the hyperparameter space without trying all combinations, and a Bandit early stopping policy was applied to stop runs that were 20% worse than the best run, saving computation resources.

Model Registration: The best-performing model was saved and registered in the workspace for future use.

## AutoML
The AutoML experiment generated a Voting Ensemble model as the best-performing solution among around 30 models such as LightGBM, XGBoost, and Logistic Regression. Hyperparameters were automatically tuned by AutoML across the ensemble, optimizing for maximum accuracy.

## Pipeline comparison
The HyperDrive pipeline achieved a best accuracy of 0.888 using a single Logistic Regression model with hyperparameters C = 0.1 and max_iter = 20. In contrast, the AutoML pipeline reached a higher accuracy of 0.908 by creating a Voting Ensemble of multiple models, including LightGBM, XGBoost, and Logistic Regression.

AutoML not only outperforms HyperDrive in accuracy but also simplifies experimentation by automatically testing and combining multiple models. It reduces implementation and tuning effort compared to manually configuring HyperDrive, while also enabling faster iteration and deployment on the Azure platform.

## Future work
Future experiments could include more detailed Exploratory Data Analysis (EDA) to better understand feature distributions and correlations, which may help in selecting or engineering more informative features.
