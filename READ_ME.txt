Hello,

This the Readme file for Austin Wright coursework submissison for Machine Learning, City University of London


The raw data that was use for the coursework can be found here:

https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Along with the poster: POSTER_ML_AW.pdf
and the supplementary material: Supplementary_ML_AW.pdf 


In this folder you will find the following files: 


1) For test set validation and model run: FinalModlRun_NoTraining.m

- Run this to see how the model preforms on the test data. 

- It requires input data DataX.csv and DataY.csv

- It requires FinalLogisticRegressionModel.mat and FinalRandomForestModel.mat which are in this directory

- It automatically loads those two files if you just add this whole directory to path 




2) ML_Coursework_Master_AW.m is the script to run 3 different versions of model training and validation (Please check it out!)


3) Logistic_Regression_AW.m and Logistic_Regression_ADAM.m are functions I wrote to do logisitic regression 


4) Predict_AW.m is a function I wrote to do a prediction given some model parameters for logisitic regresiion 


5) LogReg_HyperParam_Opt.m -> Bayesian Optimization of hyperparameters for logistic regression 

	- have included a .mat file of the parameters output from this

6) RF_HyperParam_Opt.m     -> Bayesian Optimization of hyperparameters for random forest 

	- have included a .mat file of the parameters output from this
