import logging
import catboost as cb
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class CLVPredictor():
    
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        cat_features,
        models):
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cat_features = cat_features
        self.regressors = {}
        self.predictions = {}
        self.eval_dict = {}
        self.eval_dict["train"] = {}
        self.eval_dict["test"] = {}
        self.models = models
        
        for model in models:
            self.predictions[model] = {}
            self.eval_dict["train"][model] = {}
            self.eval_dict["test"][model] = {}
            
            #Initialize predictors
            if model=="catboost":
                self.regressors[model] = cb.CatBoostRegressor(n_estimators=200,
                   loss_function='RMSE',
                   learning_rate=0.4,
                   depth=3, task_type='CPU',
                   random_state=1,
                   verbose=False)
            elif model=="linear_regression":
                self.regressors[model] = LinearRegression()
            elif model=="decision_tree":
                self.regressors[model] = DecisionTreeRegressor(random_state = 0)
                
        self.prepare_categorical_variables()
        
            
    def prepare_categorical_variables(self):

        #Poorl for catboost
        if "catboost" in self.models:
            self.pool_train = cb.Pool(self.X_train, self.y_train,
                      cat_features = self.cat_features)
            self.pool_test = cb.Pool(self.X_test, cat_features = self.cat_features)
        
        #Dummies for linear regression and regression tree
        if "linear regression" or "decision tree" in self.models:
            X_train_dummy = self.X_train
            X_train_dummy["month_start"] = X_train_dummy["month_start"].apply(str)
            X_train_dummy["day_start"] = X_train_dummy["day_start"].apply(str)
            X_train_dummy = pd.get_dummies(data=X_train_dummy)

            X_test_dummy = self.X_test
            X_test_dummy["month_start"] = X_test_dummy["month_start"].apply(str)
            X_test_dummy["day_start"] = X_test_dummy["day_start"].apply(str)
            X_test_dummy = pd.get_dummies(data=X_test_dummy)
        
            self.X_train_dummy = X_train_dummy
            self.X_test_dummy = X_test_dummy
        
    
    def fit_regressors(self):

        #Train all regressors
        for model in self.regressors.keys():
            if model=="catboost":
                self.regressors[model].fit(self.pool_train)           
            elif model in ["linear_regression", "decision_tree"]:
                self.regressors[model].fit(self.X_train_dummy, self.y_train)
        
    
    def predict_w_all_regressors(self):
        
        #Predict for train and test set with all regressors
        #Save predictions in predictions attribute
        for model in self.models:
            if model == "catboost":
                self.predictions[model]["train"] = self.regressors[model].predict(self.pool_train)
                self.predictions[model]["test"] = self.regressors[model].predict(self.pool_test)
            elif model in ["linear_regression", "decision_tree"]:
                self.predictions[model]["train"] = self.regressors[model].predict(self.X_train_dummy)
                self.predictions[model]["test"] = self.regressors[model].predict(self.X_test_dummy)
            elif model == "baseline_median":
                self.predictions[model]["train"] = [np.median(self.y_train)]*len(self.y_train)
                self.predictions[model]["test"] = [np.median(self.y_train)]*len(self.y_test)
            elif model == "baseline_mean":
                self.predictions[model]["train"] = [np.mean(self.y_train)]*len(self.y_train)
                self.predictions[model]["test"] = [np.mean(self.y_train)]*len(self.y_test)
        
        
    def get_metrics_all(self, dataset):
        
        #Get RMSE and MAE for all regressors
        #Save metrics in eval_dict attribute
        for model in self.models:
            if dataset=="train":
                self.eval_dict[dataset][model]["RMSE"] = mean_squared_error(self.y_train, self.predictions[model][dataset], squared=False)            
                self.eval_dict[dataset][model]["MAE"] = mean_absolute_error(self.y_train, self.predictions[model][dataset])
            elif dataset=="test":
                self.eval_dict[dataset][model]["RMSE"] = mean_squared_error(self.y_test, self.predictions[model][dataset], squared=False)            
                self.eval_dict[dataset][model]["MAE"] = mean_absolute_error(self.y_test, self.predictions[model][dataset])

        
    def compare_regression_performance_all(self):
        
        logger.warning("Fit all models")
        self.fit_regressors()
        logger.warning("Predict for train and test set with all models")
        self.predict_w_all_regressors()
        
        for dataset in self.eval_dict.keys():
            self.get_metrics_all(dataset)            
            metrics_df = pd.DataFrame(self.eval_dict[dataset])
            logger.warning(f"The metrics of the different models on the {dataset} dataset are as follows")
            logger.warning(f"{metrics_df}")