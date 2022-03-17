# Helper libraries
import os
import errno
import plotly
import pickle
import numpy as np
import pandas as pd
from functools import reduce

# Used Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Validation and Evaluation
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import precision_score, recall_score

# Optuna, hyperparameter optimization tool
import optuna
from optuna.visualization import plot_edf
from optuna.visualization import plot_slice
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_optimization_history
from optuna.integration import XGBoostPruningCallback, LightGBMPruningCallback


class HyperTune(object):

    """Third party class for Hyperparameter Optimization tool (based on Optuna)
    
    Constants
    ---------
    RANDOM_STATE : int
        initialized random seed
    N_JOBS : int
        initialized njobs to activate cores to proccess

    Methods
    -------
    check_dataframe(y_train): 
        Static method to convert target variable to DataFrame
    
    create_model(self, trial, model_abbrev, imbalance_ratio): 
        Selection of parameters based on specified model
        
    objective(self, trial, X, y, current_model, n_splits, n_repeats, imbalance_ratio)
        Objective function to optimize given metric
        
    fit(self, X, y, n_splits = 5, n_repeats = 2, n_trials = 10)
        Fit the model to data matrix X and target

    Examples
    --------
    >>>  # Create HyperTune object
    >>>  test = HyperTune()

    >>>  # Fit to run optuna for specified models
    >>>  test.fit(X_train, y_train, n_splits = 5, n_repeats = 2, 
    ...           n_trials = 10)

    >>>  # Get model results as a dictionary 
    >>>  model_results = test.study_group

    >>>  # Get best trial of each model
    >>>  best_df = test.total_best_df
    
    >>>  # Get tuned parameters of specified model
    >>>  all_params = test.best_params
    >>>  best_params = all_params['XGBoost']
         
    '''  To get Different model parameters, you may use the following dictionary keys:
        'Logistic Regression'
        'Decision Tree'
        'Random Forest'
        'XGBoost'
        'LightGBM'
    '''
    
    >>>  model = XGBClassifier(**best_params, random_state = 42)
    >>>  model.fit(X_train, y_train.values.ravel())

    Visualization
    -------------
    >>> plot_optimization_history(model_results['XGBoost'])
    >>> plot_parallel_coordinate(model_results['XGBoost'])
    >>> plot_param_importances(model_results['XGBoost'])
    >>> plot_slice(model_results['XGBoost'])
    >>> plot_edf(model_results['XGBoost'])

    Saving
    ------
    # Save optuna to specific path
    >>> pickle_name = 'pickled_optuna.p'

    # Write optuna as pickle
    >>> with open(pickle_name, 'wb') as f:
    >>>     pickle.dump(test, f)

    # Read pickled optuna
    >>> with open(pickle_name, 'rb') as f:
    >>>     test = pickle.load(f)
    """

    RANDOM_STATE = 42
    N_JOBS = -1
    
    # Dictionary to collect all model results
    # Toggle comment to run desired models
    models_abbreviation = {
                            'Logistic Regression': 'lg',
#                             'Decision Tree': 'dt',
#                             'Random Forest': 'rf',
#                             'XGBoost': 'xgb',
#                             'LightGBM': 'lgb'
    }
    
    def __init__(self, study_group = dict(), total_best_df = pd.DataFrame(), best_params = dict()):
        """Constructs all the necessary attributes for collecting models and its parameters
        
        Parameters
        ----------
        study_group : dict, Possible array, list or DataFrame

        total_best_df : DataFrame, Best trial of each model

        best_params : Tuned parameters of specified model
        """
        self.study_group = study_group
        self.total_best_df = total_best_df
        self.best_params = best_params

        
    @staticmethod
    def check_dataframe(y_train) :
        """Static method to convert target variable to DataFrame

        Parameters
        ----------
        y_train : numpy.ndarray or df, Possible array, list or DataFrame

        Returns
        -------
        y_train : DataFrame, One column target DataFrame
        """
        if not isinstance(y_train, pd.DataFrame):
            y_train_df = pd.DataFrame(y_train)
            return y_train_df
        else:
            return y_train
        
    
    def create_model(self, trial, model_abbrev, imbalance_ratio):
        """ Selection of parameters based on specified model
        """
        if model_abbrev == 'lg':
            
            lg_params = {
            
                'C': trial.suggest_float('C', 0.001, 10.0),
                'solver': trial.suggest_categorical("solver", ["sag", "saga", "liblinear", "newton-cg", "lbfgs"]),
                'class_weight': trial.suggest_categorical('class_weight', ["balanced", None]),
                'max_iter': 600,
                'random_state': self.RANDOM_STATE,
                'n_jobs': self.N_JOBS
            }

            if lg_params['solver'] == "saga": 
                lg_params['penalty'] = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", 'none'])
                lg_params['n_jobs'] = self.N_JOBS
                    
                if lg_params['penalty'] == 'elasticnet':
                    lg_params['l1_ratio'] = 0.5
                else:
                    lg_params['l1_ratio'] = None
                    
            elif lg_params['solver'] == "liblinear":
                lg_params['penalty'] = trial.suggest_categorical("penalty_1", ["l1", "l2"])
                lg_params['n_jobs'] = None

            else:
                # 'penalty' parameter isn't relevant for this solver,
                # so we always specify 'l2' as the dummy value.
                lg_params['penalty'] = trial.suggest_categorical("penalty_2", ["l2", 'none'])
                lg_params['n_jobs'] = self.N_JOBS

            model = LogisticRegression(**lg_params)
            
        if model_abbrev == 'dt':
            dt_params = {
                'criterion' : trial.suggest_categorical("criterion", ['gini', 'entropy']),
                'splitter' : trial.suggest_categorical("splitter", ['best', 'random']),
                'max_depth' : trial.suggest_int("max_depth", 1, 9),
                'min_samples_split' : trial.suggest_int("min_samples_split", 2, 20), 
                'min_samples_leaf' : trial.suggest_int("min_samples_leaf", 1, 20), 
                'max_features': trial.suggest_categorical("max_features", ['auto', 'sqrt', 'log2']),
                'class_weight': trial.suggest_categorical("class_weight", ['balanced', None]),
                'random_state' : self.RANDOM_STATE
            } 
            
            model = DecisionTreeClassifier(**dt_params)
            
        if model_abbrev == 'rf':
            rf_params = {
                'n_estimators': trial.suggest_int("n_estimators", 10, 500),
                'criterion' : trial.suggest_categorical("criterion", ['gini', 'entropy']),
                'max_depth' : trial.suggest_int("max_depth", 1, 9),
                'min_samples_split' : trial.suggest_int("min_samples_split", 2, 20), 
                'min_samples_leaf' : trial.suggest_int("min_samples_leaf", 1, 20), 
                'max_features': trial.suggest_categorical("max_features", ['auto', 'sqrt', 'log2']),
                'class_weight': trial.suggest_categorical("class_weight", ['balanced','balanced_subsample', None]),
                'random_state' : self.RANDOM_STATE,
                'n_jobs' : self.N_JOBS
            } 
            
            model = RandomForestClassifier(**rf_params)
            
        if model_abbrev == 'xgb':
            xgb_params = {
                'booster': trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                # L2 regularization weight.
                'lambda': trial.suggest_float("lambda", 1e-4, 1.0, log=True),
                # L1 regularization weight.
                'alpha': trial.suggest_float("alpha", 1e-4, 1.0, log=True),
                # sampling ratio for training data.
                'max_depth' : trial.suggest_int('max_depth', 2, 10), 
                'subsample': trial.suggest_float("subsample", 0.2, 1.0),
                'min_child_weight' : trial.suggest_int('min_child_weight', 1, 12),
                'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
                'learning_rate' : trial.suggest_loguniform('learning_rate', 0.005, 0.5),
                'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.1, 1, 0.01),
                'scale_pos_weight' : trial.suggest_int("scale_pos_weight", 1, imbalance_ratio, step = imbalance_ratio-1),
                'use_label_encoder' : False,
                'seed': self.RANDOM_STATE
                #'nthread' : self.N_JOBS
            }

            if xgb_params["booster"] in ["gbtree", "dart"]:
                xgb_params["max_depth"] = trial.suggest_int("max_depth", 2, 10, step=2)
                # minimum child weight, larger the term more conservative the tree.
                xgb_params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 12)
                xgb_params["eta"] = trial.suggest_float("eta", 1e-4, 1.0, log=True)
                # defines how selective algorithm is.
                xgb_params["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
                xgb_params["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

            if xgb_params["booster"] == "dart":
                xgb_params["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
                xgb_params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
                xgb_params["rate_drop"] = trial.suggest_float("rate_drop", 1e-4, 1.0, log=True)
                xgb_params["skip_drop"] = trial.suggest_float("skip_drop", 1e-4, 1.0, log=True)

            model = XGBClassifier(**xgb_params)
            
        if model_abbrev == 'lgb':
            
            lgb_params = {
                'n_estimators': trial.suggest_int("n_estimators", 10, 500),
                'objective': 'binary',
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
                #'reg_lambda': trial.suggest_int('reg_lambda', 1e-2,1e-1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2,0.8),
                'subsample': trial.suggest_float('subsample', 0.4,0.8),
                'subsample_freq': trial.suggest_categorical('subsample_freq',[1,2]),
                'learning_rate': trial.suggest_float('learning_rate',5e-3,1),
                'max_depth': trial.suggest_int('max_depth', 1, 9),
                'num_leaves' : trial.suggest_int('num_leaves', 2, 256),
                #'importance_type': 'gain',
                'boosting_type': trial.suggest_categorical('boosting_type',['gbdt', 'dart', 'rf']), # 'goss'
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'class_weight' : trial.suggest_categorical('class_weight',['balanced', None]),
                'random_state': self.RANDOM_STATE,
                'n_jobs' : self.N_JOBS
            }
    
            model = LGBMClassifier(**lgb_params)
            
        if trial.should_prune():
            raise optuna.TrialPruned()
            
        return model
    
    
    def objective(self, trial, X, y, current_model, n_splits, n_repeats, imbalance_ratio = 1)-> float:
        """Objective function to optimize given metric.

        Parameters
        ----------
        trial : process of evaluating an objective function 
            This object is passed to an objective function 
            and provides interfaces to get parameter suggestion.
        
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification).
        
        current_model : str, abbreviations of iterated models.
            Currently there are 5 models.
        
        n_splits : int, default=5 
            number of folds in RepeatedStratifiedKFold.
                   
        n_repeats : int, default=3
            number of times cross-validator needs to be repeated.
                    
        imbalance_ratio : int, default=1
            The total number of negative instances divided by total positive instances.
            sum(negative instances) / sum(positive instances)
        
        Returns
        -------
        test_score_mean : Average score of calculated metric 
        """
        model = self.create_model(trial, current_model, imbalance_ratio)
        test_score = []
        
        # XGBoost modeling
        if current_model == 'xgb':
            rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.RANDOM_STATE)
            for train_index, test_index in rskf.split(X, y):
                X_train, X_val = X.values[train_index], X.values[test_index]
                y_train, y_val = y.values[train_index], y.values[test_index]

                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="auc",
                    early_stopping_rounds=100,
                    callbacks=[XGBoostPruningCallback(trial, "validation_0-auc")],
                )
                
                y_val_pred = model.predict(X_val)
                test_score.append(f1_score(y_val, y_val_pred))
            test_score_mean = reduce(lambda a, b: a+b, test_score)/len(test_score)
            
        # LGBM modeling
        elif current_model == 'lgb':
            rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.RANDOM_STATE)
            for train_index, test_index in rskf.split(X, y):
                X_train, X_val = X.values[train_index], X.values[test_index]
                y_train, y_val = y.values[train_index], y.values[test_index]
                
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="auc",
                    early_stopping_rounds=100,
                    callbacks=[LightGBMPruningCallback(trial, "auc")],
                )
                
                y_val_pred = model.predict(X_val)
                test_score.append(f1_score(y_val, y_val_pred))
            test_score_mean = reduce(lambda a, b: a+b, test_score)/len(test_score)
            
        else:
            rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.RANDOM_STATE)
            for train_index, test_index in rskf.split(X, y):
                X_train, X_val = X.values[train_index], X.values[test_index]
                y_train, y_val = y.values[train_index], y.values[test_index]

                model.fit(X_train, y_train.ravel())
                y_val_pred = model.predict(X_val)
                test_score.append(f1_score(y_val, y_val_pred))
            test_score_mean = reduce(lambda a, b: a+b, test_score)/len(test_score)

        return test_score_mean
    
    
    def fit(self, X, y, n_splits = 5, n_repeats = 2, n_trials = 10):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification).
            
        n_splits : int, default=5 
            number of folds in RepeatedStratifiedKFold.
                   
        n_repeats : int, default=3
            number of times cross-validator needs to be repeated.
        
        n_trials : int, default=15
            The number of trials.
        
        """
        # Change target type to dataframe
        y = self.check_dataframe(y)
        best_results_df = pd.DataFrame()
        self.total_best_df = pd.DataFrame()
        
        # Imbalance ratio for class weight
        imbalance_ratio = int(sum(y.values.ravel() == 0) / sum(y.values.ravel() == 1))

        for _, (model_name, model_abrv) in enumerate(self.models_abbreviation.items()):

            print('*' * 40, '\n')
            print(model_name.upper(), '\n')
            print('*' * 40)

            self.study_group[model_name] = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed = self.RANDOM_STATE),
                                pruner=optuna.pruners.MedianPruner())
    
            self.study_group[model_name].optimize(lambda trial: self.objective(trial,
                                                     X,
                                                     y,
                                                     model_abrv,
                                                     n_splits,
                                                     n_repeats,
                                                     imbalance_ratio), n_trials, show_progress_bar = True)  # n_jobs=self.N_JOBS


            current_model = self.study_group[model_name]
            self.best_params[model_name] = current_model.best_trial.params
            best_score = current_model.best_value

            # For Logistic Regression
            if model_abrv == 'lg':
                key_index = list(current_model.best_trial.params.keys())
                self.best_params[model_name]['penalty'] = self.best_params[model_name].pop(key_index[3])
               
            # Get best result row from each model
            best_results_df = current_model.trials_dataframe()[best_score == current_model.trials_dataframe()['value']]
            best_results_df.insert(1, 'Model', model_name)
            self.total_best_df = pd.concat([self.total_best_df, best_results_df], axis = 0, ignore_index = True)
            
            print('*' * 40, '\n')
            print(model_name.upper(), '\n')
            print("Number of finished trials: ", n_trials)
            print('Best Value: {:.4f}'.format(best_score))
            print("Best trial:")
            print("  Params: ")
            for key, value in self.best_params[model_name].items():
                print("    {:<12s}: {}".format(key, value))
            print()
            print('*' * 40, '\n')
            
        print('Hypertuning is Completed!')        