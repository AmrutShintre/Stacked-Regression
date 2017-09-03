####################################################################################
#
# Kaggle Competition: https://www.kaggle.com/c/mercedes-benz-greener-manufacturing
# Sponsor : Mercedes-Benz Greener Manufacturing
# Author: Amrut Shintre
#
####################################################################################

#####################
# Importing Libraries
#####################

import numpy as np
import random
import gc
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd 
from sklearn.linear_model import LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_array
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split


####################
# Importing datasets
####################

# Training Dataset
train = pd.read_csv('/Users/amrutshintre/Downloads/train.csv')

# Testing Dataset
test = pd.read_csv('/Users/amrutshintre/Downloads/test.csv')

# ----------------------------------------Basic Inspection-------------------------------------------

##################
# Basic Inspection
##################


# glimpse 
print (train.head())
print (test.head())
print (train.tail())
print (train.tail())

# Shape
print (train.shape)
print (test.shape)

# Summary

print (train.describe())
print (test. describe())

################
# Missing Values
################

# Checking for missing values

column_list = train.columns.values.tolist()
missing_values = pd.DataFrame()
missing_values['Columns'] = column_list
for i in column_list:
    missing_values['No. of missing values in train data'] = train[i].isnull().values.ravel().sum()
    missing_values['No. of missing values in test data'] = train[i].isnull().values.ravel().sum()

# There are no missing values

# ----------------------------------------Feature Engineering-------------------------------------------


#######################
# Categorical Variables
#######################

# Extracting the columns having categorical Variables.

column_list_tr = train.columns
column_list_test = test.columns
categorical_columns = []
for i in column_list_tr:
    if train[i].dtype == 'O':
        categorical_columns.append(i)

# Label Encoding

for c in categorical_columns:
    lbl = LabelEncoder()
    lbl.fit(list(train[c].values) + list(test[c].values))
    train[c] = lbl.transform(list(train[c].values))
    test[c] = lbl.transform(list(test[c].values))

###############
# Decomposition
###############

n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))
srp_results_test = srp.transform(test)

#save columns list before adding the decomposition components

usable_columns = list(set(train.columns) - set(['y']))

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]

#usable_columns = list(set(train.columns) - set(['y']))

y_train = train['y'].values
y_mean = np.mean(y_train)
id_test = test['ID'].values
#finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays) 
finaltrainset = train[usable_columns].values
finaltestset = test[usable_columns].values

# ----------------------------------------Model-------------------------------------------

####################
# XGBOOST Parameters
####################

xgb_params = {
    'n_trees': 520, 
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.93,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}
# NOTE: Make sure that the class is labeled 'class' in the data file

################
# XGBoost Matrix 
################

train = train.drop('ID', axis = 1)
test = test.drop('ID', axis = 1)
dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)


##################
# Cross-Validation  
##################

def docv(param, iterations, nfold):
    model_CV = xgb.cv(params = param, num_boost_round = iterations, nfold = nfold, dtrain = dtrain, seed = random.randint(1, 10000), early_stopping_rounds = 100, maximize = False, verbose_eval = 50)
    gc.collect()
    best = min(model_CV['test-rmse-mean'])
    best_iter = model_CV.shape[0]
    print (best)
    return (best_iter)

#########
# Testing  
#########

def doTest(param, iteration):
    X_tr, X_val, y_tr, y_val = train_test_split(train.drop('y', axis = 1), y_train, test_size = 0.2, random_state = random.randint(1,1000))
    watchlist = [(xgb.DMatrix(X_tr, y_tr), 'train'), (xgb.DMatrix(X_val, y_val), 'validation')]
    model = xgb.train(params = param, dtrain = xgb.DMatrix(X_tr, y_tr), num_boost_round = iteration, evals = watchlist, verbose_eval = 50, early_stopping_rounds = 100)
    #score = metrics.r2_score(y_val, model.predict(xgb.DMatrix(X_val)))
    predicted_class = model.predict(dtest)
    #print (score)
    return (predicted_class)

#########
# Bagging
#########

def Bagging(N, params, best_iter):
    for i in range(N):
        param = params
        p = doTest(param, best_iter)
        if i == 0:
            preds = p.copy()
        else:
            preds = preds + p
    predictions = preds/N
    predictions = pd.DataFrame(predictions)
    return (predictions)
num_boost_rounds = 1250
# train model
#model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
#y_pred = model.predict(dtest)

cvmodel = docv(xgb_params, 10000, 2)

predicted_class = doTest(xgb_params, cvmodel)


################
# Stacked Models
################

class StackingEstimator(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self
    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed

stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
    LassoLarsCV()

)


stacked_pipeline.fit(finaltrainset, y_train)
results = stacked_pipeline.predict(finaltestset)

# ----------------------------------------Submission--------------------------------------------------

#Average the preditionon test data  of both models then save it on a csv file

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = predicted_class*0.75 + results*0.25
sub.to_csv('submission.csv', index=False)

# -------------------------------------------- Project Layout ---------------------------------------------

# 1) Importing essential libraries and dataset
# 2) Basic inspection
# 3) Feature Engineering
# 4) Building a Model and trying out different models
# 5) Parameter Tuning
# 6) Stacking
# 7) Submission

