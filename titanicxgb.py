#converted from a jupyter notebook from a kaggle titanic entry
#frankenstein of ideas: Feature engineering and ensembling
import os
import pandas as pd
import numpy as np
import math as ma
import re
import sklearn
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)

df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

df_train = df_train.drop(['Ticket', 'Cabin'], axis=1)
df_test = df_test.drop(['Ticket', 'Cabin'], axis=1)
combine = [df_train, df_test]

# Store our passenger ID for easy access
PassengerId = df_test['PassengerId']

#feature engineering
def makeOrdinal(df,label,show_dict):
    filtered = df.sort_values([label], ascending = [True])
    df_filtered = filtered.groupby(label).first().reset_index()
    strcat_dict = {}
    
    for i,row in df_filtered.iterrows():
        strcat_dict[row[label]] = i 
    
    if show_dict:
        print(strcat_dict)
    
    for j,row in df.iterrows():
        df.at[j,label] = strcat_dict.get(row[label])
    
    return df

age_comb_dropped = []

for df in combine:
    makeOrdinal(df,"Sex",False)
    df_dropped = df[['Sex','Age','Pclass']].dropna()
    age_comb_dropped.append(df_dropped)

X = age_comb_dropped[0][['Sex','Pclass']]
y = age_comb_dropped[0]['Age']

train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=1)

rf_model_age = RandomForestRegressor(random_state=1,n_estimators=10)
rf_model_age.fit(train_X,train_y)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for df in combine:
    #male = 1, female = 0
    makeOrdinal(df,"Sex",False)
    #S = 2, C = 0, Q = 1
    makeOrdinal(df,"Embarked",False)
    
    #create an 'engineered feature', by using regex to extract their Title
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
    #convert the french names and the 'rare' titles like Rev or Don into general categories
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    	'Don','Dr','Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)

combine = [df_train,df_test]

#sex: female = 0, male = 1
def guessAge(sex,pclass):
    return int(rf_model_age.predict(np.array([[sex,pclass]])))

for df in combine:
    for i,row in df.iterrows():
        if ma.isnan(row['Age']):
            df.at[i,'Age'] = guessAge(row['Sex'],row['Pclass'])
combine = [df_train,df_test]

df_train['AgeBand'] = pd.cut(df_train['Age'], 5)
df_train[['AgeBand', 'Survived']].groupby(['AgeBand'],as_index=False).mean().sort_values(by='AgeBand',ascending=True)
#change the age values in the combined data set using the categories
#if the age in this entry of combine is in the bounds of the check assign it to 1,2,3 or not
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = int(0)
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = int(1)
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = int(2)
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = int(3)
    dataset.loc[ dataset['Age'] > 64, 'Age']

df_train = df_train.drop(['AgeBand'], axis=1)
combine = [df_train, df_test]

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

#find the most frequent departure
#we will use this to fill the missing data in this instance
freq_port = df_train.Embarked.dropna().mode()[0]

#fill the empty 'Embarked' rows with the most frequent
#then print the survival 
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

df_train[['Embarked', 'Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived', ascending=False)

df_test['Fare'].fillna(df_test['Fare'].dropna().median(), inplace=True)
df_train['FareBand'] = pd.qcut(df_train['Fare'], 4)
df_train[['FareBand', 'Survived']].groupby(['FareBand'],as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

df_train = df_train.drop(['FareBand'], axis=1)
df_train = df_train.drop(['Name', 'PassengerId'], axis=1)
df_test = df_test.drop(['Name'], axis=1)
combine = [df_train, df_test]

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

combine = [df_train, df_test]

ntrain = df_train.shape[0]
ntest = df_test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(n_splits = NFOLDS, random_state = SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)

def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = df_train['Survived'].ravel()
df_train = df_train.drop(['Survived'], axis=1)
copy_df_test = df_test.copy()
copy_df_test = copy_df_test.drop(['PassengerId'],axis=1)
x_train = df_train.values # Creates an array of the train data
x_test = copy_df_test.values # Creats an array of the test data

# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("Level 1 Training is complete")

rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)

#outputs I got, for reproduction purposes
rf_features = [0.13260013, 0.22765906, 0.02992623, 0.03908793, 0.02088596, 0.08141679,
 0.02863122, 0.274605,   0.08449276, 0.06539506, 0.01529985]
et_features = [0.14680118, 0.40326415, 0.02266595, 0.03404534, 0.01734894, 0.07033382,
 0.02984221, 0.17960615, 0.02936249, 0.04327344, 0.02345634]
ada_features = [0.014, 0.114, 0.042, 0.134, 0.114, 0.05,  0.018, 0.25,  0.078, 0.182, 0.004]
gb_features = [0.14403243, 0.01577795, 0.02833046, 0.02015712, 0.00816722, 0.05770074,
 0.03543125, 0.53664249, 0.03808127, 0.11055575, 0.00512332]


cols = df_train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_features,
     'Extra Trees  feature importances': et_features,
      'AdaBoost feature importances': ada_features,
    'Gradient Boost feature importances': gb_features
    })

# axis = 1 computes the mean row-wise
feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) 

base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel()
    })

x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)

StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions })

StackingSubmission.to_csv("StackingSubmission.csv", index=False)