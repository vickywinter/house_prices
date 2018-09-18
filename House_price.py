#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 20:05:47 2018

@author: vickywinter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
import sklearn.model_selection

test=pd.read_csv('/Users/vickywinter/Documents/NYC/Machine Learning Proj/Data/all/test.csv')
train=pd.read_csv('/Users/vickywinter/Documents/NYC/Machine Learning Proj/Data/all/train.csv')

test["data_type"]="test"
train["data_type"]="train"

all_data=pd.concat([train,test])

sns.distplot(train['SalePrice'],fit=norm)
# the plot is right skew
# transfer the sales to sqrt
sns.distplot(np.sqrt(train['SalePrice']),fit=norm)
fig=plt.figure()
stats.probplot(np.sqrt(train['SalePrice']), plot=plt)

#trnsfer the sales to log, much better
sns.distplot(np.log(train['SalePrice']),fit=norm)
fig=plt.figure()
stats.probplot(np.log(train['SalePrice']), plot=plt)



# correltion between numb variablea
corrmat = train.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
# High Correlation
# OverallQual Vs SalesPrice
# GrLivArea Vs SalesPrice
# GarageCars Vs GarageArea
# GarageCars Vs SalesPrice
#GarageYrBlt vs YearBuilt
# TotRmsAbvGrd Vs 1stFlrSF
#TotRmsAbvGrd Vs. TotRmsAbvGrd




train.plot.scatter(x='LotFrontage',y='SalePrice')



all_data['Bath']=all_data['FullBath']+all_data['HalfBath']*0.25
all_data['BsmtBath']=all_data['BsmtFullBath']+all_data['BsmtHalfBath']*0.25
all_data.drop(columns=['FullBath', 'HalfBath','BsmtFullBath','BsmtHalfBath'])


missing_value=pd.DataFrame({'miss':all_data.isnull().sum()})
missing_value=missing_value.sort_values(by=['miss'],ascending=False)


all_data['PoolQC']=all_data['PoolQC'].fillna("None")
all_data['MiscFeature']=all_data['MiscFeature'].fillna("None")
all_data['Alley']=all_data['Alley'].fillna("None")
all_data['Fence']=all_data['Fence'].fillna("None")
all_data['FireplaceQu']=all_data['FireplaceQu'].fillna("None")
all_data['Fence']=all_data['Fence'].fillna("None")

all_data['GarageFinish']=all_data['GarageFinish'].fillna("None")
all_data['GarageCond']=all_data['GarageCond'].fillna("None")
all_data['GarageYrBlt']=all_data['GarageYrBlt'].fillna(all_data['YearBuilt'])
all_data['GarageQual']=all_data['GarageQual'].fillna("None")
all_data['GarageType']=all_data['GarageType'].fillna("None")
all_data['BsmtCond']=all_data['BsmtCond'].fillna("None")
all_data['BsmtExposure']=all_data['BsmtExposure'].fillna("None")

all_data['BsmtQual']=all_data['BsmtQual'].fillna("None")
all_data['BsmtFinType2']=all_data['BsmtFinType2'].fillna("None")
all_data['BsmtFinType1']=all_data['BsmtFinType1'].fillna("None")
all_data['MasVnrType']=all_data['MasVnrType'].fillna("None")
all_data['BsmtFinType2']=all_data['BsmtFinType2'].fillna("None")
all_data['BsmtFinType1']=all_data['BsmtFinType1'].fillna("None")


missing_value=pd.DataFrame({'miss':all_data.isnull().sum()})
missing_value=missing_value.sort_values(by=['miss'],ascending=False)

all_data.groupby(['Neighborhood', 'LotShape','LotConfig'])['LotFrontage'].median()



train.plot.scatter(x='LotFrontage',y='LotArea')
train.plot.bar(y='Street',x='LotShape')
sns.boxplot(x='Alley', y='LotFrontage', data=all_data)
sns.boxplot(x='Neighborhood', y='LotFrontage', data=all_data)
sns.boxplot(x='LotShape', y='LotFrontage', data=all_data)
sns.boxplot(x='LotConfig', y='LotFrontage', data=all_data)

all_data['LotFrontage']=all_data.groupby(['Neighborhood', 'LotShape','LotConfig'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))


lot=all_data[['LotFrontage','Street','Alley','LotShape','LotConfig','Neighborhood']]

notnans=lot.loc[lot['LotFrontage'].notnull()]

# Split into 75% train and 25% test
from sklearn.model_selection import train_test_split
lot_train, lot_test, y_train, y_test = train_test_split(notnans[['Street','Alley','LotShape','LotConfig','Neighborhood']], notnans[['LotFrontage']],train_size=0.75,random_state=4)

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


regr_multirf = RandomForestRegressor(max_depth=30,random_state=0)

# Fit on the train data
regr_multirf.fit(lot_train, y_train)

# Check the prediction score
score = regr_multirf.score(lot_test, y_test)
print("The prediction score on the test data is {:.2f}%".format(score*100))











#train['SalePrice'].describe()

#fig = sns.boxplot(y=train['SalePrice'])


num_data=['SalePrice','LotFrontage','LotArea','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinType2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath', 'BedroomAbvGr', 'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']

num_data_train=train[num_data]
corrmat = num_data_train.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);










list(train.columns.values)

var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
