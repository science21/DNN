# -*- coding: utf-8 -*-
'''
Created on Fri Jun 22 23:33:37 2018

@author: Jianyong Wu, Ph.D
'''
#Set the directory of workspace

#Import nessary python libary
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Import data
ec= pd.read_csv('all.csv')

predictorc=[ 'Well_age',  'Dist_wellatrine', 'Unsan100m', 'SAN100m', 'Totlatr100m', 'Depth',
       'Pop100m', 'URBAN', 'WATER', 'AGR',  'r7', 't7', 'Rain', 'Tair','hr30', 't30', 'hr3']

predictor=[ 'CONCRETE_B','Well_age',  'Dist_wellatrine', 'NEAREST_WA', 'LATRINE_TY','Unsan100m', 'SAN100m', 'Totlatr100m', 'Depth',
       'Pop100m', 'URBAN', 'WATER', 'AGR',  'r7', 't7', 'Rain', 'HR', 'Tair','month','hr30', 't30', 'hr3']


total_variable=['ecdetect'] +predictor
EC=ec[total_variable]

EC['CONCRETE_B']=EC['CONCRETE_B'].astype(str)
EC['NEAREST_WA']=EC['NEAREST_WA'].astype(str)
EC['LATRINE_TY']=EC['LATRINE_TY'].astype(str)
EC['HR']=EC['HR'].astype(str)
EC['month']=EC['month'].astype(str)

EC[predictorc]=EC[predictorc].apply(lambda x: (x - x.min()) / (x.max() - x.min())) #scale predictor

#continous features
wellagef=tf.feature_column.numeric_column('Well_age')

dist_well_latrinef=tf.feature_column.numeric_column('Dist_wellatrine')
unsan100mf=tf.feature_column.numeric_column('Unsan100m')
san100mf=tf.feature_column.numeric_column('SAN100m')
totlatr100mf=tf.feature_column.numeric_column('Totlatr100m')
depthf=tf.feature_column.numeric_column('Depth')
pop100mf=tf.feature_column.numeric_column('Pop100m')
depthf=tf.feature_column.numeric_column('Depth')
urbanf=tf.feature_column.numeric_column('URBAN')
waterf=tf.feature_column.numeric_column('WATER')
agrf=tf.feature_column.numeric_column('AGR')
r7f=tf.feature_column.numeric_column('r7')
t7f=tf.feature_column.numeric_column('t7')
rainf=tf.feature_column.numeric_column('Rain')
tempf=tf.feature_column.numeric_column('Tair')
hr30f=tf.feature_column.numeric_column('hr30')
hr3f=tf.feature_column.numeric_column('hr3')
t30f=tf.feature_column.numeric_column('t30')


#categorical features
monthf=tf.feature_column.categorical_column_with_hash_bucket('month',hash_bucket_size=12)
hrf=tf.feature_column.categorical_column_with_hash_bucket('HR',hash_bucket_size=5)
concretef=tf.feature_column.categorical_column_with_hash_bucket('CONCRETE_B',hash_bucket_size=5)
nearwaterf=tf.feature_column.categorical_column_with_hash_bucket('NEAREST_WA',hash_bucket_size=5)
latrinetypef=tf.feature_column.categorical_column_with_hash_bucket('LATRINE_TY',hash_bucket_size=5)
monthfeb=tf.feature_column.embedding_column(monthf, dimension=12)
hrfeb=tf.feature_column.embedding_column(hrf, dimension=2)
concretefeb=tf.feature_column.embedding_column(concretef, dimension=2)
nearwaterfeb=tf.feature_column.embedding_column(nearwaterf, dimension=3)
latrinetypefeb=tf.feature_column.embedding_column(latrinetypef, dimension=4)

# dataset for predictors
x_data =EC.drop('ecdetect', axis=1)

#dataset for the depend variable
labels=EC['ecdetect']

#split the data into traning set and testing set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test =train_test_split(x_data, labels, test_size=0.02, random_state=40)


#select predictors
feat_column=[concretefeb,   unsan100mf,  totlatr100mf, pop100mf, urbanf, waterf,  agrf,  hr30f,  t30f, r7f, rainf, tempf]

#input function
input_func=tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=8, num_epochs=2000, shuffle=True)

#create model
dnn_model = tf.estimator.DNNClassifier(hidden_units=[20], feature_columns=feat_column, n_classes=2)


dnn_model.train(input_fn = input_func, steps=2000)

#evalate model
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test, y=y_test, batch_size=8, num_epochs=1, shuffle=False)

dnn_model.evaluate(eval_input_func)

pred=dnn_model.predict(eval_input_func)

#get predicted values
a=list(pred)
ids = [y['class_ids'] for y in a if 'class_ids' in y]




    
