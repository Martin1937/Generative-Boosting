#!/usr/bin/env python

import numpy as np
import pandas as pd 

from support_function import random_split, MI_selection
from sklearn.preprocessing import MinMaxScaler

#Mutual Information Clustering Algorithm Sampling Scaler
Scaler = 3

#Sample time series data using sliding observation window
def gen(window_size,time_gap,MI_split):
	feature = pd.DataFrame()
	feature = pd.read_csv('Visi_20_new_hr.csv', index_col = None)
	
	train_data, test_data = random_split(feature,0.8)   #training 80%, testing 20% (in numpy form)
	cols = train_data.shape[1]

	#find min & max value for each feature in training data, use these value to normalize the test, vali data.
	min_value = np.amin(train_data,axis=0)
	max_value = np.amax(train_data,axis=0)
	train_data = train_data.astype("float32")
	scaler = MinMaxScaler(feature_range=(0,1))
	train_data = scaler.fit_transform(train_data)

	if MI_split ==1:
		#conver to dataframe to selection
		headers = feature.columns
		train_data = pd.DataFrame(train_data,columns = headers)
		MI_order = MI_selection(train_data)
		#print(MI_order)
		sampled_patient_index = MI_order[::Scaler]
		unsampled_patient_index = np.array([i for i in MI_order if i not in sampled_patient_index])
		train_data_new = train_data[train_data['Subject No'].isin(MI_order)].values
		vali_data_new = train_data[train_data['Subject No'].isin(unsampled_patient_index)].values  #(both in numpy form)
		train_data,vali_data = train_data_new, vali_data_new 
	else:
		train_data,vali_data = random_split(train_data,0.75)   #training 60%, validation 20% (in numpy from)
	print("-------Random Split Done----------")

	#normalize test & vali data using the min & max value obtained above
	test_data = test_data.astype("float32")
	vali_data = vali_data.astype("float32")
	for col in range(0,test_data.shape[1]):
		test_data[:,col] = (test_data[:,col] - min_value[col])/(max_value[col] - min_value[col])
		vali_data[:,col] = (vali_data[:,col] - min_value[col])/(max_value[col] - min_value[col])

	vali_data_p3 = []
	vali_data_label_p3 = []
	count = 0
	for i in range(0,vali_data.shape[0]):
		curr_index = vali_data[i,0]
		if (i+window_size+1+time_gap) < vali_data.shape[0]:
			next_index = vali_data[i+window_size+time_gap,0]
		else:
			next_index = -1
		if curr_index == next_index:
			if len(vali_data_p3)==0:
				vali_data_p3 = vali_data[i:i+window_size,1:cols]
			else:
				vali_data_p3 = np.vstack((vali_data_p3,vali_data[i:i+window_size,1:cols]))
			count = count + 1

			vali_data_label = vali_data[i+window_size,3:cols]
			if len(vali_data_label_p3)==0:
				vali_data_label_p3 = vali_data_label
			else:
				vali_data_label_p3 = np.vstack((vali_data_label_p3,vali_data_label))

	vali_data = vali_data_p3.reshape(count,window_size,cols-1)
	#vali_label = np.ones((count,1))
	vali_label = vali_data_label_p3.reshape(count,cols-3)
	print("-------Validation Data Reshape Done----------")

	train_data_p3 = []
	train_label = []
	count = 0
	for i in range(0,train_data.shape[0]):
		curr_index = train_data[i,0]
		if (i+window_size+1+time_gap) < train_data.shape[0]:
			next_index = train_data[i+window_size+time_gap,0]
		else:
			next_index = -1
		if curr_index == next_index :
			train_label.append(train_data[i+window_size+time_gap-1,cols-1])
			if len(train_data_p3)==0:
				train_data_p3 = train_data[i:i+window_size,1:cols]
			else:
				train_data_p3 = np.vstack((train_data_p3,train_data[i:i+window_size,1:cols]))
			count = count + 1
	train_data = train_data_p3.reshape(count,window_size,cols-1)
	print("-------Training Data Reshape Done----------")

	test_data_p3 = []
	test_label = []
	count = 0
	for i in range(0,test_data.shape[0]):
		curr_index = test_data[i,0]
		if (i+window_size+1+time_gap) < test_data.shape[0]:
			next_index = test_data[i+window_size+time_gap,0]
		else:
			next_index = -1
		if curr_index == next_index :
			test_label.append(test_data[i+window_size+time_gap-1,cols-1])
			if len(test_data_p3)==0:
				test_data_p3 = test_data[i:i+window_size,1:cols]
			else:
				test_data_p3 = np.vstack((test_data_p3,test_data[i:i+window_size,1:cols]))
			count = count + 1
	test_data = test_data_p3.reshape(count,window_size,cols-1)
	print("-------Test Data Reshape Done----------")

	return train_data,train_label,test_data,test_label,vali_data,vali_label,min_value[cols-1],max_value[cols-1]




