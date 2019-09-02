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
	
	train_data, test_data = random_split(feature,0.8)   #training 80%, testing 20%

	headers = feature.columns
	train_data = pd.DataFrame(train_data,columns = headers)

	if MI_split ==1:
		MI_order = MI_selection(train_data)
		vali_data = []
		new_train_data = []
		for i in range(0,len(MI_order)):
			sample = MI_order[i]
			if i%Scaler==0:
				if len(vali_data)==0:
					vali_data = train_data[train_data['Subject No']==sample]
				else:
					vali_data = np.vstack((vali_data,train_data[train_data['Subject No']==sample]))			
			else:
				if len(new_train_data)==0:
					new_train_data = train_data[train_data['Subject No']==sample]
				else:
					new_train_data = np.vstack((new_train_data,train_data[train_data['Subject No']==sample]))

		train_data = new_train_data

	else:
		train_data,vali_data = random_split(train_data,0.75)   #training 60%, validation 20%

	print("-------Random Split Done----------")

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
			vali_data_p2 = vali_data[i:i+window_size,1:8]
			vali_data_label = vali_data[i+window_size,3:8]

			if len(vali_data_p3)==0:
				vali_data_p3 = vali_data_p2
			else:
				vali_data_p3 = np.vstack((vali_data_p3,vali_data_p2))
			count = count + 1
			if len(vali_data_label_p3)==0:
				vali_data_label_p3 = vali_data_label
			else:
				vali_data_label_p3 = np.vstack((vali_data_label_p3,vali_data_label))

	count_vali = count
	vali_data = vali_data_p3
	vali_label = vali_data_label_p3

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
			train_data_p2 = train_data[i:i+window_size,1:8]
			train_label.append(train_data[i+window_size+time_gap-1,7])
			if len(train_data_p3)==0:
				train_data_p3 = train_data_p2
			else:
				train_data_p3 = np.vstack((train_data_p3,train_data_p2))
			count = count + 1

	train_data = train_data_p3
	count_train = count 

	print("-------Training Data Reshape Done----------")

	count = 0
	test_label = []
	test_data_p3 = []

	for i in range(0,test_data.shape[0]):
		curr_index = test_data[i,0]
		if (i+window_size+1+time_gap) < test_data.shape[0]:
			next_index = test_data[i+window_size+time_gap,0]
		else:
			next_index = -1
		if curr_index == next_index :
			test_data_p2 = test_data[i:i+window_size,1:8]
			test_label.append(test_data[i+window_size+time_gap-1,7])
			if len(test_data_p3)==0:
				test_data_p3 = test_data_p2
			else:
				test_data_p3 = np.vstack((test_data_p3,test_data_p2))
			count = count + 1
	test_data = test_data_p3
	count_test = count

	print("-------Test Data Reshape Done----------")

	scaler = MinMaxScaler(feature_range=(0,1))
	train_data = train_data.astype("float32")
	train_data = scaler.fit_transform(train_data)

	test_data = test_data.astype("float32")
	test_data = scaler.fit_transform(test_data)

	vali_data = vali_data.astype('float32')
	vali_data = scaler.fit_transform(vali_data)

	train_data = np.reshape(train_data,(count_train,window_size,7))
	test_data = test_data.reshape(count_test,window_size,7)
	vali_data = vali_data.reshape(count_vali,window_size,7)

	vali_label = np.reshape(vali_label,(count_vali,5))
	vali_label = scaler.fit_transform(vali_label).tolist()
	vali_label = np.reshape(vali_label,(count_vali,5))

	print("-------Normalization Done----------")	

	return train_data,train_label,test_data,test_label,vali_data,vali_label




