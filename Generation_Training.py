#!/usr/bin/env python

from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.layers import Input, Dense,LSTM,Dropout
from keras.models import Model, Sequential
import keras
import os
from sklearn.metrics import mean_squared_error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#generate synthetic data and train LSTM, finally evaluate it
def cal(train_data,train_label,test_data,test_label,vali_data,vali_label,window_size,Feature,time_gap,min_val,max_val):
	count_train, count_test, count_vali = train_data.shape[0], test_data.shape[0], vali_data.shape[0]

	for gen in range(0,1):
		result_all = []	
		result_all_test = []
		for k in range(2,Feature):
			print("generating feature "+str(k))
			test_feature_all = []
			vali_feature_all = []
			test_feature_all_test = []
			for i in range(0,count_train):
				value = train_data[i,:,k].tolist()
				if len(test_feature_all)==0:
					test_feature_all = value
				else:
					test_feature_all = np.vstack((test_feature_all,value))

			for j in range(0,count_vali):
				vali_value = vali_data[j,:,k].tolist()
				if len(vali_feature_all)==0:
					vali_feature_all = vali_value
				else:
					vali_feature_all = np.vstack((vali_feature_all,vali_value))

			vali_label_temp = np.reshape(vali_label[:,k-2],(count_vali,1)) 

			#print(vali_label_temp)

			for m in range(0,count_test):
				value = test_data[m,:,k].tolist()
				test_feature = value
				if len(test_feature_all_test)==0:
					test_feature_all_test = test_feature
				else:
					test_feature_all_test = np.vstack((test_feature_all_test,test_feature))

			test_feature_all_test = np.reshape(test_feature_all_test,(count_test,window_size,1))
			test_feature_all = np.reshape(test_feature_all,(count_train,window_size,1))
			vali_feature_all = np.reshape(vali_feature_all,(count_vali,window_size,1))

			#print(vali_feature_all)

			model_1 = Sequential()
			model_1.add(LSTM(1,input_shape=(window_size,1),activation='sigmoid',return_sequences=False,kernel_initializer='random_uniform',
	                	bias_initializer='random_uniform'))
			model_1.compile(optimizer='Adam',loss='mean_squared_error',metrics=['mae'])
			model_1.fit(vali_feature_all,vali_label_temp,epochs=1, batch_size=20,verbose=1)
			result = model_1.predict(test_feature_all)
			result_all.append(result)
			result_test = model_1.predict(test_feature_all_test)
			result_all_test.append(result_test)
		
		result_all_test = np.reshape(result_all_test,(5,count_test))	
		result_all = np.reshape(result_all,(5,count_train))

		for i in range(0,count_train):
			temp_data = train_data[i,:,:]
			temp_data = np.delete(temp_data, (0), axis=0)
			new_gen = []
			for k in range(0,Feature):
				value = train_data[i,:,k].tolist()
				if k<2:
					new_gen.append(value[0])
				else:
					new_gen.append(result_all[k-2,i])

			new_gen = np.reshape(new_gen,(1,Feature))
			if i == 0:
				temp_data_store = temp_data
				temp_data_store = np.vstack((temp_data_store,new_gen))
			else:
				temp_data_store = np.vstack((temp_data_store,temp_data))
				temp_data_store = np.vstack((temp_data_store,new_gen))

		train_data = np.reshape(temp_data_store,(count_train,window_size,Feature))

		print('Training Data Done')
		for i in range(0,count_test):
			temp_data = test_data[i,:,:]
			temp_data = np.delete(temp_data, (0), axis=0)
			new_gen = []
			for k in range(0,Feature):
				value = test_data[i,:,k].tolist()
				if k<2:
					new_gen.append(value[0])
				else:
					new_gen.append(result_all_test[k-2,i])

			new_gen = np.reshape(new_gen,(1,Feature))
			if i == 0:
				temp_data_store = temp_data
				temp_data_store = np.vstack((temp_data_store,new_gen))
			else:
				temp_data_store = np.vstack((temp_data_store,temp_data))
				temp_data_store = np.vstack((temp_data_store,new_gen))


		test_data = np.reshape(temp_data_store,(count_test,window_size,Feature))
		print("Testing Date Done")

	error_all = []
	model = Sequential()
	model.add(LSTM(1,input_shape=(window_size,Feature),activation='sigmoid',return_sequences=False,kernel_initializer='zeros',
	                bias_initializer='zeros'))
	model.compile(optimizer='Adam',loss='mean_squared_error',metrics=['mae'])
	model.fit(train_data,train_label,epochs=1, batch_size=20,verbose=1)

	test_label = np.reshape(test_label,(count_test,1))
	pred_result = model.predict(test_data)
	pred_result = pred_result *(max_val - min_val) + min_val
	test_label = test_label *(max_val - min_val) + min_val
	error = mean_squared_error(pred_result,test_label)
	mae_error = np.mean(np.abs((pred_result-test_label)/test_label))*100
	print("Mean Square Error is "+str(error)+"MAE is "+str(mae_error))

	return 0
