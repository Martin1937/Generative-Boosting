#!/usr/bin/env python

from mdentropy import mutinf
import numpy as np
import random

#MI based clustering algorithm
#Input: DataFrame, Output: Ranked Scores for Each Patient (High - Low)
def MI_selection(all_data):
	all_MI = []
	original_number = np.unique(all_data.values[:,0])
	for i in range(0,len(original_number)):
		MI = 0
		sample_num = original_number[i]
		for j in range(0,len(original_number)):
			sample_num_2 = original_number[j]
			if sample_num_2 != sample_num:
				P1 = all_data[all_data['Subject No']==sample_num].T
				P2 = all_data[all_data['Subject No']==sample_num_2].T
				MI = MI + mutinf(3,P1,P2)
		all_MI.append(MI)
	order = np.argsort(all_MI)[::-1]
	new_order = original_number[order]

	return new_order


#Random Split Data Patient at Patient Level
#Input: DataFrame, Percentage of Training Data, Output:Splited Training Data & Test Data (in Numpy Form) 
def random_split(all_data,percentage):
	train_data_new = []
	test_data_new = []
	original_number = np.unique(all_data.values[:,0])
	sampled_patient_index = np.random.choice(original_number,int(len(original_number)*percentage),replace=False)
	unsampled_patient_index = np.array([i for i in original_number if i not in sampled_patient_index])
	train_data_new = all_data[all_data['Subject No'].isin(sampled_patient_index)].values
	test_data_new = all_data[all_data['Subject No'].isin(unsampled_patient_index)].values
	return train_data_new, test_data_new

'''
	order = random.sample(range(len(original_number)-1),int(len(original_number)*percentage))
	new_order = original_number[order]

	for i in range(0,len(original_number)):
		number = original_number[i]
		key = 0
		for j in range(0,len(new_order)):
			if number == new_order[j]:
				key = 1
		if key==1:
			if len(train_data_new)==0:
				train_data_new = all_data[all_data['Subject No']==number]
			else:
				train_data_new = np.vstack((train_data_new,all_data[all_data['Subject No']==number]))
		else:
			if len(test_data_new)==0:
				test_data_new = all_data[all_data['Subject No']==number]
			else:
				test_data_new = np.vstack((test_data_new,all_data[all_data['Subject No']==number]))							

	return train_data_new, test_data_new
'''
