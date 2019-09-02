from Sample_Data import gen
from Generation_Training import cal

if __name__ == '__main__':
	for gap in range(1,26):
		window_size = 20
		train_data,train_label,test_data,test_label,vali_data,vali_label = gen(window_size,gap,0)
		for average in range(0,30):
			print("Window size: "+str(window_size)+" Time Lag is: "+str(gap))
			test = cal(train_data,train_label,test_data,test_label,vali_data,vali_label,window_size,7,gap)