# Generative-Boosting
The source code for generative boosting. In the code, we use our private dataset which is a 2d array, the horizontal axis and  vertical axis represents for time step and features, respectively. Moreover, in our dataset, it contains 8 features: patient number, age, gender, SpO2, Resp_Num, Skin_Num, Systolic Blood Pressure and Heart Rate (index 0 to 7).If you want to implement generative boosting on your own dataset, please change the code accordingly.
The code consists of 4 parts:
1. main.py -> The main function.
2. support_function.py -> Some supporting functions (e.g., mutual information based clustering algorithm and random split function at patient level).
3. Sample_Data.py -> Sample data via an observation window and normalize the sampled data.
4. Generation_Training.py -> Generate synthetic data & Training 
