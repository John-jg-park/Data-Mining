import pandas as pd
import sklearn as sk
import numpy as np
from scipy.fft import fft, fftfreq, rfft
import pickle

test = pd.read_csv(r'test.csv')
#print(test.shape)
data = np.empty(shape=[0, 8])
label = np.array([])
for q_arr in test.to_numpy():
    #print(q_arr)
    max_value = None
    min_value = None
    for idx, num in enumerate(q_arr):
        #print(num)
        if (max_value is None or num > max_value) :
            max_value = num
            max_idx = idx
        if (min_value is None or num < min_value) :
            min_value = num
            min_idx = idx
    diff_max1 = None
    diff_max2 = None
    for num in np.diff(q_arr):
        if (diff_max1 is None or num > diff_max1) :
            diff_max1 = num
    for num in np.diff(q_arr, n=2):
        if (diff_max2 is None or num > diff_max2) :
            diff_max2 = num
    #print(np.diff(q_arr))    
    #print(np.diff(q_arr, n=2))
  
    
    #print('max, idx:', max_value, max_idx)       
    #print('meal, idx:', meal_value, meal_idx) 
    #print('tau(minites):', abs(max_idx-min_idx)*5) #f1
    #print('dG =', (max_value-min_value)/min_value) #f2
    #print('diff_max1:', diff_max1) #f3
    #print('diff_max2:', diff_max2) #f4
    
    #print('abs fft:', abs(c))
    x = q_arr
    #print(x)
    xi = np.arange(len(x))
    mask = np.isfinite(x)
    xfiltered = np.interp(xi, xi[mask], x[mask])
    
    c = rfft(xfiltered)
    #print(c)
    #plt.plot(abs(c), "o")
    #plt.show()
    
    #----find max2 max3
    fft = abs(c)
    list_fft = fft.tolist()
    #print(max(fft))
    list_fft.remove(max(list_fft))
    #print(list_fft)
    max2=max(list_fft)
    #print('max2', max2)
    list_fft.remove(max(list_fft))
    #print(list_fft)
    max3=max(list_fft)
    #print('max3', max3)
    
    for idx, num in enumerate(abs(c)):
        #print(num)
        if max2 == num:
            max2_idx = idx
        elif max3 == num:
            max3_idx = idx
    #print('max2_idx', max2_idx)
    #print('max3_idx', max3_idx)
    
    list_feature = [abs(max_idx-min_idx)*5, (max_value-min_value)/min_value, diff_max1, diff_max2,max2,max2_idx,max3,max3_idx]
    #print(list_feature)
    array_sum = np.sum(np.array(list_feature))
    if np.isnan(array_sum) == False:
        data = np.append(data, [np.array(list_feature)], axis = 0)
        #print(data)
        #label = np.append(label, [0])
        #print(label)
    #----null test
    array_sum = np.sum(data)
    array_has_nan = np.isnan(array_sum)
    #print(array_has_nan)


print(data.shape)


#read
with open('model_pickle.pkl', 'rb') as f:
    model = pickle.load(f)

pred_model = model.predict(data)
results_df = pd.DataFrame(data=pred_model)
print(results_df.shape)
results_df.to_csv(r'Result.csv', index = False) #header=False
