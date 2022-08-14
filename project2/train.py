import pandas as pd
import sklearn as sk
#print('pandas ' + pd.__version__)
#print('sklearn ' + sk.__version__)


cgm_raw = pd.read_csv(r'CGMData.csv', index_col = 0, low_memory=False)
cgm_raw['datetime'] = pd.to_datetime(cgm_raw['Date']) + pd.to_timedelta(cgm_raw['Time'])
cgm_df = cgm_raw[['datetime','Sensor Glucose (mg/dL)']]

insulin_raw = pd.read_csv(r'InsulinData.csv', index_col = 0, low_memory=False)
insulin_raw['datetime'] = pd.to_datetime(insulin_raw['Date']) + pd.to_timedelta(insulin_raw['Time'])
insulin_df = insulin_raw[['datetime', 'BWZ Carb Input (grams)']]

cgm_df = cgm_df.sort_values(by='datetime')
insulin_df = insulin_df.sort_values(by='datetime')
cgm_df.reset_index(drop=True, inplace=True)


meal_df = insulin_df[insulin_df['BWZ Carb Input (grams)'] > 0]
meal_df=meal_df.iloc[:,0]
meal_df.reset_index(drop=True, inplace=True)
meal = meal_df.copy()

from datetime import datetime
from datetime import timedelta
for i in range(len(meal)-1):
    #print(meal[i])
    if meal[i] + timedelta(minutes=120) > meal[i+1]:
        #print(i)
        meal.drop(i, inplace=True)
meal.reset_index(drop=True, inplace=True)

P_df = pd.DataFrame(columns = ['C' + str(i+1) for i in range(30)])
#print(P_df)
Q_df = pd.DataFrame(columns = ['C' + str(i+1) for i in range(24)])
#print(Q_df)

for i,cgm_date, glucose in cgm_df.itertuples():
    #print(i,cgm_date, glucose)
    c=0
    t=0
    list=[]
    list2=[]
    for j in range(len(meal)-1):
        #print(insul_date)
        if meal[j] < cgm_date and cgm_date < meal[j] + timedelta(minutes=5):
            #print(meal[j], cgm_date)
            m=i-6
            for k in range(m, m+30):
                #print(m, cgm_df.iloc[m,1])
                list.append(cgm_df.iloc[m,1])
                m += 1
            #print(list)
            for l in list:
                t += 1
                if l > 0:
                    c += 1
            #print(c, t)
            if c >= 30*0.8 and t==30:
                #print('save P_df')
                P_df = P_df.append(pd.DataFrame([list], columns=P_df.columns), ignore_index=True)
            #----------Q matrix
            
            loop = True
            date = cgm_date
            z = i 
            while(loop):
                list2=[]
                date = date + timedelta(minutes=120)
                #print(meal[j], meal[j+1], date)
                if date < meal[j+1] :
                    z = z+24
                    for k in range(z, z+24):
                        #print(k, cgm_df.iloc[k,1])
                        list2.append(cgm_df.iloc[k,1])
                else:
                    loop = False
                #print(list2)
                c=0
                t=0
                for l2 in list2:
                    t += 1
                    if l2 > 0:
                        c += 1
                #print(c, t)
                if c >= 24*0.8 and t==24:
                    #print('save Q_df')
                    Q_df = Q_df.append(pd.DataFrame([list2], columns=Q_df.columns), ignore_index=True)
                    

import numpy as np
from scipy.fft import fft, fftfreq, rfft
data = np.empty(shape=[0, 8])
label = np.array([])

for p_arr in P_df.to_numpy():
    #print(p_arr)
    max_value = None
    min_value = None
    for idx, num in enumerate(p_arr):
        #print(num)
        if (max_value is None or num > max_value) :
            max_value = num
            max_idx = idx
        if (min_value is None or num < min_value) :
            min_value = num
            min_idx = idx
    diff_max1 = None
    diff_max2 = None
    for num in np.diff(p_arr):
        if (diff_max1 is None or num > diff_max1) :
            diff_max1 = num
    for num in np.diff(p_arr, n=2):
        if (diff_max2 is None or num > diff_max2) :
            diff_max2 = num
    #print(np.diff(p_arr))    
    #print(np.diff(p_arr, n=2))
  
    
    #print('max, idx:', max_value, max_idx)       
    #print('meal, idx:', meal_value, meal_idx) 
    #print('tau(minites):', abs(max_idx-meal_idx)*5) #f1
    #print('dG =', (max_value-meal_value)/meal_value) #f2
    #print('diff_max1:', diff_max1) #f3
    #print('diff_max2:', diff_max2) #f4
    
    #print('abs fft:', abs(c))
    x = p_arr
    #print(x)
    #------null value
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
        label = np.append(label, [1])
        #print(label)
    #----null test
    array_sum = np.sum(data)
    array_has_nan = np.isnan(array_sum)
    #print(array_has_nan)



for q_arr in Q_df.to_numpy():
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
        label = np.append(label, [0])
        #print(label)
    #----null test
    array_sum = np.sum(data)
    array_has_nan = np.isnan(array_sum)
    #print(array_has_nan)
        

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.2, random_state=121 )
dt_clf = DecisionTreeClassifier(random_state=111)
dt_clf.fit(X_train, Y_train)

#pickling
import pickle
#write
with open('model_pickle.pkl', 'wb') as f:
    pickle.dump(dt_clf, f)

