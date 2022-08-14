import pandas as pd
import numpy as np
import sklearn as sk
import math

cgm_raw = pd.read_csv(r'CGMData.csv', index_col = 0, low_memory=False)
cgm_raw['datetime'] = pd.to_datetime(cgm_raw['Date']) + pd.to_timedelta(cgm_raw['Time'])
cgm_df = cgm_raw[['datetime','Sensor Glucose (mg/dL)']]
cgm_df = cgm_df.sort_values(by='datetime')
cgm_df.reset_index(drop=True, inplace=True)
insulin_raw = pd.read_csv(r'InsulinData.csv', index_col = 0, low_memory=False)
insulin_raw['datetime'] = pd.to_datetime(insulin_raw['Date']) + pd.to_timedelta(insulin_raw['Time'])
insulin_df = insulin_raw[['datetime', 'BWZ Carb Input (grams)']]
insulin_df = insulin_df.sort_values(by='datetime')
insulin_df.reset_index(drop=True, inplace=True)

meal_df = insulin_df[insulin_df['BWZ Carb Input (grams)'] > 0]
df = pd.merge(cgm_df,meal_df,on='datetime',how='outer').sort_values(by='datetime')
df.reset_index(drop=True, inplace=True)

P_df = pd.DataFrame(columns = ['C' + str(i+1) for i in range(24)])
#print(P_df)
P_label = np.array([])
for idx, date, glucose, insulin in df.itertuples():
    #print(idx, date, glucose, insulin)
    list = []
    c=0
    t=0
    if insulin > 0:
        n_meal = 0
        for i in range(24):
            #print(df.iloc[idx+i+1,2])
            if df.iloc[idx+i+1,2] > 0:
                n_meal += 1
        
        if n_meal == 0 and (idx-8) > 0:
            #print('no meals in 2 hours')
            #print(idx, date, glucose, insulin)          
            
            #for k in range(6):
                #print(idx,idx+k-6,k)
                #list.append(df.ix[idx+k-6,1])
            for k in range(24):
                #print(idx,idx+k+1,k)
                list.append(df.iloc[idx+k+1,1])                
        #print(list)
        for l in list:
            t += 1
            if l > 0:
                c += 1
            #print(c, t)
        if c >= 24*0.8 and t==24:
            #print('save P_df')
            P_df = P_df.append(pd.DataFrame([list], columns=P_df.columns), ignore_index=True)
        
            if insulin >= 3 and insulin < 23 :
                P_label = np.append(P_label, [0])
            elif insulin >= 23 and insulin < 43 :
                P_label = np.append(P_label, [1])
            elif insulin >= 43 and insulin < 63 :
                P_label = np.append(P_label, [2])
            elif insulin >= 63 and insulin < 83 :
                P_label = np.append(P_label, [3])
            elif insulin >= 83 and insulin < 103 :
                P_label = np.append(P_label, [4])
            elif insulin >= 103 and insulin < 130 :
                P_label = np.append(P_label, [5])
        

#print('P_label', P_label)


data = np.empty(shape=[0, 4])
label = np.array([])
i=0
for p_arr in P_df.to_numpy():
    #print(p_arr)
    max_value = None
    meal_value = p_arr[0]
    meal_idx = 0
    for idx, num in enumerate(p_arr):
        #print(idx,num)
        if (max_value is None or num > max_value) :
            max_value = num
            max_idx = idx
    #print(meal_idx, meal_value, max_idx, max_value)
       
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
  
    

    #print('tau(minites):', abs(max_idx-meal_idx)*5) #f1
    #print('dG =', (max_value-meal_value)/meal_value) #f2
    #print('diff_max1:', diff_max1) #f3
    #print('diff_max2:', diff_max2) #f4
    
    
    list_feature = [abs(max_idx-meal_idx)*5, (max_value-meal_value)/meal_value, diff_max1, diff_max2]
    #print(list_feature)
    
    array_sum = np.sum(np.array(list_feature))
    
    if np.isnan(array_sum) == False:
        data = np.append(data, [np.array(list_feature)], axis = 0)
        #print(data)
        label = np.append(label, P_label[i])
        #print(label)
    i = i + 1
    #----null test
    array_sum = np.sum(data)
    array_has_nan = np.isnan(array_sum)
    #print(array_has_nan)


data_df = pd.DataFrame(data=data)



from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, init='k-means++', random_state=1)
kmeans.fit(data_df)
#kmeans.labels_
#kmeans.cluster_centers_
k_result_df = data_df.copy()
k_result_df['target'] = label
k_result_df['cluser'] = kmeans.labels_
cluster_result = k_result_df.groupby (['cluser','target'])[0].count()
#r = result_df.groupby (['cluser'])['target'].value_counts()
#print(cluster_result)


zero_data = np.zeros(shape=(6,7))
kmeans_df = pd.DataFrame(zero_data)
total_sum_k = 0
for i in range(6):
    total = 0 
    r=k_result_df[k_result_df['cluser'] == i][['target','cluser']].groupby('target').count()
    r_arr = r.to_numpy()
    for idx, count in enumerate(r_arr):
        kmeans_df.iloc[i, idx] = count
        total += float(count)
    kmeans_df.iloc[i, 6] = total
    total_sum_k += total
kmeans_df


from sklearn.cluster import DBSCAN
#dbscan = DBSCAN(eps=5.12, min_samples=7, metric='euclidean')
dbscan = DBSCAN(eps=5.12, min_samples=7, metric='euclidean')
dbscan = dbscan.fit(data_df)
dbscan.labels_

d_result_df = data_df.copy()
d_result_df['target'] = label
d_result_df['cluster'] = dbscan.labels_
d_result = d_result_df.groupby (['cluster','target'])[0].count()

zero_data = np.zeros(shape=(6,7))
dbscan_df = pd.DataFrame(zero_data)
total_sum_d = 0
for i in range(6):
    total = 0 
    r=d_result_df[d_result_df['cluster'] == i][['target','cluster']].groupby('target').count()
    r_arr = r.to_numpy()
    for idx, count in enumerate(r_arr):
        dbscan_df.iloc[i, idx] = count
        total += float(count)
    dbscan_df.iloc[i, 6] = total
    total_sum_d += total 
dbscan_df

#---------SSE for Kmeans-----------------------
final_result=[]
final_result.append(kmeans.inertia_)
#print('SSE for Kmeans', kmeans.inertia_)
#---------SSE for DBSCAN---------------------
data_arr = data_df.to_numpy()
arr = []
centroid_arr = []
for i in range(6):
    c = 0
    for j in range(len(data_arr)) :
        if d_result_df.iloc[j,5] == i : 
            arr.append(data_arr[j])
            c += 1        
    #print(arr)
    centroid = np.sum(arr, axis=0)/c
    centroid_arr.append(centroid)
    #print(centroid)
#print(centroid_arr)
dbscan_sse=0
for i in range(6):

    for j in range(len(data_arr)) :
        if d_result_df.iloc[j,5] == i : 
            #print(i, data_arr[j], centroid_arr[i])
            dbscan_sse += (np.linalg.norm(data_arr[j]-centroid_arr[i]))**2
final_result.append(dbscan_sse)
#print('SSE for DBSCAN', dbscan_sse)

#---------Entropy for Kmeans ------------------------
E_total = 0
for i in range(6):
    E = 0
    for j in range(6):
        
        frac = kmeans_df.iloc[i, j] / kmeans_df.iloc[i, 6]
        #print(frac, kmeans_df.iloc[i, j], kmeans_df.iloc[i, 6])
        if frac != 0:
            E += -frac*math.log2(frac)
            #print(E)
        
    E_total += E * kmeans_df.iloc[i, 6] / total_sum_k
    #print('E_total', E_total)
final_result.append(E_total)    
#print('Entropy for Kmeans :', E_total)

#---------Entropy for DBSCAN ------------------------
E_total = 0
for i in range(6):
    E = 0
    for j in range(6):
        
        frac = dbscan_df.iloc[i, j] / dbscan_df.iloc[i, 6]
        #print(frac, kmeans_df.iloc[i, j], kmeans_df.iloc[i, 6])
        if frac != 0:
            E += -frac*math.log2(frac)
            #print(E)
        
    E_total += E * dbscan_df.iloc[i, 6] / total_sum_d
    #print('E_total', E_total)
final_result.append(E_total)    
#print('Entropy for DBSCAN :', E_total)

#---------Purity for Kmeans ------------------------
P_max_total = 0
for i in range(6):
    P_max = 0
    for j in range(6):
        if P_max < kmeans_df.iloc[i, j]:
            #print(P_max, kmeans_df.iloc[i, j])
            P_max = kmeans_df.iloc[i, j]            
    P_max_total += P_max + P_max
purity= P_max_total / total_sum_k
final_result.append(purity)  
#print('Purity for Kmeans :', purity)

#---------Purity for DBSCAN ------------------------
P_max_total = 0
for i in range(6):
    P_max = 0
    for j in range(6):
        if P_max < dbscan_df.iloc[i, j]:
            #print(P_max, kmeans_df.iloc[i, j])
            P_max = dbscan_df.iloc[i, j]            
    P_max_total += P_max + P_max
purity= P_max_total / total_sum_d
final_result.append(purity.tolist())
#print('Purity for DBSCAN :', purity)

final_result

final_result_df = pd.DataFrame(data=[final_result])
final_result_df

final_result_df.to_csv(r'Result.csv', index = False, header=False)

