import pandas as pd
from datetime import datetime
from datetime import timedelta

#read 2 CSV files
cgm_raw = pd.read_csv(r'CGMData.csv', index_col = 0)
cgm_raw['datetime'] = pd.to_datetime(cgm_raw['Date']) + pd.to_timedelta(cgm_raw['Time'])
cgm_df = cgm_raw[['datetime','Sensor Glucose (mg/dL)']]
insulin_raw = pd.read_csv(r'InsulinData.csv')
insulin_raw['datetime'] = pd.to_datetime(insulin_raw['Date']) + pd.to_timedelta(insulin_raw['Time'])
insulin_df = insulin_raw[['datetime', 'Alarm']]

#sorting datatime
cgm_df = cgm_df.sort_values(by='datetime')
insulin_df = insulin_df.sort_values(by='datetime')

#filterig to satisfy daily 80% data intergrity of total daily data 
date_df=cgm_df[['Sensor Glucose (mg/dL)']].groupby(cgm_df['datetime'].dt.date).count().reset_index()
cond1 = (date_df['Sensor Glucose (mg/dL)'] > 288*0.8)
date_df= date_df.loc[cond1]
#print(date_df)
len_of_date = len(date_df)
#print(len_of_date)

#change datetime data type to string and change dataframe to list
cgm_df['datetime'] = cgm_df['datetime'].astype(str)
date_df['datetime'] = date_df['datetime'].astype(str)
date_list=date_df['datetime'].tolist() 

#find datetime on AUTO MODE ACTIVE PLGM OFF and change dataframe to list
mode_df = insulin_df[insulin_df['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF']
mode_df['datetime'] = mode_df['datetime'].astype(str)
mode_list = mode_df['datetime'].tolist()

#initialized variables
M_cgm180 = 0
A_cgm180 = 0
M_cgm250 = 0
A_cgm250 = 0
M_cgm70_180 = 0
A_cgm70_180 = 0
M_cgm70_150 = 0
A_cgm70_150 = 0
M_cgm70 = 0
A_cgm70 = 0
M_cgm50 = 0
A_cgm50 = 0
mode = 0  #0 = Manual, 1 = Auto 

#overnight (midnight to 6 am)
for date, glucose in cgm_df.itertuples(index=False):
    if mode ==0 and date[0:10] in date_list and date[11:] >= '00:00:00' and date[11:] < '06:00:00':
        #print(date,date[11:],mode)
        if glucose > 180:
            M_cgm180 = M_cgm180 + 1
        if glucose > 250:
            M_cgm250 = M_cgm250 + 1
        if glucose >= 70 and glucose <= 180:
            M_cgm70_180 = M_cgm70_180 + 1
        if glucose >= 70 and glucose <= 150:
            M_cgm70_150 = M_cgm70_150 + 1
        if glucose < 70:
            M_cgm70 = M_cgm70 + 1
        if glucose < 50:
            M_cgm50 = M_cgm50 + 1
    if mode ==1 and date[0:10] in date_list and date[11:] >= '00:00:00' and date[11:] < '06:00:00':
        #print(date,date[11:],mode)
        if glucose > 180:
            A_cgm180 = A_cgm180 + 1
        if glucose > 250:
            A_cgm250 = A_cgm250 + 1
        if glucose >= 70 and glucose <= 180:
            A_cgm70_180 = A_cgm70_180 + 1
        if glucose >= 70 and glucose <= 150:
            A_cgm70_150 = A_cgm70_150 + 1
        if glucose < 70:
            A_cgm70 = A_cgm70 + 1
        if glucose < 50:
            A_cgm50 = A_cgm50 + 1
    #print(date, glucose,mode)
    for d in mode_list:
        if date < d and d < str(datetime.fromisoformat(date) + timedelta(minutes=5)):
            if mode == 0: mode = mode + 1
            elif mode == 1: mode = mode - 1
            #print(date, d, mode)

#store the results to lists 
o_list1 = [M_cgm180/len_of_date/288*100, A_cgm180/len_of_date/288*100]
o_list2 = [M_cgm250/len_of_date/288*100, A_cgm250/len_of_date/288*100]
o_list3 = [M_cgm70_180/len_of_date/288*100, A_cgm70_180/len_of_date/288*100]
o_list4 = [M_cgm70_150/len_of_date/288*100, A_cgm70_150/len_of_date/288*100]
o_list5 = [M_cgm70/len_of_date/288*100, A_cgm70/len_of_date/288*100]
o_list6 = [M_cgm50/len_of_date/288*100, A_cgm50/len_of_date/288*100]

M_cgm180 = 0
A_cgm180 = 0
M_cgm250 = 0
A_cgm250 = 0
M_cgm70_180 = 0
A_cgm70_180 = 0
M_cgm70_150 = 0
A_cgm70_150 = 0
M_cgm70 = 0
A_cgm70 = 0
M_cgm50 = 0
A_cgm50 = 0
mode = 0  #0 = Manual, 1 = Auto 

#daytime (6 am to midnight)
for date, glucose in cgm_df.itertuples(index=False):
    if mode ==0 and date[0:10] in date_list and date[11:] >= '06:00:00' and date[11:] < '24:00:00': 
        #print(date,date[11:],mode)
        if glucose > 180:
            M_cgm180 = M_cgm180 + 1
        if glucose > 250:
            M_cgm250 = M_cgm250 + 1
        if glucose >= 70 and glucose <= 180:
            M_cgm70_180 = M_cgm70_180 + 1
        if glucose >= 70 and glucose <= 150:
            M_cgm70_150 = M_cgm70_150 + 1
        if glucose < 70:
            M_cgm70 = M_cgm70 + 1
        if glucose < 50:
            M_cgm50 = M_cgm50 + 1
    if mode ==1 and date[0:10] in date_list and date[11:] >= '06:00:00' and date[11:] < '24:00:00':
        #print(date,date[11:],mode)
        if glucose > 180:
            A_cgm180 = A_cgm180 + 1
        if glucose > 250:
            A_cgm250 = A_cgm250 + 1
        if glucose >= 70 and glucose <= 180:
            A_cgm70_180 = A_cgm70_180 + 1
        if glucose >= 70 and glucose <= 150:
            A_cgm70_150 = A_cgm70_150 + 1
        if glucose < 70:
            A_cgm70 = A_cgm70 + 1
        if glucose < 50:
            A_cgm50 = A_cgm50 + 1
    #print(date, glucose,mode)
    for d in mode_list:
        if date < d and d < str(datetime.fromisoformat(date) + timedelta(minutes=5)):
            if mode == 0: mode = mode + 1
            elif mode == 1: mode = mode - 1
            #print(date, d, mode)

#store the results to lists 
d_list1 = [M_cgm180/len_of_date/288*100, A_cgm180/len_of_date/288*100]
d_list2 = [M_cgm250/len_of_date/288*100, A_cgm250/len_of_date/288*100]
d_list3 = [M_cgm70_180/len_of_date/288*100, A_cgm70_180/len_of_date/288*100]
d_list4 = [M_cgm70_150/len_of_date/288*100, A_cgm70_150/len_of_date/288*100]
d_list5 = [M_cgm70/len_of_date/288*100, A_cgm70/len_of_date/288*100]
d_list6 = [M_cgm50/len_of_date/288*100, A_cgm50/len_of_date/288*100]

M_cgm180 = 0
A_cgm180 = 0
M_cgm250 = 0
A_cgm250 = 0
M_cgm70_180 = 0
A_cgm70_180 = 0
M_cgm70_150 = 0
A_cgm70_150 = 0
M_cgm70 = 0
A_cgm70 = 0
M_cgm50 = 0
A_cgm50 = 0
mode = 0  #0 = Manual, 1 = Auto 

#whole day (12 am to 12 am).
for date, glucose in cgm_df.itertuples(index=False):
    if mode ==0 and date[0:10] in date_list:
        if glucose > 180:
            M_cgm180 = M_cgm180 + 1
        if glucose > 250:
            M_cgm250 = M_cgm250 + 1
        if glucose >= 70 and glucose <= 180:
            M_cgm70_180 = M_cgm70_180 + 1
        if glucose >= 70 and glucose <= 150:
            M_cgm70_150 = M_cgm70_150 + 1
        if glucose < 70:
            M_cgm70 = M_cgm70 + 1
        if glucose < 50:
            M_cgm50 = M_cgm50 + 1
    if mode ==1 and date[0:10] in date_list:
        if glucose > 180:
            A_cgm180 = A_cgm180 + 1
        if glucose > 250:
            A_cgm250 = A_cgm250 + 1
        if glucose >= 70 and glucose <= 180:
            A_cgm70_180 = A_cgm70_180 + 1
        if glucose >= 70 and glucose <= 150:
            A_cgm70_150 = A_cgm70_150 + 1
        if glucose < 70:
            A_cgm70 = A_cgm70 + 1
        if glucose < 50:
            A_cgm50 = A_cgm50 + 1
    #print(date, glucose,mode)
    for d in mode_list:
        if date < d and d < str(datetime.fromisoformat(date) + timedelta(minutes=5)):
            if mode == 0: mode = mode + 1
            elif mode == 1: mode = mode - 1
            #print(date, d, mode)

a_list1 = [M_cgm180/len_of_date/288*100, A_cgm180/len_of_date/288*100]
a_list2 = [M_cgm250/len_of_date/288*100, A_cgm250/len_of_date/288*100]
a_list3 = [M_cgm70_180/len_of_date/288*100, A_cgm70_180/len_of_date/288*100]
a_list4 = [M_cgm70_150/len_of_date/288*100, A_cgm70_150/len_of_date/288*100]
a_list5 = [M_cgm70/len_of_date/288*100, A_cgm70/len_of_date/288*100]
a_list6 = [M_cgm50/len_of_date/288*100, A_cgm50/len_of_date/288*100]

#combine with the results in lists to make them dataframe
results_df = pd.DataFrame(list(zip(o_list1, o_list2, o_list3, o_list4, o_list5, o_list6, d_list1, d_list2, d_list3, d_list4, d_list5, d_list6, a_list1, a_list2, a_list3, a_list4, a_list5, a_list6)))
#export to csv file
results_df.to_csv(r'Results.csv', index = False, header=False)
