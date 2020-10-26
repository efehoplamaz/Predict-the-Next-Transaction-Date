# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:12:49 2020

@author: S64615
"""

import pandas as pd
from statistics import mean
from statistics import stdev
import datetime
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.svm import SVR

#############################################################################

"""
    Helper functions.
    1) Sorting function will sort the dates from the furthest to the closest to date.
    2) Distance will calulate the magnitude of the distance between two vectors.
"""

def sorting(L):
    splitup = L.split('-')
    return splitup[2], splitup[1], splitup[0]

def distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))  

#############################################################################

#Importing the transaction dataframe and Finansbank's "predicted pattern" dataframe.
# Finansbank's dataframe consists data of each unique customer assigned to an expected date
# of which a transfaction can happen according to Finansbank.

df_transaction = pd.read_csv('path/to/data')
df_finans_pattern = pd.read_csv('path/to/data')

# Creating a dataframe from the dataframes above. This dataframe will include attributes of
# (1)transaction type (e.g. money transfer, rent payment etc.),
# (2)customer unique key (each customer will have a unique encoded key)
# (3)a list of day differences between payment dates
# (4)mean of thre day differences for that customer (i.e for that unique key)
# (5)standard devaiton of the day differences for that customer
# (6)multiple flag, this indicates whether that customer has made multiple payments in a month 
# for previous 3 months.
# (7) Finansbanks expected date
# (8) The actual payment date
# (9) The last payment date


all_data = []

for tr_type, tr_group in df_transaction.groupby('MONEY_TRANSFER_TP'):
        
    for unq_key, unq_key_group in tr_group.groupby('UNQ_KEY_MAPPED'):
        
        unq_row_data = []
        
        trx_dts = [trx_dt.split(' ')[0] for trx_dt in unq_key_group.TRX_DT.tolist()]
        trx_dts.reverse()
        
        trx_to_predict = trx_dts[-1]

        
        datetime_objs = [datetime.date(int(date.split('-')[2]),
                                       int(date.split('-')[1]),
                                       int(date.split('-')[0])) for date in trx_dts]
    
        day_diffs = [(datetime_objs[i+1] - datetime_objs[i]).days for i in range(len(datetime_objs)-1)]
        
        if len(day_diffs) >= 2:
            mean_diff = mean(day_diffs)
            std_diff = stdev(day_diffs)
            
            trx_one_before_last = trx_dts[-2]
            
            df_multi_key = df_finans_pattern[(df_finans_pattern['MONEY_TRANSFER_TP'] == tr_type) &
                                              (df_finans_pattern['UNQ_KEY_MAPPED']  == unq_key)]
            
            multi_dates = [elm.split(' ')[0] for elm in df_multi_key.AS_OF_DT.tolist()]
            multi_dates = sorted(multi_dates, key=sorting)
            
            if multi_dates:
                multi_dt = multi_dates[-1] + ' 00:00:00'
                
                multi_f = df_multi_key[df_multi_key.AS_OF_DT == multi_dt].MULTI_F.values[0]
                
                exp_date_finans = df_multi_key[df_multi_key.AS_OF_DT == multi_dt].EXP_DT.values[0].split(' ')[0]
    
                unq_row_data.extend((tr_type, unq_key, day_diffs, mean_diff, std_diff, multi_f, exp_date_finans, trx_to_predict, trx_one_before_last))
                
                all_data.append(unq_row_data)
        

df_modified = pd.DataFrame(all_data, columns = ['TR_TYPE', 'UNQ_KEY', 'DAY_DIFFS', 'MEAN', 'ST_DEV', 'MULTI_F', 'FINANS_EXP_DT', 'TRX_TO_PRED', 'TRX_BEFORE_LAST'])

################################################################

# Create a copy of the new dataframe above to process it for the K-means clustering since 
# I would like to group the customers by their day difference mean, standard devation and multiple flag.
# Also no categorical data is accepted for K-means, that's why created a copy and dropped the columns.

df_modified_copy = df_modified.copy()
df_modified_copy.drop(['TR_TYPE', 'UNQ_KEY','DAY_DIFFS', 'FINANS_EXP_DT', 'TRX_TO_PRED', 'TRX_BEFORE_LAST'], axis = 1, inplace= True)

###############################################################

# K-Means clustering on the customer data, grouping the customers into clusters. By that way
# I will find the customer groups and different regression techniques can be applied to different
# customer groups to create more successful predicitons.
# Also with clustering customers, I can find the customer clusters which Finansbank will predict
# poorly so I can compare if I have done a better predictions.

sse = []

for k in range(1, 25):

    kmeans = KMeans(n_clusters = k).fit(df_modified_copy)

    centroids = kmeans.cluster_centers_

    pred_clusters = kmeans.predict(df_modified_copy)

    curr_sse = 0

    for i in range(len(df_modified_copy)):

      curr_center = centroids[pred_clusters[i]]

      curr_sse += distance(df_modified_copy.iloc[i,:], curr_center)
  
    sse.append(curr_sse)

print(sse)

############################################################

# If we plot the graph we can find the optimal k value by using the elbow method.

plt.plot(list(range(1, len(sse)+1)), sse)
plt.show()

############################################################

# After finding the optimal k, I investigted the characteristics of each clusters.

optimal_k = ...
kmeans = KMeans(n_clusters = optimal_k).fit(df_modified_copy)
pred_clusters = kmeans.predict(df_modified_copy)


df_modified['CLUSTER_NO'] = pred_clusters

for cluster_no, cluster_group in df_modified.groupby('CLUSTER_NO'):
    
    print('This is cluster number ' + str(cluster_no))
    
    print('Mean of the means are ' + str(mean(cluster_group.MEAN.tolist())))
    print('Min of the mean is ' + str(min(cluster_group.MEAN.tolist()))) 
    print('Max of the mean is ' + str(max(cluster_group.MEAN.tolist())))
    
    print('Mean of the stdevs are ' + str(mean(cluster_group.ST_DEV.tolist())))
    print('Min of the stdevs is ' + str(min(cluster_group.ST_DEV.tolist()))) 
    print('Max of the stdevs is ' + str(max(cluster_group.ST_DEV.tolist())))
    
    print('There are ' + str(sum(cluster_group.MULTI_F.tolist())) + ' many multi flagged customers')
    
    
    cluster_size = 0
    efe_correct = 0
    finans_correct = 0
    efe_finans_yakin = 0
    
    # After investigating the characteristics of the clusters, for each customer/unique key
    # in the cluster I have created a personalized Support Vector Regressor. Then I predicted
    # the next day difference. I have calculated the new predicted date by adding the predicted
    # day difference on the last transaction date. If the predicted date is on a weekend,
    # (1) if its on a Saturday then predict it as it is on Friday since it will be better to
    # remind the customers as early and close to the actual date as possible, (2) if its on a
    # Sunday then predict it as Monday.     
    
    for index, row in cluster_group.iterrows():
        
        day_diffs = row.DAY_DIFFS
        
        if len(day_diffs) >= 6:
            
            X_train = []
            y_train = []
            last_i = 0
            
            for i in range(3,len(day_diffs)-2):
                X_train.append([i-2, day_diffs[i-3], day_diffs[i-2], day_diffs[i-1], mean(day_diffs[:i]), stdev(day_diffs[:i])])
                y_train.append([day_diffs[i]])
                last_i = i-2
            
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            X_test = np.array([[last_i, day_diffs[-4], day_diffs[-3], day_diffs[-2], mean(day_diffs[:-1]), stdev(day_diffs[:-1])]])
            y_test = day_diffs[-1]
    
            
            SVR_regression_history = SVR(gamma = 'auto', kernel='rbf')
            SVR_regression_history.fit(X_train, y_train.ravel())
            predicted_daydiff = round(SVR_regression_history.predict(X_test)[0])            
            
            
        elif (len(day_diffs) < 6) and (len(day_diffs) >= 3):
            
            X_train = np.array([list(range(len(day_diffs)-1))]).T
            y_train = np.array(day_diffs[:-1])
                                
            X_test =  np.array([[len(day_diffs)]])
            y_test = day_diffs[-1]
                
            SVR_regression = SVR(gamma = 'auto', kernel='rbf')
            SVR_regression.fit(X_train,y_train)
            predicted_daydiff = round(SVR_regression.predict(X_test)[0])
            
        else:
            predicted_daydiff = mean(day_diffs)
        
        
        datetime_last = datetime.date(int(row.TRX_BEFORE_LAST.split('-')[2]),
                                      int(row.TRX_BEFORE_LAST.split('-')[1]),
                                      int(row.TRX_BEFORE_LAST.split('-')[0]))
        
        predicted_date = datetime_last + datetime.timedelta(days = predicted_daydiff)
        
        day_indicator = predicted_date.weekday()
                    
        if day_indicator == 5:
            predicted_daydiff -= 1
            
        if day_indicator == 6:                  
            predicted_daydiff += 1
                
        predicted_date = datetime_last + datetime.timedelta(days = predicted_daydiff)
        
        
        y_true = datetime.date(int(row.TRX_TO_PRED.split('-')[2]), 
                               int(row.TRX_TO_PRED.split('-')[1]), 
                               int(row.TRX_TO_PRED.split('-')[0]))
        
        finans_date = datetime.date(int(row.FINANS_EXP_DT.split('-')[2]), 
                               int(row.FINANS_EXP_DT.split('-')[1]), 
                               int(row.FINANS_EXP_DT.split('-')[0]))
        
        efe_actual_diff = (y_true - predicted_date).days
        finans_actual_diff = (y_true - finans_date).days
            
        if (efe_actual_diff <= 2) and (efe_actual_diff >= 0):
            efe_correct += 1
        if (finans_actual_diff <= 2) and (finans_actual_diff >= 0):
            finans_correct += 1
        if abs(finans_actual_diff - efe_actual_diff) <= 1:
            efe_finans_yakin += 1
        cluster_size += 1
    
    
    print('Finans predicted {} many times correctly, Efe predicted {} many times correctly. Total predictions are {}. Efe finans yakin {}'.format(efe_correct, finans_correct, cluster_size, efe_finans_yakin))
    print('\n')
