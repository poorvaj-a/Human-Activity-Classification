

import pandas as pd
import numpy as np
import time
start = time.time()

Data = pd.read_csv("PAMAP2_Dataset/Protocol/subject101.dat",delimiter=" ",header=None)
Data = Data.to_numpy()
for x in range(2,10):
    path = "PAMAP2_Dataset/Protocol/subject10" + str(x) + ".dat"
    temp = pd.read_csv(path,delimiter=" ",header=None).to_numpy()
    Data = np.append(Data,temp,axis=0)
#######################################
# Use Data_NO and Data_NO_SENSOR only # 
#######################################
prev=0
#linear interpolation of heart rate sensor values. Roundoff to nearest integer.
remove = []
for i in range(0,Data[:,2].shape[0]):
    
    for j in range(3,Data.shape[1]):
        if np.isnan(Data[i][j]) or Data[i][1] == 0:
            remove.append(i)
            break
    if not np.isnan(Data[:,2][i]):
        prev = i
        next = i+1
        for j in range(prev+1,Data[:,2].shape[0]):
            if not np.isnan(Data[:,2][j]):
                next = j
                break
            if j == Data[:,2].shape[0]-1:
                next = j
                Data[:,2][next]=Data[:,2][prev]
        for j in range(prev+1,next):
            Data[:,2][j] = np.rint(Data[:,2][prev]+((Data[:,2][next]-Data[:,2][prev])/(next-prev))*(j-prev))
Data=np.delete(Data,remove,axis=0)
print(Data.shape)

    
# Use Data for normal all channel
#In readme it is said that orientation values are invalid for this data. So we also check after removing them
# print(Data[:,-[1,2]].shape)
orientation = [19,18,17,16,36,35,34,33,53,52,51,50]
Data_NO=Data[:,np.delete(np.arange(Data.shape[1]),orientation)]
#Data_NO is without the orientation information

#each IMU different
hand = np.arange(20)
Data_hand = Data[:,hand]

chest = np.append(np.arange(3),np.arange(20,37))
Data_chest = Data[:,chest]

ankle = np.append(np.arange(3),np.arange(37,54))
Data_ankle = Data[:,ankle]

#each no orintations IMU data. USe this 
###USE THESE along with DATA_NO
Data_NO_hand=Data_hand[:,0:16]
Data_NO_chest=Data_chest[:,0:16]
Data_NO_ankle=Data_ankle[:,0:16]
###USE THESE

## Time stamp may make sense for a single entity but for generalised it may be better to take it out
Data_NO_NT = Data_NO[:,1:]
# for x in range(0,Data_NO_NT.shape[0]):
#     for j in range(0,Data_NO_NT.shape[1]):
#         if np.isnan(Data[x][j]):
#             print(x,j)
Data_NO_NT_hand=Data_hand[:,1:16]
Data_NO_NT_chest=Data_chest[:,1:16]
Data_NO__NT_ankle=Data_ankle[:,1:16]
print(Data_NO_NT.shape)

print(time.time() - start)



        
        
