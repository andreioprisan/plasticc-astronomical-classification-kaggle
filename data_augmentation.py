
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import warnings
import itertools
from random import randint


# In[48]:


# read csv files
train = pd.read_csv('/modules/cs342/Assignment2/training_set.csv',header=0)
meta_train = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv',header=0)


# ### Helper Functions

# In[3]:


# Helper function used to plot a single object based on object id
def plotSingleObject(obj):
    # Define all unique passbands
    unique_passbands = [0,1,2,3,4,5]

    # plot flux for each passband
    for passband in unique_passbands:
        specific_passband = obj[obj['passband'] == passband]
        plt.scatter(specific_passband['mjd'], specific_passband['flux'], label=passband, alpha=0.9, s=10)

    # Show plot
    plt.title("Object ID: " + str(obj['object_id'].head(1).values))
    plt.xlabel("MJD from Nov 17, 1858")
    plt.ylabel("Flux")
    plt.show()


# ### Implement window warping and window slicing

# In[50]:


# Get all objects
unique_objects = meta_train['object_id'].unique()

# Taken from: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
def scaleMinMax(df, new_min, new_max):
    df_std = (df - df.min()) / (df.max() - df.min())
    df_scaled = df_std * (new_max - new_min) + new_min
    return df_scaled

# Window warping for a single object
def warpObject(df, init):
    obj = df.copy()

    # Choose a 20% chunk
    count = obj['mjd'].count()
    low_index = randint(0, int(0.8*count))
    high_index = low_index + int(0.2*count)

    # Get the CHUNK's mjd max and min
    chunk_max = obj['mjd'][low_index:high_index].max()
    chunk_min = obj['mjd'][low_index:high_index].min()
    chunk_range = chunk_max - chunk_min

    # SQUEEZE CHUNK
    obj.loc[obj.index[low_index:high_index], 'mjd'] = scaleMinMax(obj.loc[obj.index[low_index:high_index], 'mjd'],
                                                                  new_min= chunk_min+0.25*chunk_range,
                                                                  new_max= chunk_max-0.25*chunk_range)
    # Append
    if(init==True):
        obj.to_csv('warped_train.csv',  header=True, index=False)
    else:
        obj.to_csv('warped_train.csv',  header=False, mode='a', index=False)

# Window slicing for a single object
def sliceObject(df, init):
    obj = df.copy()

    # Choose a 20% chunk
    count = obj['mjd'].count()
    low_index = randint(0, int(0.8*count))
    high_index = low_index + int(0.2*count)

    # Drop the randomly chosen 20%
    obj = obj.drop(obj.index[low_index:high_index]).reset_index()

    # Append
    if(init==True):
        obj.to_csv('sliced_train.csv',  header=True, index=False)
    else:
        obj.to_csv('sliced_train.csv',  header=False, mode='a', index=False)


# ### Run window warping and window slicing

# In[51]:


# Iterate window warping and slicing for all objects
for index, obj_id in enumerate(unique_objects):
    if(index==0):
        warpObject(train[train['object_id'] == obj_id], init=True)
    else:
        warpObject(train[train['object_id'] == obj_id], init=False)
print "Done warping time"

for index, obj_id in enumerate(unique_objects):
    if(index==0):
        sliceObject(train[train['object_id'] == obj_id], init=True)
    else:
        sliceObject(train[train['object_id'] == obj_id], init=False)
print "Done slicing time"


# In[4]:


# Check values
sliced_train = pd.read_csv('./sliced_train.csv')
warped_train = pd.read_csv('./warped_train.csv')
print sliced_train.shape
print warped_train.shape
