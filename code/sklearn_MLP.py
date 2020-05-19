
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import warnings
import itertools
from random import randint
import multiprocessing

import tensorflow as tf
import keras
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

from tsfresh.feature_extraction import extract_features
from astropy.cosmology import FlatLambdaCDM


# In[2]:


# Set cores to improve speed
cores=multiprocessing.cpu_count()


# ### Extracting simple aggregates of time series

# In[3]:


# Enable garbage collection
gc.enable()

# Read files
meta_train = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv')
train = pd.read_csv('/modules/cs342/Assignment2/training_set.csv')


# Add flux ratio, flux by flux ratio and magnitudes
train['flux_ratio_sq'] = np.power(train['flux'] / train['flux_err'], 2.0)
train['flux_by_flux_ratio_sq'] = train['flux'] * train['flux_ratio_sq']
train['flux_magnitude'] = -2.5*np.log(train['flux'])


# Perform simple aggregate functions on time series
aggs = {
    'flux_magnitude': ['min', 'std', 'skew'],
    'mjd': ['size'],
    'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
    'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
    'detected': ['mean'],
    'flux_ratio_sq':['sum','skew'],
    'flux_by_flux_ratio_sq':['sum','skew'],
}

agg_train = train.groupby('object_id').agg(aggs)
new_columns = [
    k + '_' + agg for k in aggs.keys() for agg in aggs[k]
]

# Add additional simple features
agg_train.columns = new_columns
agg_train['flux_diff'] = agg_train['flux_max'] - agg_train['flux_min']
agg_train['flux_dif2'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_mean']
agg_train['flux_w_mean'] = agg_train['flux_by_flux_ratio_sq_sum'] / agg_train['flux_ratio_sq_sum']
agg_train['flux_dif3'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_w_mean']

# Garbage collection
gc.collect()


# ### Method I & III: Fourier Transform and 'mjd' detected differences

# In[4]:


# Taken from https://www.kaggle.com/iprapas/ideas-from-kernels-and-discussion-lb-1-135?fbclid=IwAR29tt_a4NtF8_eYeFTGyKnsJlLWHgRcDDiRLXmaIl9sy3Tjw_VeeoUNj_0
def featurize(df):
    # METHOD I
    # Create fourier transform coefficients here.
    # Fft coefficient is meant to capture periodicity
    # Features to compute with tsfresh library.
    fcp = {'fft_coefficient': [{'coeff': 0, 'attr': 'abs'},{'coeff': 1, 'attr': 'abs'}],'kurtosis' : None, 'skewness' : None}
    agg_df_ts = extract_features(df, column_id='object_id', column_sort='mjd', column_kind='passband', column_value = 'flux', default_fc_parameters = fcp, n_jobs=cores)

    # METHOD III
    # Find bursts decay rate based on detected == 1
    # Get mjd_diff_det which is the difference of mjd where detected == 1
    # Taken from https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696#410538
    df_det = df[df['detected']==1].copy()
    agg_df_mjd = extract_features(df_det, column_id='object_id', column_value = 'mjd', default_fc_parameters = {'maximum':None, 'minimum':None}, n_jobs=4)
    agg_df_mjd['mjd_diff_det'] = agg_df_mjd['mjd__maximum'] - agg_df_mjd['mjd__minimum']
    del agg_df_mjd['mjd__maximum'], agg_df_mjd['mjd__minimum']
    agg_df_ts = pd.merge(agg_df_ts, agg_df_mjd, on = 'id')

    # tsfresh returns a dataframe with an index name='id'
    agg_df_ts.index.rename('object_id',inplace=True)

    return agg_df_ts

# Merge them together
agg_train_ts = featurize(train)
agg_train = pd.merge(agg_train, agg_train_ts, on='object_id')


# ### Merging extracted features with meta data

# In[5]:


# Merge agg_train and meta data
full_train = agg_train.reset_index().merge(
    right=meta_train,
    how='outer',
    on='object_id'
)

# Remove y label
if 'target' in full_train:
    y = full_train['target']
    del full_train['target']
classes = sorted(y.unique())

# Taken from Giba's topic : https://www.kaggle.com/titericz
# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
# with Kyle Boone's post https://www.kaggle.com/kyleboone
class_weight = {
    c: 1 for c in classes
}
for c in [64, 15]:
    class_weight[c] = 2

print('Unique classes : ', classes)


# In[6]:


# METHOD II
# Find the peak absolute magnitude of the object to better identify burst objects.
# Find new distmod based on redshift for more accurate values.
def addAbsoluteMagnitudes(df):

    # Add new distmod based on FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    df['distmod_flatlambdacdm'] = cosmo.distmod(df['hostgal_photoz'])
    df['distmod_flatlambdacdm'] = df['distmod_flatlambdacdm'].replace(-np.inf, np.nan)

    # Add absolute magnitudes for peak values
    df['flux_abs_magnitude_min'] = df['flux_magnitude_min'] - df['distmod_flatlambdacdm'].values
    df['flux_abs_magnitude_std'] = df['flux_magnitude_std'] - df['distmod_flatlambdacdm'].values
    df['flux_abs_magnitude_skew'] = df['flux_magnitude_skew'] - df['distmod_flatlambdacdm'].values
    df['flux_abs_magnitude_min'] = df['flux_abs_magnitude_min'].replace(np.nan, 100)
    df['flux_abs_magnitude_std'] = df['flux_abs_magnitude_std'].replace(np.nan, 100)
    df['flux_abs_magnitude_skew'] = df['flux_abs_magnitude_skew'].replace(np.nan, 100)

    return df

# Apply function on full_train
full_train = addAbsoluteMagnitudes(full_train)


# In[7]:


# Delete unwanted features
if 'object_id' in full_train:
    oof_df = full_train[['object_id']]
    del full_train['object_id'], full_train['distmod'], full_train['hostgal_specz']
    del full_train['ra'], full_train['decl'], full_train['gal_l'],full_train['gal_b'],full_train['ddf']

# Fill NaN values with mean
train_mean = full_train.mean(axis=0)
full_train.fillna(train_mean, inplace=True)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


# ### Standard Scaler

# In[8]:


# Scale data using Standard Scaler
full_train_new = full_train.copy()
ss = StandardScaler()
full_train_ss = ss.fit_transform(full_train_new)


# ### Define Loss Function

# In[9]:

# Calculates multi weighted logloss used for scoring
def multi_weighted_logloss(y_ohe, y_p):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# ### Tune models

# In[10]:


# Convert y to one hot encoding labels
unique_y = np.unique(y)
class_map = dict()
for i,val in enumerate(unique_y):
    class_map[val] = i

y_map = np.zeros((y.shape[0],))
y_map = np.array([class_map[val] for val in y])
y_categorical = to_categorical(y_map)


# In[15]:


oof_preds = np.zeros((len(full_train_ss), len(classes)))

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
importances = pd.DataFrame()

for fold_, (trn_, val_) in enumerate(folds.split(y_map, y_map)):
    # Get training and validation data
    x_train, y_train = full_train_ss[trn_], y_categorical[trn_]
    x_valid, y_valid = full_train_ss[val_], y_categorical[val_]

    # Multi Layer Perceptron
    clf = MLPClassifier(hidden_layer_sizes=(20,20,20),
                        activation='relu', # default is relu
                        solver='adam', # default is adam
                        alpha=0.01, # default is 0.0001
                        max_iter=200, # default is 200
                        random_state=1)
    clf.fit(x_train, y_train)

    oof_preds[val_, :] = clf.predict_proba(x_valid)
    print(multi_weighted_logloss(y_valid, oof_preds[val_, :]))

print('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(y_categorical,oof_preds))


# In[16]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_map, np.argmax(oof_preds,axis=-1))
np.set_printoptions(precision=2)

# Get columns from sample submission
sample_sub = pd.read_csv('/modules/cs342/Assignment2/sample_submission.csv')
class_names = list(sample_sub.columns[1:-1])
del sample_sub;gc.collect()

# Plot non-normalized confusion matrix
plt.figure(figsize=(10,10))
foo = plot_confusion_matrix(cnf_matrix, classes=class_names,normalize=True,
                      title='Confusion matrix')


# ### Fit model for final predictions

# In[17]:


# Multilayer Perceptron
mlp = MLPClassifier(hidden_layer_sizes=(20,20,20),
                    activation='relu',
                    solver='adam',
                    alpha=0.01,
                    max_iter=200,
                    random_state=1)
mlp.fit(full_train_ss, y_categorical)


# ### Predict on test set

# In[18]:


"""
meta_test = pd.read_csv('/modules/cs342/Assignment2/test_set_metadata.csv')

import time

start = time.time()
chunks = 5000000
for i_c, df in enumerate(pd.read_csv('/modules/cs342/Assignment2/test_set.csv', chunksize=chunks, iterator=True)):
    # Featurize Fourier Transform
    agg_test_ts = featurize(df)

    df['flux_ratio_sq'] = np.power(df['flux'] / df['flux_err'], 2.0)
    df['flux_by_flux_ratio_sq'] = df['flux'] * df['flux_ratio_sq']
    df['flux_magnitude'] = -2.5*np.log(df['flux'])

    # Group by object id
    agg_test = df.groupby('object_id').agg(aggs)
    agg_test.columns = new_columns
    agg_test['flux_diff'] = agg_test['flux_max'] - agg_test['flux_min']
    agg_test['flux_dif2'] = (agg_test['flux_max'] - agg_test['flux_min']) / agg_test['flux_mean']
    agg_test['flux_w_mean'] = agg_test['flux_by_flux_ratio_sq_sum'] / agg_test['flux_ratio_sq_sum']
    agg_test['flux_dif3'] = (agg_test['flux_max'] - agg_test['flux_min']) / agg_test['flux_w_mean']

    # Merge all tsfresh features
    agg_test = pd.merge(agg_test, agg_test_ts, on='object_id')

    # Merge with meta data
    full_test = agg_test.reset_index().merge(
        right=meta_test,
        how='left',
        on='object_id'
    )

    # add absolute magnitude
    full_test = addAbsoluteMagnitudes(full_test)

    # Get relevant columns
    full_test[full_train.columns] = full_test[full_train.columns].fillna(train_mean)
    full_test_ss = ss.transform(full_test[full_train.columns])

    # Make predictions
    #preds = rf.predict_proba(full_test_ss) # RF
    preds = mlp.predict_proba(full_test_ss) # MLP

    # Compute preds_99 as the proba of class not being any of the others
    # preds_99 = 0.1 gives 1.769
    preds_99 = np.ones(preds.shape[0])
    for i in range(preds.shape[1]):
        preds_99 *= (1 - preds[:, i])

    # Store predictions
    preds_df = pd.DataFrame(preds, columns=class_names)
    preds_df['object_id'] = full_test['object_id']
    preds_df['class_99'] = 0.14 * preds_99 / np.mean(preds_99)

    if i_c == 0:
        preds_df.to_csv('predictions.csv',  header=True, index=False)
    else:
        preds_df.to_csv('predictions.csv',  header=False, mode='a', index=False)

    del agg_test, full_test, preds_df, preds

    if (i_c + 1) % 10 == 0:
        print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))

z = pd.read_csv('predictions.csv')
print z.dtypes

print(z.groupby('object_id').size().max())
print((z.groupby('object_id').size() > 1).sum())

z = z.groupby('object_id').mean()

z.to_csv('single_predictions.csv', index=True)
"""
