
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import warnings
import itertools
import multiprocessing

import tensorflow as tf
import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical
from keras import backend as K
from keras import regularizers
from collections import Counter

from tsfresh.feature_extraction import extract_features
from astropy.cosmology import FlatLambdaCDM


# In[16]:


# Set cores to improve speed
cores=multiprocessing.cpu_count()


# ### Extracting Features from train set

# In[17]:


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


# In[5]:


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

# In[18]:


# Merge agg_train and meta data
full_train = agg_train.reset_index().merge(
    right=meta_train,
    how='outer',
    on='object_id'
)

# Take out y label
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


# In[8]:


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


# In[19]:


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

# In[20]:


# Scale data using Standard Scaler
full_train_new = full_train.copy()
ss = StandardScaler()
full_train_ss = ss.fit_transform(full_train_new)


# ### Optional: Task 3, Add Data Augmentation

# In[21]:


"""
sliced_train = pd.read_csv('./sliced_train.csv')
warped_train = pd.read_csv('./warped_train.csv')

def proprocessDataAugmentation(df):

    df['flux_ratio_sq'] = np.power(df['flux'] / df['flux_err'], 2.0)
    df['flux_by_flux_ratio_sq'] = df['flux'] * df['flux_ratio_sq']

    # Group by object id
    agg_df = df.groupby('object_id').agg(aggs)
    agg_df.columns = new_columns
    agg_df['flux_diff'] = agg_df['flux_max'] - agg_df['flux_min']
    agg_df['flux_dif2'] = (agg_df['flux_max'] - agg_df['flux_min']) / agg_df['flux_mean']
    agg_df['flux_w_mean'] = agg_df['flux_by_flux_ratio_sq_sum'] / agg_df['flux_ratio_sq_sum']
    agg_df['flux_dif3'] = (agg_df['flux_max'] - agg_df['flux_min']) / agg_df['flux_w_mean']

    del agg_df['mjd_max'], agg_df['mjd_min']

    # Merge with meta data
    full_df = agg_df.reset_index().merge(
        right=meta_train,
        how='left',
        on='object_id'
    )

    # Take out target
    y = full_df['target']
    del full_df['target']

    full_df[full_train.columns] = full_df[full_train.columns].fillna(train_mean)
    full_df_ss = ss.transform(full_df[full_train.columns])

    return full_df_ss, y

# Process data augmentation the same way
full_sliced_train_ss, sliced_y = proprocessDataAugmentation(sliced_train)
full_warped_train_ss, warped_y = proprocessDataAugmentation(warped_train)

# Add Data Augmentation values to full train
full_train_ss = np.append(full_train_ss, full_sliced_train_ss, axis=0)
full_train_ss = np.append(full_train_ss, full_warped_train_ss, axis=0)

# Add new y values to original
y = y.append(sliced_y)
y = y.append(warped_y)
y.reset_index()
"""


# ### Define Loss Function

# In[22]:


def mywloss(ytrue,ypred):
    yc = tf.clip_by_value(ypred,1e-15,1-1e-15)
    loss = -(tf.reduce_mean(tf.reduce_mean(ytrue*tf.log(yc),axis=0)/wtable))
    return loss

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


# ### Define Simple Neural Network

# In[30]:


K.clear_session()
def build_model(dropout_rate=0.25,activation='relu'):
    """
    Create Model:
    1) Layers of (512, 256, 128, 64)
    2) BatchNormalization for better stability
    3) Dropout rate of 0.25
    4) Rectified Linear activation and softmax at fully connected layer
    """
    start_neurons = 512

    model = Sequential()
    model.add(Dense(start_neurons, input_dim=full_train_ss.shape[1], activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons//2,activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons//4,activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons//8,activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate/2))

    model.add(Dense(len(classes), activation='softmax'))
    return model


# ### Calculate Class Weights

# In[24]:


# Convert y to one hot encoding labels
unique_y = np.unique(y)
class_map = dict()
for i,val in enumerate(unique_y):
    class_map[val] = i

y_map = np.zeros((y.shape[0],))
y_map = np.array([class_map[val] for val in y])
y_categorical = to_categorical(y_map)


# In[25]:


y_count = Counter(y_map)
global wtable
wtable = np.zeros((len(unique_y),))
for i in range(len(unique_y)):
    wtable[i] = float(y_count[i])/float(y_map.shape[0]) # OMG BUG HERE


# In[26]:


def plot_loss_acc(history):
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.title('model loss')
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    plt.legend(['train','Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['acc'][1:])
    plt.plot(history.history['val_acc'][1:])
    plt.title('model Accuracy')
    plt.ylabel('val_acc')
    plt.xlabel('epoch')
    plt.legend(['train','Validation'], loc='upper left')
    plt.show()


# In[31]:


clfs = []
oof_preds = np.zeros((len(full_train_ss), len(classes)))
epochs = 350
batch_size = 100


# Train classifier for 5 folds
for fold_, (trn_, val_) in enumerate(folds.split(y_map, y_map)):
    checkPoint = ModelCheckpoint("./keras.model",monitor='val_loss',mode = 'min', save_best_only=True, verbose=0)
    # Split test and train
    x_train, y_train = full_train_ss[trn_], y_categorical[trn_]
    x_valid, y_valid = full_train_ss[val_], y_categorical[val_]

    # Build model
    model = build_model(dropout_rate=0.5,activation='tanh')
    model.compile(loss=mywloss, optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid],
                    epochs=epochs,
                    batch_size=batch_size,shuffle=True,verbose=1,callbacks=[checkPoint])

    # Plot accuracy
    plot_loss_acc(history)
    print('Loading Best Model')

    # remember to delete this before training
    model.load_weights('./keras.model')

    # Get predicted probabilities for each class
    oof_preds[val_, :] = model.predict_proba(x_valid,batch_size=batch_size)
    print(multi_weighted_logloss(y_valid, model.predict_proba(x_valid,batch_size=batch_size)))

    # Append to classifier list
    clfs.append(model)

print('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(y_categorical,oof_preds))


# In[18]:


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


# In[19]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_map, np.argmax(oof_preds,axis=-1))
np.set_printoptions(precision=2)

# Get class names from sample submission
sample_sub = pd.read_csv('/modules/cs342/Assignment2/sample_submission.csv')
class_names = list(sample_sub.columns[1:-1])
del sample_sub;gc.collect()

# Plot non-normalized confusion matrix
plt.figure(figsize=(12,12))
foo = plot_confusion_matrix(cnf_matrix, classes=class_names,normalize=True,
                      title='Confusion matrix')


# ### Test Set Predictions

# In[22]:


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

    # Simple Aggregate
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
    del full_test['index']


    # Add: Absolute Magnitudes
    full_test = addAbsoluteMagnitudes(full_test)

    # Get relevant columns
    full_test[full_train.columns] = full_test[full_train.columns].fillna(train_mean)
    full_test_ss = ss.transform(full_test[full_train.columns])

    # Make predictions
    preds = None
    for clf in clfs:
        if preds is None:
            preds = clf.predict_proba(full_test_ss) / folds.n_splits
        else:
            preds += clf.predict_proba(full_test_ss) / folds.n_splits

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
        preds_df.to_csv('predictions2.csv',  header=True, index=False)
    else:
        preds_df.to_csv('predictions2.csv',  header=False, mode='a', index=False)

    del agg_test, full_test, preds_df, preds

    if (i_c + 1) % 10 == 0:
        print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))

z = pd.read_csv('predictions2.csv')

print z.dtypes
print z.shape

print(z.groupby('object_id').size().max())
print((z.groupby('object_id').size() > 1).sum())

z = z.groupby('object_id').mean()

z.to_csv('single_predictions2.csv', index=True)
"""
