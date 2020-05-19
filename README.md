## plasticc-astronomical-classification-kaggle

CS342 Machine Learning Project 2

https://www.kaggle.com/c/PLAsTiCC-2018

## Abstract

The LSST telescope is an important tool utilised to decipher the domain of our universe. It captures a glimpse of astronomical phenomenons, such as light sources and dark energy. Whilst it is impossible cover every aspect of the LSST, this report aims to handpick certain things to enrich our understanding of it.

The methods used for this Kaggle competition can be summarised in three main steps. In the first step, we focus on an in-depth understanding of our classes, exploring concepts such as time series flux/passbands, Redshift and Galactic/Extragalactic objects. Next, we generate new features based on different shapes of its time series, depending on whether an object is periodic or a burst. Finally, five different models were developed (sklearn_RF, sklearn_MLP, keras_MLP, keras_CNN, light_gbm). These were tuned methodologically using graphs and a forward/backward stepwise selection approach, followed by an analysis on the impact of feature engineering. Scores and submissions were documented accordingly throughout the process. The best model developed scored a weighted multi-log loss of 1.052 on the leaderboard, equivalent to approx. 131st place on Kaggle or Top 14% in the world.

## Directory:

    .
    ├── report.pdf                      # Full Report
    ├── light_gbm.ipynb                 # LightGBM + FFT + Feature Engineering
    ├── data_augmentation.ipynb         # Dynamic Time Warping
    ├── data_exploration.ipynb          # Exploratory Data Analysis
    └── code/                           # Folder for other experimental code