## CS342 Assignment 2, Kaggle ID: cs342u1500212

## FILE DIRECTORY:
1. data_exploration.py - used for exploratory data analysis and plotting graphs for the report.
2. data_augmentation.py - generates a new sliced and warped training set for time series to be stored in the current directory.
3. sklearn_RF.py - implementation of Random Forest for Task 4 and 5.
4. sklearn_MLP.py - implementation of Multilayer Perceptron using sklearn for Task 4 and 5.
5. keras_MLP.py - implementation of Multilayer Perceptron using keras for Task 4 and 5.
6. keras_CNN.py - implementation of Convolutional Neural Networks using keras for Task 6.
7. light_gbm_best_model.py - Best model developed with a leaderboard score of 1.052 (131st place on Kaggle)

## NOTES:
1. Data Augmentation for Task 6 was done using keras_MLP instead of keras_CNN,
   because keras_CNN is unable to predict the test set as discussed in the report.
2. All prediction functions are commented out.
3. All tuning functions are commented out.

## UNUSUAL LIBRARIES NEEDED:
1. tsfresh
2. y_categorical from keras (Some old versions of keras do not have this)
3. astropy (version 2.0.0 works for python 2.7)
4. lightgbm
