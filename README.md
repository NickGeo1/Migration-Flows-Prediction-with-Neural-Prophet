# Migration-Flows-Prediction-with-Neural-Prophet
A machine learning oriented thesis project for migration flows prediction regarding 9 western Balkan countries

Neural Prophet is a machine/deep learning extension of Prophet which is made by Facebook.
It is a fairly user-friendly and capable time series prediction model, as it includes several default settings and classical prediction modules, some of which can be extended to neural network architecture. It has built-in, state-of-the-art machine learning functionalities, such as the cost function optimization and reguralization algorithms, which are flexible to tune thanks to the Neural Prophet implementation package, pytorch.

More Info can be found at the official website: https://neuralprophet.com <br>
or at the official GitHub repositiry: https://github.com/ourownstory/neural_prophet

We introduce two different approaches regarding the training of the models:
1. We train one model on the first 80% for each timeseries and we test it at the rest 20%
2. We train 9 models where each one of them is using 8 timeseries as train set and 1 as test set

The results can be observed at the "80_20_metrics.xlsx" and "100_all_100_one_metrics.xlsx" files. <br>
The metrics we used to evaluate the models are MAE normalised by the range of values and R^2:

![εικόνα](https://github.com/NickGeo1/Migration-Flows-Prediction-with-Neural-Prophet/assets/60719667/957cc48b-648c-4bc3-819b-2bda9fa42d15) <br>
![εικόνα](https://github.com/NickGeo1/Migration-Flows-Prediction-with-Neural-Prophet/assets/60719667/f90a9ec4-57c4-4bad-a68e-cc6b2489f3e6)



The dataset can be found at "balkan_route.xlsx" file.
