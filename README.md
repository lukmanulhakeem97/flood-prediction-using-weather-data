# flood-prediction-using-weather-data

This project focuses on the importance of flood prediction in Kerala, considering its
geographical features and susceptibility to heavy monsoon rains. To address this, a flood
prediction model was developed for Alappuzha using binary classification with various
machine learning algorithms, including Logistic Regression, K-Nearest Neighbors
(KNN), Gaussian Naive Bayes, and XGBoost Classifier. The study utilized a dataset
obtained from the open-source Open Meteo platform, which consisted of features such as
maximum temperature, minimum temperature, and rainfall. Through comparative
analysis, it was determined that the XGBoost Classifier exhibited superior performance in
flood prediction. This research highlights the significance of accurate flood prediction
methods and the potential of machine learning algorithms to aid in timely evacuation
measures, resource allocation, and effective disaster management, thereby minimizing the
loss of life and property in flood-prone regions like Kerala.

##  1. Data Sources
We utilized two main data sources for our study: the Open Meteo dataset and a
Kaggle dataset. The Open Meteo dataset provided us with daily weather reports for the
Alappuzha region, starting from March 1940. This dataset served as the foundation for
our weather data, which we used for model training. Additionally, we leveraged the
Kaggle dataset to obtain the actual dates of high rainfall events. This dataset included
information on red and yellow alerts, allowing us to identify flood possibilities. We added
a binary class label column to our weather dataset, assigning a value of 1 to flood
possibilities and 0 to other instances.

## 2. Data Preparation
To ensure that our weather dataset contained the necessary class label
information, we merged the Open-Meteo and Kaggle datasets. We matched the dates
from the Open Meteo dataset with the corresponding flood labels from the Kaggle
dataset. However, we encountered missing label information for the flood occurrences in
2005 and 1961. To address this, we employed semi-supervised learning techniques to
label the missed information for these years. After combining and labeling the datasets,
we proceeded with the data preparation steps, including cleaning the data, handling
missing values, and encoding categorical variables.

## 3. Model Selection
To train our model, we evaluated various machine learning algorithms. After
comparing their performance, we determined that AG Boost exhibited superior results for
our flood prediction task. Therefore, we selected AG Boost as our chosen algorithm for
model training and prediction.

## 4 Model Training
We performed model training using the labeled weather dataset and AG Boost
algorithm. The training process involved iteratively adjusting the model's parameters to
minimize a predefined loss function. We optimized the hyperparameters of the AG Boost
algorithm through techniques such as cross-validation and grid search. Following
training, we evaluated the performance of the trained model using appropriate evaluation
metrics, such as accuracy, precision, recall, and F1-score.

## 5. Real-Time Prediction
For real-time predictions, we utilized the trained model. However, before making
predictions on unseen data, we needed to account for differences in the data distribution.
To address this, we converted the mean and variance of the training data to another file.
This transformation enabled us to perform standard scaling on the unseen data, ensuring
it conformed to the same normal distribution as the training data. Real-time prediction
uses real-time daily weather data from Open-Meteo API. This is done using the
Open-Meteo python library called Openmeteopy.

## 6. Evaluation Metrics
Our model has a good performance score on validation dataset. Validation carried
on input sample of size 6080 records. Out of 6080, only 10 records are positive labels (1)
and remaining 6070 are negative labels (0).
Trained XGBoost model able to correctly classify 9 positive out of 10 true
positive labels and have one miss classification. Negative classes 6070 records are
completely correctly classified. Since our model is for predicting a disaster, we focus
more on improving the recall matrix in order to minimize miss-classification of positive
labels as negative. Our model was able to achieve a recall of 0.90. This improvement is
achieved by applying balancing on model training data.
