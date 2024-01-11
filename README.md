# Stock Market Prediction Project

This repository contains a machine learning project focused on predicting the stock market using a dataset of around 44 stocks and various machine learning algorithms.

## Dataset

The dataset used for this project consists of approximately 44 stocks, with 10 features for each stock. These features include static values like Sector, P/E ratio, Return on Equity, Return on Assets, gross margin, sentiment score, etc. Each stock also has a target column that shows the likelihood of buying, holding, or selling the stock (ranging from 1.0 to 5.0).

## Project Overview

The project involves the following steps:

1. **Data Collection**: Historical market data of approximately 44 stocks was collected for the analysis.

2. **Data Preprocessing**: The collected data was preprocessed to make it suitable for training machine learning models including Naive Bayes, SVM, Decision Tree, and Neural Networks.

    - Label Encoded the categorical columns “Sectors” and “Observed”.
    - Transformed 5 class target variable to 3 classes (buy/hold/sell).
    - Scaled numeric input columns with StandardScaler() from sklearn.

    Preprocessing for Sentiment Analysis:
    - Removed Stop Words
    - Removed Punctuation
    - Removed digits
    - Converted all text to lowercase
    - Performed stemming

3. **Model Training**: The models were trained to predict whether to buy, hold or sell given stock details.

4. **Model Evaluation**: The trained models were evaluated based on their accuracy with a graph showcasing the performance of each model.

## Techniques and Tools Used

The project employed various data collection and preprocessing techniques. Machine learning algorithms used include Naive Bayes, SVM, Decision Tree, and Neural Networks for predicting decisions based on given stock details.

## Results and Findings

The results are visualized in an accuracy graph that compares the performance of each model. The project demonstrates the potential of machine learning for stock market prediction with detailed insights into algorithm accuracy and effectiveness in modeling financial datasets.
