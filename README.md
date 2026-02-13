# Email-spam-detection
# Project Overview
This project is a simple Email Spam Detection System built using Machine Learning.The model classifies emails as:
Spam
Not Spam
It uses text preprocessing and classification techniques to achieve high accuracy.
# Problem Statement
Email inboxes receive many unwanted spam messages.
Manually filtering them is difficult and time-consuming.This project aims to automatically detect spam emails using machine learning.

# Objective
To classify emails as Spam or Not Spam.
To preprocess text data using NLP techniques.
To train a classification model.
To evaluate performance using accuracy and confusion matrix.
# How It Works
1.Load dataset
2.Clean and preprocess text
3.Convert text into numerical form using CountVectorizer
4.Split data into training and testing sets
5.Train Naive Bayes classifier
6.Evaluate using accuracy and confusion matrix

# Model Performance
Accuracy: 93%
Confusion Matrix:

	Predicted Spam Predicted Not Spam
Actual Spam	13	2
Actual Not Spam 0	15
 
# output screenshot: ![Confusion Matrix](confusion_matrix.png)

