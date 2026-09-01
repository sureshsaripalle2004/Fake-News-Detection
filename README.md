# Fake News Detector

## Elevate Labs Internship – Project Phase

### Project Domain
Python Programming 

## Introduction

The Fake News Detector is a machine-learning-based application designed to classify news articles as FAKE or REAL. The project uses Natural Language Processing and machine learning techniques to analyze textual news content and provide a classification result with a confidence score.

## Abstract

This project implements an automated Fake News Detection system using TF-IDF feature extraction and machine learning classification. Fake and real news datasets were combined, cleaned, and preprocessed before training multiple machine learning models.

Three models were evaluated:

- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes

Linear SVM achieved the best overall performance with 99.77% accuracy and 99.75% F1-score and was selected as the final classifier. A calibrated version of the model is used in the deployed application to provide probability-based confidence scores.

The project includes a Streamlit web application for individual news classification and CSV-based batch prediction.

## Tools Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes
- Joblib
- Streamlit
- Matplotlib
- Google Colab
- GitHub
- Cloudflare Quick Tunnel

## Project Workflow

1. Dataset Collection
2. Data Preparation
3. Text Preprocessing
4. Train-Test Split
5. TF-IDF Feature Extraction
6. Machine Learning Model Training
7. Model Evaluation
8. Linear SVM Selection
9. Confidence Calibration
10. Streamlit Application Development
11. Testing and Deployment

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Linear SVM | 99.77% | 99.67% | 99.83% | 99.75% |
| Logistic Regression | 98.88% | 98.48% | 99.17% | 98.83% |
| Multinomial Naive Bayes | 96.18% | 94.87% | 97.22% | 96.03% |

## Features

- Individual news article classification
- FAKE / REAL prediction
- Confidence score
- Fake and Real probability
- CSV batch classification
- Downloadable batch results
- Model performance comparison

## Google Colab

The complete model training, evaluation, and model-generation workflow is available in Google Colab.

Colab Notebook:
https://colab.research.google.com/drive/1cV31sQfhx_1wuu7-R3LV99E7IeOHi8Eb?usp=sharing

The notebook includes:
- Dataset preprocessing
- TF-IDF feature extraction
- Model training
- Model comparison
- Linear SVM selection
- Confidence calibration
- Model saving
- Performance evaluation

## Project Structure

```text
Fake-News-Detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── fake_news_calibrated_pipeline.pkl
│
├── results/
│   ├── model_comparison.csv
│   ├── best_model_performance.csv
│   ├── calibrated_model_performance.csv
│   ├── model_accuracy_comparison.png
│   ├── model_metrics_comparison.png
│   └── linear_svm_confusion_matrix.png
│
└── sample_data/
    └── sample_news.csv
