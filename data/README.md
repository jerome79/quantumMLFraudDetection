# Data Directory

This directory should contain the credit card fraud detection dataset.

## Structure

- `raw/` - Raw data files
  - `creditcard.csv` - The ULB/Worldline Credit Card Fraud Detection dataset
- `processed/` - Preprocessed data files

## Downloading the Dataset

The dataset can be downloaded from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place the `creditcard.csv` file in the `data/raw/` directory.

## Dataset Description

The dataset contains credit card transactions made by European cardholders in September 2013. 
It contains 284,807 transactions with 492 frauds (0.172% fraud rate).

Features:
- Time: Seconds elapsed between this transaction and the first transaction
- V1-V28: Principal components obtained with PCA (anonymized features)
- Amount: Transaction amount
- Class: Target variable (1 for fraud, 0 for legitimate)
