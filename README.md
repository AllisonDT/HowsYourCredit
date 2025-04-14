# HowsYourCredit

This repository contains the code and resources needed for the HowsYourCredit project

## 1. Cloning the Repository

Open your terminal and run the following command to clone the repository to your local machine:

```bash
git clone https://github.com/AllisonDT/HowsYourCredit.git
cd HowsYourCredit
```

## Add Data Sets to local repository
Download these datasets and add them to your local: https://www.kaggle.com/datasets/parisrohan/credit-score-classification

## 2. Cleaning Training Data
```bash
python3 cleanTrainingData.py
```

## 3. Cleaning Testing Data
```bash
python3 cleanTestingData.py
```

## 4. Comparisons
Make sure that each folder has a predictions_model.csv file in it!!
```bash
python3 predictionComparison.py
```
Make sure that each folder has a training_model.csv file in it!!
```bash
python3 trainingComparison.py
```
