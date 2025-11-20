## ***Titanic Survival Prediction***

Machine Learning Project – Random Forest Classifier

A complete end-to-end Machine Learning project that predicts whether a passenger survived the Titanic disaster.
This project uses Random Forest Classifier, including data preprocessing, EDA, feature engineering, model training, evaluation, and model saving.

---

**Project Overview**

The Titanic dataset is one of the most famous beginner-friendly ML datasets.
The aim is to build a classification model that predicts Survived (1) or Not Survived (0) based on passenger details such as:

Age

Sex

Pclass

Fare

No.of.Parents visited

No.of.Siblings visited

---

**This project includes:**

✔ Data cleaning

✔ Handling missing values

✔ Feature engineering

✔ Model building

✔ Hyperparameter tuning

✔ Evaluation (accuracy, confusion matrix, classification report)

✔ Saving model with Pickle

✔ A simple prediction script

---

**Algorithm Used – Random Forest Classifier**

Random Forest is an ensemble technique that combines multiple Decision Trees.

**Why Random Forest?**

Handles missing values well

Works for both categorical & numerical data

Resistant to overfitting

Provides high accuracy

---

**Technologies Used**

Language	Python

Libraries	pandas, numpy, sklearn, matplotlib, seaborn

Algorithm	Random Forest Classifier

IDE / Notebook	Jupyter Notebook / VS Code

---

**Steps Followed**

1️⃣ Load the dataset

import pandas as pd

df = pd.read_csv("titanic.csv")


2️⃣ Handle missing values

Fill missing Age with median

Fill Embarked with the mode

Drop irrelevant columns (Name, Ticket, Cabin)


3️⃣ Convert categorical → numerical

df['Sex'] = df['Sex'].map({'male':0, 'female':1})

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


4️⃣ Split the dataset

from sklearn.model_selection import train_test_split

X = df.drop('Survived', axis=1)

y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


5️⃣ Train Random Forest

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)

model.fit(X_train, y_train)


6️⃣ Evaluate the model

from sklearn.metrics import accuracy_score, classification_report

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))

print(classification_report(y_test, preds))


7️⃣ Save the trained model

import pickle

pickle.dump(model, open('model.pkl', 'wb'))

---
**Model Performance**

Metric	Score

Accuracy	~82–86%

Precision	Good

Recall	Good

F1-Score	Stable

(Random Forest performance may slightly vary based on preprocessing.)

---

**Conclusion**

This project shows how Random Forest can be used to model real-world classification problems efficiently.
It demonstrates data preprocessing, handling categorical data, model training, and deployment-ready saving
