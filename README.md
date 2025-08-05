# 🚢 Titanic Survival Analysis & Prediction using Machine Learning

This project explores the famous [Titanic dataset](https://www.kaggle.com/competitions/titanic/) using **Exploratory Data Analysis (EDA)** and applies **Machine Learning models** to predict passenger survival.

## 📌 Objective
Predict whether a passenger survived the Titanic disaster based on personal and ticket-related features like Age, Sex, Pclass, Fare, Embarked, and Family size.

---

## 🔍 EDA (Exploratory Data Analysis)

Key questions explored during analysis:

- What is the overall survival rate?
- How does gender affect survival?
- Does class (Pclass) influence survival chances?
- What's the age distribution of survivors vs non-survivors?
- Does fare vary with class and survival?
- What role does family (SibSp & Parch) play?
- What is the survival rate across embarkation ports?

🧪 Tools used:
- `Pandas`, `NumPy` for data manipulation
- `Seaborn`, `Matplotlib` for visualizations

---

## 🧹 Data Preprocessing

- Removed duplicates and handled null values (Age imputed using **KNNImputer**)
- Dropped unnecessary columns like `PassengerId`, `Cabin`
- Feature engineered `FamilySize` from `SibSp + Parch`
- Encoded categorical variables using **LabelEncoder**

---

## 🤖 Machine Learning

Used **Random Forest Classifier** for prediction after splitting data into train-test sets.

### 📈 Model Performance:

- **Accuracy**: 81%
- Evaluated using **confusion matrix**, **precision**, **recall**, and **f1-score**

---

## 🧠 Libraries Used

```python
pandas, numpy, matplotlib, seaborn, scikit-learn
