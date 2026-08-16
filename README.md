# 🚢 Titanic Survival Prediction — Logistic Regression

<p align="center">

  <img src="https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas" />
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" />

</p>

<p align="center">
  <b>🚢 A Machine Learning Classification Project using Logistic Regression to predict Titanic passenger survival.</b>
</p>

---

## 📌 Project Overview

The **Titanic Survival Prediction** project is a Machine Learning classification project built using **Logistic Regression**.

The objective of this project is to predict whether a passenger on the Titanic **survived or did not survive** based on passenger-related features.

This project demonstrates the complete basic Machine Learning workflow:

> **Data → Data Cleaning → Feature Selection → Train/Test Split → Model Training → Prediction → Evaluation**

---

## 🎯 Objective

The main objective is to build a classification model that predicts:

| Value | Meaning           |
| ----- | ----------------- |
| `0`   | ❌ Did Not Survive |
| `1`   | ✅ Survived        |

The model learns patterns from historical Titanic passenger data and uses those patterns to make predictions for unseen passengers.

---

## 🧠 Machine Learning Algorithm

### Logistic Regression

**Logistic Regression** is a supervised Machine Learning algorithm mainly used for **classification problems**.

In this project, Logistic Regression is used for **Binary Classification** because there are only two possible outcomes:

```text
0 → Not Survived
1 → Survived
```

The model calculates the probability of survival and converts it into a class prediction.

---

## 🔄 Machine Learning Workflow

```text
                🚢 Titanic Dataset
                       │
                       ▼
                🔍 Data Understanding
                       │
                       ▼
                 🧹 Data Cleaning
                       │
                       ▼
                🎯 Feature Selection
                       │
                       ▼
              ✂️ Train/Test Split
                       │
                       ▼
            🤖 Logistic Regression
                       │
                       ▼
                  🔮 Prediction
                       │
                       ▼
              📊 Model Evaluation
                       │
                       ▼
                 ✅ Final Result
```

---

## 📂 Dataset

The project uses the **Titanic passenger dataset**.

The dataset contains information about passengers such as:

* Passenger Class
* Age
* Fare
* Sex
* Survival status
* Family-related information
* Passenger-related information

### 🎯 Target Variable

```text
Survived
```

Where:

```text
0 → Not Survived
1 → Survived
```

---

## 📊 Features Used

The model uses selected passenger features as input variables.

| Feature  | Description     |
| -------- | --------------- |
| `Pclass` | Passenger class |
| `Age`    | Passenger age   |
| `Fare`   | Ticket fare     |

### Target

```text
Survived
```

---

## 🛠️ Technologies & Libraries

### 💻 Programming Language

* 🐍 Python

### 📚 Libraries

* 🐼 Pandas
* 🔢 NumPy
* 🤖 Scikit-learn
* 📊 Matplotlib

### 🧰 Tools

* Jupyter Notebook
* VS Code
* GitHub

---

## 🧹 Data Preprocessing

Before training the model, the dataset is prepared.

### 1️⃣ Handling Missing Values

Missing values can affect Machine Learning models.

For example, missing values in the `Age` column can be handled using the median:

```python
X["Age"] = X["Age"].fillna(X["Age"].median())
```

---

### 2️⃣ Feature Selection

Input features:

```python
X = df[["Pclass", "Age", "Fare"]]
```

Target:

```python
y = df["Survived"]
```

---

## ✂️ Train-Test Split

The dataset is divided into:

```text
80% → Training Data
20% → Testing Data
```

Using:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Why?

The model learns from the training data and is then tested on unseen data.

---

# 🤖 Model Building

The Logistic Regression model is created using Scikit-learn:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
```

---

## 🏋️ Model Training

The model is trained using:

```python
model.fit(X_train, y_train)
```

The model learns the relationship between the passenger features and survival outcome.

---

## 🔮 Prediction

After training, predictions are generated using:

```python
y_pred = model.predict(X_test)
```

The model predicts:

```text
0 → Not Survived
1 → Survived
```

---

## 📈 Model Evaluation

The model can be evaluated using classification metrics such as:

### 🎯 Accuracy

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

Accuracy tells us the percentage of predictions that were correct.

---

### 📊 Confusion Matrix

A confusion matrix helps understand:

* True Positive
* True Negative
* False Positive
* False Negative

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
```

---

### 📋 Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

This provides:

* Precision
* Recall
* F1-score
* Support

---

## 🔮 Predicting a New Passenger

The trained model can also predict survival for a new passenger.

Example:

```python
new_passenger = pd.DataFrame(
    [[1, 25, 100]],
    columns=["Pclass", "Age", "Fare"]
)

prediction = model.predict(new_passenger)

print("Prediction:", prediction)
```

Possible output:

```text
Prediction: [1]
```

Meaning:

```text
✅ Passenger predicted to survive
```

---

## 🧮 Probability Prediction

Logistic Regression can also provide prediction probabilities.

```python
probability = model.predict_proba(new_passenger)

print(probability)
```

Example:

```text
[[0.15 0.85]]
```

Meaning approximately:

```text
15% → Not Survived
85% → Survived
```

---

## 📁 Project Structure

```text
Titanic-Survival-Prediction-Logistic-Regression/
│
├── 📄 Titanic_Survival_Prediction.ipynb
├── 📄 Titanic-Dataset.csv
├── 📄 README.md
│
└── 📁 images/
    ├── 📊 confusion_matrix.png
    └── 📈 model_results.png
```

> File names can be changed according to the files uploaded in the repository.

---

## 📊 Results

The Logistic Regression model successfully performs binary classification on the Titanic dataset.

### Model Output

```text
Algorithm       : Logistic Regression
Problem Type    : Binary Classification
Target          : Survived
Class 0         : Not Survived
Class 1         : Survived
```

Add your **actual accuracy** here after running the final version of your notebook:

```text
🎯 Accuracy: XX%
```

---

## 💡 Key Learnings

Through this project, I learned:

* ✅ What is Logistic Regression
* ✅ Binary Classification
* ✅ Feature and Target Selection
* ✅ Train-Test Split
* ✅ Data Preprocessing
* ✅ Handling Missing Values
* ✅ Model Training
* ✅ Model Prediction
* ✅ Accuracy Score
* ✅ Confusion Matrix
* ✅ Classification Report
* ✅ Prediction Probability
* ✅ Making predictions for new data

---

## 🚀 Future Improvements

This project can be improved by:

* 🔹 Adding more useful features
* 🔹 Encoding categorical variables such as `Sex`
* 🔹 Feature scaling where appropriate
* 🔹 Hyperparameter tuning
* 🔹 Comparing multiple classification algorithms
* 🔹 Using Decision Tree
* 🔹 Using Random Forest
* 🔹 Comparing model performance
* 🔹 Adding ROC-AUC analysis

---

## 🏆 Project Highlights

<p align="center">

|          🚢 Project         |     🤖 Algorithm    |       🎯 Problem      | 🐍 Language |
| :-------------------------: | :-----------------: | :-------------------: | :---------: |
| Titanic Survival Prediction | Logistic Regression | Binary Classification |    Python   |

</p>

---

## 👨‍💻 Author

### **Atharva Santosh Avhad**

🎓 **B.Tech — Artificial Intelligence & Data Science**

🏫 **Matoshri College of Engineering and Research Center, Nashik**

🔗 **GitHub:** [atharva00771](https://github.com/atharva00771)

🔗 **LinkedIn:** [Atharva Avhad](https://www.linkedin.com/in/atharvaavhad07)

---

## ⭐ If You Like This Project

If you found this project useful or interesting, consider giving the repository a ⭐ **Star**!

It motivates me to build and share more **Data Science & Machine Learning projects.**

---

<p align="center">

### 🚢 Predict • Learn • Improve • Repeat 🔥

**Made with 🐍 Python & 🤖 Machine Learning**

</p>
