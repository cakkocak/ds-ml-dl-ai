# main.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 ML Model Deployment — Iris Dataset")

# load iris dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

st.write("### Dataset Preview")
st.write(X.head())
st.write("### Target Classes")
st.write(list(iris.target_names))

# sidebar for model selection
st.sidebar.header("Choose Model & Params")
model_name = st.sidebar.selectbox(
    "Select ML Model",
    (
        "Logistic Regression",
        "Random Forest",
        "SVM",
        "KNN",
        "Naive Bayes",
        "Gradient Boosting",
        "Decision Tree",
        "MLP Classifier"
    ),
)

# split data
test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 30)
random_state = st.sidebar.number_input("Random State", value=42, step=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=random_state
)

@st.cache(allow_output_mutation=True)
def get_model(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=200)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=100)
    if name == "SVM":
        return SVC()
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=5)
    if name == "Naive Bayes":
        return GaussianNB()
    if name == "Gradient Boosting":
        return GradientBoostingClassifier()
    if name == "Decision Tree":
        return DecisionTreeClassifier()
    if name == "MLP Classifier":
        return MLPClassifier(max_iter=500)
    return LogisticRegression()

model = get_model(model_name)

# train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.write(f"### 🧠 Model: **{model_name}**")
st.write(f"**Accuracy:** {acc:.2%}")

st.write("### 🔢 Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.write("### 📦 Model Details")
st.write(model)
