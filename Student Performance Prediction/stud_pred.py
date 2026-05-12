import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Ultra Student Predictor", layout="wide")

st.title("🎓 Student Performance Predictor (Ultra)")

# ===== FILE UPLOAD =====
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    # Default dataset
    df = pd.DataFrame({
        'hours': [1,2,3,4,5,6,7,8],
        'attendance': [60,65,70,75,80,85,90,95],
        'marks': [35,40,50,55,65,70,80,90]
    })

# ===== SHOW DATA =====
st.subheader("📊 Dataset Preview")
st.dataframe(df)

# ===== FEATURES =====
X = df.drop('marks', axis=1)
y = df['marks']

# ===== SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ===== MODELS =====
lr = LinearRegression()
rf = RandomForestRegressor()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# ===== EVALUATION =====
lr_score = r2_score(y_test, lr.predict(X_test))
rf_score = r2_score(y_test, rf.predict(X_test))

# ===== SIDEBAR INPUT =====
st.sidebar.header("Input Features")

input_data = []
for col in X.columns:
    val = st.sidebar.slider(f"{col}", int(X[col].min()), int(X[col].max()))
    input_data.append(val)

model_choice = st.sidebar.selectbox("Select Model", ["Linear Regression", "Random Forest"])

# ===== PREDICTION =====
if model_choice == "Linear Regression":
    prediction = lr.predict([input_data])[0]
else:
    prediction = rf.predict([input_data])[0]

# ===== RESULT =====
st.subheader("🎯 Prediction")
st.success(f"Predicted Marks: {round(prediction,2)}")

# ===== MODEL PERFORMANCE =====
st.subheader("📈 Model Comparison")

col1, col2 = st.columns(2)

with col1:
    st.metric("Linear Regression", round(lr_score,2))

with col2:
    st.metric("Random Forest", round(rf_score,2))

# ===== FEATURE IMPORTANCE =====
st.subheader("🧠 Feature Importance (Random Forest)")

importance = rf.feature_importances_

fig, ax = plt.subplots()
ax.bar(X.columns, importance)
ax.set_title("Feature Importance")

st.pyplot(fig)

# ===== GRAPH =====
st.subheader("📉 Visualization")

fig2, ax2 = plt.subplots()
ax2.scatter(df[X.columns[0]], y)
ax2.set_xlabel(X.columns[0])
ax2.set_ylabel("Marks")

st.pyplot(fig2)

# ===== HISTORY =====
if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Save Prediction"):
    st.session_state.history.append(prediction)

st.subheader("📜 Prediction History")
st.write(st.session_state.history)