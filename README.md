# 💳 Credit Risk Prediction for Loan Applicants

A machine learning-powered Streamlit web app that predicts whether a loan applicant is a **good** or **bad credit risk** based on their financial and personal details.

> 🧠 **Let data decide who gets credit!**

---

## 🚀 Features

- Interactive web interface using **Streamlit**
- Predict credit risk using a trained **XGBoost classifier**
- Uses **feature scaling** and **categorical encoding**
- Clean, responsive design with **custom UI styling**
- Input fields include age, job, account details, credit amount, loan duration, and more

---

## 📦 Project Structure

```
📁 Credit_Risk_Prediction_for_Loan_Applicants/
├── Credit_Risk_Prediction_for_Loan_Applicants.ipynb  # Model training & evaluation
├── app.py                                            # Streamlit web app
├── xgb_model.pkl                                     # Trained XGBoost model
├── scalar.pkl                                        # Fitted scaler for input normalization
├── requirements.txt                                  # Dependencies
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Aishee06/Credit_Risk_Prediction_for_Loan_Applicants.git
cd Credit_Risk_Prediction_for_Loan_Applicants
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Launch the App

```bash
streamlit run app.py
```

---

## 📝 Input Features

| Feature              | Type       | Description                                 |
|----------------------|------------|---------------------------------------------|
| Age                 | Numeric    | Applicant's age                             |
| Sex                 | Categorical (M/F) | Gender of the applicant              |
| Job                 | Categorical (0–3) | Job skill level                           |
| Housing             | Categorical (own/rent/free) | Housing situation         |
| Saving accounts     | Categorical | Level of savings                            |
| Checking account    | Categorical | Level of checking account                   |
| Credit amount       | Numeric    | Total credit amount requested           |
| Duration            | Numeric    | Duration of the loan in months              |
| Purpose             | Categorical | Reason for loan (e.g., education, car)     |

---

## 🔍 Model Details

- **Algorithm**: XGBoost Classifier
- **Training Dataset**: [German Credit Data](https://www.kaggle.com/datasets/uciml/german-credit?resource=download)
- **Preprocessing**:
  - Label encoding for categorical variables
  - Feature scaling with `StandardScaler`

---

## 📊 Example Output

The app classifies the applicant as either:

- ✅ **Good Credit Risk** (likely to repay the loan)
- ❌ **Bad Credit Risk** (high risk of default)

---

## 🎯 Conclusion

The Credit Risk Prediction project highlights how machine learning can make a significant impact in the financial sector. By using XGBoost and a well-prepared dataset, a tool has been built that can help financial institutions make smarter, more data-driven lending decisions, potentially reducing biases in the credit approval process.

This project is just the beginning—a practical application with room for improvement and further development into more advanced credit risk systems.
