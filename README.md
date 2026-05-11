# 🍽️ Restaurant Tip Prediction & Customer Segmentation

### Analyzing restaurant tipping behavior and grouping customers using Machine Learning.

## 📌 Project Overview
Can we predict how much tip a customer will leave based on their order? This project applies both **Supervised** and **Unsupervised Machine Learning** techniques to analyze tipping behavior in a restaurant. 

We analyzed the dataset to predict the exact tip amount using regression models, and grouped customers into specific segments based on their spending habits to help restaurants understand their client base better.

## 🧠 Machine Learning Models Used
**1. Tip Prediction (Supervised Learning):**
- **Linear Regression:** Achieved the highest accuracy ($R^2$ = 0.44) in predicting the tip amount.
- **Random Forest Regressor:** Used for comparison to evaluate predictive performance.

**2. Customer Segmentation (Unsupervised Learning):**
- **K-Means Clustering:** Segmented customers into 3 distinct groups:
  - 🥉 *Economic:* Low bill, low tip.
  - 🥈 *Standard:* Average bill, average tip.
  - 🥇 *High-End:* High bill, generous tip.

## 📊 Dataset & Features
The dataset (`tips.csv`) contains the following features:
- **Total Bill:** Cost of the meal (The most important feature).
- **Tip:** The target variable.
- **Sex & Smoker:** Customer demographics and preferences.
- **Day & Time:** Day of the week and meal time (Lunch/Dinner).
- **Size:** Number of people at the table.

## 🏆 Key Findings
- **Total Bill** is by far the most significant factor in determining the tip amount.
- Factors like gender or smoking status have a very minimal impact on the tip size.
- Linear Regression outperformed Random Forest for this specific dataset.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
- **Concepts:** Regression, Clustering, Data Visualization, Exploratory Data Analysis (EDA).

## 👥 Project Team
Developed by Computer Engineering students at Esenyurt University:
- **Ömer** (Me)
- Abdu Khalek Aktaa

---
*Check out the attached `rapor.pdf` (Academic Report) and `sonuclar.pdf` (Visual Results) for deep insights and visualizations!*
