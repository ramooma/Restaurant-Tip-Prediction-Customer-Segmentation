import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 1. تحميل البيانات (نفترض وجود ملف بيانات تجارة إلكترونية)
# المتغيرات: Time_on_App, Time_on_Website, Length_of_Membership, Yearly_Amount_Spent
df = pd.read_csv('Ecommerce_Customers.csv') 

# 2. المعالجة المسبقة (Preprocessing)
X = df[['Time_on_App', 'Time_on_Website', 'Length_of_Membership']]
y = df['Yearly_Amount_Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. التنبؤ باستخدام Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)

# 4. التنبؤ باستخدام Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)

print(f"Linear Regression R2 Score: {lr_r2}")
print(f"Random Forest R2 Score: {rf_r2}")

# 5. تجزئة العملاء باستخدام K-Means
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Yearly_Amount_Spent', 'Length_of_Membership']])

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# 6. التصوير البياني للمجموعات
sns.scatterplot(data=df, x='Length_of_Membership', y='Yearly_Amount_Spent', hue='Cluster', palette='viridis')
plt.title('Customer Segmentation (K-Means)')
plt.show()