import pandas as pd
import numpy as np
import os
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cấu hình
warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x: '%.4f' % x)

print("--- BAT DAU DATA MINING ---")

# 1. Tải dữ liệu
try:
    df = pd.read_csv("cleaned/All.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df[df['Date'].dt.year >= 2024].copy().sort_values(by=['Ticker', 'Date'])
    
    if df.empty:
        print("Khong co du lieu tu nam 2024.")
        exit()
        
    print(f"Da tai {len(df)} dong du lieu.")
except Exception as e:
    print(f"Loi doc file: {e}")
    exit()

# Tao thu muc luu ket qua
os.makedirs("datamining_results", exist_ok=True)

# ---------------------------------------------------------
# 2. HOI QUY TUYEN TINH (LINEAR REGRESSION)
# ---------------------------------------------------------
print("\n--- 2. Hoi quy tuyen tinh ---")

# Cach 1: Du doan Close cung ngay (Close ~ Open, High, Low, Volume)
results_c1 = []
for ticker in df['Ticker'].unique():
    df_t = df[df['Ticker'] == ticker].dropna(subset=['Open', 'High', 'Low', 'Volume', 'Close'])
    if len(df_t) < 20: continue
    
    X = df_t[['Open', 'High', 'Low', 'Volume']]
    y = df_t['Close']
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    results_c1.append({
        'Ticker': ticker,
        'R2_Score': r2_score(y, y_pred),
        'MSE': mean_squared_error(y, y_pred),
        'Intercept': model.intercept_
    })

df_res_c1 = pd.DataFrame(results_c1).sort_values('R2_Score', ascending=False)
df_res_c1.to_csv("datamining_results/01_regression_same_day.csv", index=False)
print("Cach 1 (Cung ngay) - Da luu file CSV.")
print(df_res_c1[['Ticker', 'R2_Score', 'MSE']].head())

# Cach 2: Du doan Close ngay mai (Target_NextDay ~ Open, High, Low, Close, Volume)
results_c2 = []
df_reg = df.copy()
df_reg['Target_NextDay'] = df_reg.groupby('Ticker')['Close'].shift(-1)
df_reg = df_reg.dropna(subset=['Target_NextDay'])

scaler = StandardScaler()
features = ['Open', 'High', 'Low', 'Close', 'Volume']
df_reg[features] = scaler.fit_transform(df_reg[features])

for ticker in df['Ticker'].unique():
    df_t = df_reg[df_reg['Ticker'] == ticker]
    if len(df_t) < 20: continue

    X = df_t[features]
    y = df_t['Target_NextDay']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results_c2.append({
        'Ticker': ticker,
        'R2_Score': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'Intercept': model.intercept_
    })

df_res_c2 = pd.DataFrame(results_c2).sort_values('R2_Score', ascending=False)
df_res_c2.to_csv("datamining_results/02_regression_next_day.csv", index=False)
print("\nCach 2 (Ngay mai) - Da luu file CSV.")
print(df_res_c2[['Ticker', 'R2_Score', 'MSE']].head())

# ---------------------------------------------------------
# 3. GOM CUM K-MEANS
# ---------------------------------------------------------
print("\n--- 3. Gom cum K-Means ---")

df['Return'] = df.groupby('Ticker')['Close'].pct_change()
df_cluster = df.groupby('Ticker')[['Return', 'Volume']].mean().reset_index().dropna()

# Chuan hoa du lieu
scaler_km = StandardScaler()
X_cluster = scaler_km.fit_transform(df_cluster[['Return', 'Volume']])

# K-Means k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_cluster['Cluster'] = kmeans.fit_predict(X_cluster)

df_cluster.to_csv("datamining_results/03_kmeans_clusters.csv", index=False)
print("Da luu ket qua gom cum CSV.")
print(df_cluster.sort_values('Cluster'))

# ---------------------------------------------------------
# 4. PHAT HIEN BAT THUONG (Z-SCORE)
# ---------------------------------------------------------
print("\n--- 4. Phat hien bat thuong ---")

df['Z_Score'] = df.groupby('Ticker')['Close'].transform(lambda x: (x - x.mean()) / x.std())
anomalies = df[abs(df['Z_Score']) > 2].sort_values(['Ticker', 'Date'])

anomalies.to_csv("datamining_results/04_anomalies.csv", index=False)
print(f"Phat hien {len(anomalies)} dong bat thuong.")
print("Da luu ket qua bat thuong CSV.")

print("\n--- HOAN THANH DATA MINING ---")
