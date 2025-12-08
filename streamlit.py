import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="Stock Analytics Dashboard",
    page_icon="",
    layout="wide"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #4e8cff; }
</style>
""", unsafe_allow_html=True)

st.title("Integrated Stock Analytics & Data Mining")
st.markdown("Combined OLAP Analysis, Visualization, and Machine Learning Models.")

# === 1. DATA LOADING & PREPROCESSING ===
@st.cache_data
def load_and_process_data(uploaded_file=None):
    # Load Data
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv("cleaned/All.csv")
        except FileNotFoundError:
            return None

    # Preprocessing logic from your script
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Filter for >= 2024 (As per your OLAP script)
    df = df[df['Date'].dt.year >= 2024].sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['YearMonth'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
    
    return df

# Sidebar
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload 'All.csv'", type=['csv'])

if uploaded_file:
    df = load_and_process_data(uploaded_file)
else:
    df = load_and_process_data()

if df is None:
    st.error("File 'cleaned/All.csv' not found. Please upload the file in the sidebar.")
    st.stop()

# Global Constants
TICKERS = sorted(df['Ticker'].unique())
COLORS = {
    'VTR': '#1f77b4', 'VJC': '#2ca02c', 'HVN': '#9467bd',
    'SCS': '#ff7f0e', 'AST': '#d62728', 'NCT': '#8c564b'
}
# Fallback color if ticker not in dictionary
def get_color(ticker):
    return COLORS.get(ticker, '#333333')

# === TABS ===
tab_olap, tab_indiv, tab_mining, tab_anom = st.tabs([
    "OLAP & Market Overview", 
    "Individual Ticker Analysis", 
    "Regression Models", 
    "Clustering & Anomalies"
])

# ==============================================================================
# TAB 1: OLAP ANALYSIS (Replicating olap_analysis.py tables & overview)
# ==============================================================================
with tab_olap:
    st.header("Market Overview (From 2024)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Quarterly Average Price")
        pivot_quarter = df.pivot_table('Close', ['Year', 'Quarter'], 'Ticker', 'mean').round(2)
        st.dataframe(pivot_quarter.style.background_gradient(cmap='Blues'), use_container_width=True)

    with col2:
        st.subheader("2. Yearly Growth (2024 vs 2025)")
        yearly_avg = df.groupby(['Ticker', 'Year'])['Close'].mean().unstack()
        if 2024 in yearly_avg.columns and 2025 in yearly_avg.columns:
            yearly_avg['Growth 24-25 (%)'] = ((yearly_avg[2025] / yearly_avg[2024]) - 1) * 100
            st.dataframe(yearly_avg.style.format("{:.2f}").background_gradient(subset=['Growth 24-25 (%)'], cmap='RdYlGn'), use_container_width=True)
        else:
            st.info("Insufficient data to calculate 2024-2025 growth.")

    st.divider()
    
    col3, col4 = st.columns([1, 2])
    with col3:
        st.subheader("3. Monthly Volume Matrix")
        pivot_volume_month = df.pivot_table('Volume', ['Year', 'Month'], 'Ticker', 'mean').fillna(0).astype(int)
        st.dataframe(pivot_volume_month.style.background_gradient(cmap='Greens'), use_container_width=True)
        
    with col4:
        st.subheader("4. Combined Price Trend")
        monthly_pivot = df.pivot_table('Close', 'YearMonth', 'Ticker', 'mean')[TICKERS]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        for t in TICKERS:
            ax.plot(monthly_pivot.index, monthly_pivot[t], label=t, color=get_color(t), linewidth=2)
        ax.set_title("Average Monthly Price Trend")
        ax.set_xticklabels(monthly_pivot.index, rotation=45)
        ax.legend()
        st.pyplot(fig)

# ==============================================================================
# TAB 2: INDIVIDUAL ANALYSIS (Fixed Date Overlap)
# ==============================================================================
with tab_indiv:
    st.header("Deep Dive: Individual Stock")
    
    selected_ticker = st.selectbox("Select Ticker to Analyze:", TICKERS)
    
    # Filter Data
    data_t = df[df['Ticker'] == selected_ticker].copy()
    
    # Layout: 2 Columns for charts
    c1, c2 = st.columns(2)
    
    # Chart 1: Price Trend
    with c1:
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(data_t['Date'], data_t['Close'], color='#1f77b4', linewidth=2)
        ax1.set_title(f'[{selected_ticker}] Daily Close Price')
        # Fix 1: Format date on Line Chart
        fig1.autofmt_xdate(rotation=45) 
        st.pyplot(fig1)
        
    # Chart 2: Dual Axis (Price + Volume)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(data_t['Date'], data_t['Close'], color='#1f77b4')
        ax2.set_ylabel('Price', color='#1f77b4')
        ax3 = ax2.twinx()
        ax3.bar(data_t['Date'], data_t['Volume'], alpha=0.3, color='gray')
        ax3.set_ylabel('Volume', color='gray')
        ax2.set_title(f'[{selected_ticker}] Price & Volume (Dual Axis)')
        # Fix 2: Format date on Dual Chart
        fig2.autofmt_xdate(rotation=45)
        st.pyplot(fig2)

    st.divider()
    c3, c4, c5 = st.columns(3)
    
    # Chart 3: Quarterly Price Bar
    with c3:
        q_data = data_t.groupby(['Year', 'Quarter'])['Close'].mean()
        labels = [f"{y}-Q{q}" for y, q in q_data.index]
        fig3, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(labels, q_data.values, color='#1f77b4', alpha=0.8)
        ax.set_title("Avg Price by Quarter")
        
        # Fix 3: Rotate 45 degrees + Align Right + Smaller Font
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        st.pyplot(fig3)
        
    # Chart 4: Monthly Volume (THE MAIN ISSUE IN YOUR IMAGE)
    with c4:
        m_data = data_t.groupby(['Year', 'Month'])['Volume'].mean() / 1e6
        labels_m = [f"{y}-{m:02d}" for y, m in m_data.index]
        fig4, ax = plt.subplots(figsize=(5, 4))
        ax.bar(labels_m, m_data.values, color='#ff7f0e', alpha=0.8)
        ax.set_title("Avg Volume (Millions) by Month")
        
        # Fix 4: Rotate 90 degrees because there are too many months
        ax.set_xticks(range(len(labels_m)))
        ax.set_xticklabels(labels_m, rotation=90, fontsize=8)
        st.pyplot(fig4)
        
    # Chart 5: Quarterly Volatility
    with c5:
        v_data = data_t.groupby(['Year', 'Quarter'])['Close'].std()
        labels_v = [f"{y}-Q{q}" for y, q in v_data.index]
        fig5, ax = plt.subplots(figsize=(5, 4))
        ax.plot(labels_v, v_data.values, 'o-', color='#9467bd')
        ax.fill_between(labels_v, v_data.values, alpha=0.2, color='#9467bd')
        ax.set_title("Volatility (Std Dev) by Quarter")
        
        # Fix 5: Rotate 45 degrees + Align Right
        ax.set_xticks(range(len(labels_v)))
        ax.set_xticklabels(labels_v, rotation=45, ha='right', fontsize=9)
        st.pyplot(fig5)

# ==============================================================================
# TAB 3: REGRESSION MODELS
# ==============================================================================
with tab_mining:
    st.header("Linear Regression Analysis")
    
    # ... (Keep your helper functions run_regression_same_day and run_regression_next_day here) ...
    # Helper functions code remains exactly the same as before
    def run_regression_same_day(data):
        results = []
        for ticker in data['Ticker'].unique():
            df_t = data[data['Ticker'] == ticker].dropna(subset=['Open', 'High', 'Low', 'Volume', 'Close'])
            if len(df_t) < 20: continue
            X = df_t[['Open', 'High', 'Low', 'Volume']]
            y = df_t['Close']
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            results.append({
                'Ticker': ticker, 'R2': r2_score(y, y_pred), 
                'MSE': mean_squared_error(y, y_pred)
            })
        return pd.DataFrame(results).sort_values('R2', ascending=False)

    def run_regression_next_day(data):
        results = []
        df_reg = data.copy()
        df_reg['Target_NextDay'] = df_reg.groupby('Ticker')['Close'].shift(-1)
        df_reg = df_reg.dropna(subset=['Target_NextDay'])
        scaler = StandardScaler()
        feats = ['Open', 'High', 'Low', 'Close', 'Volume']
        df_reg[feats] = scaler.fit_transform(df_reg[feats])
        
        for ticker in data['Ticker'].unique():
            df_t = df_reg[df_reg['Ticker'] == ticker]
            if len(df_t) < 20: continue
            X = df_t[feats]
            y = df_t['Target_NextDay']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.append({
                'Ticker': ticker, 'R2_NextDay': r2_score(y_test, y_pred),
                'MSE_NextDay': mean_squared_error(y_test, y_pred)
            })
        return pd.DataFrame(results).sort_values('R2_NextDay', ascending=False)

    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.subheader("Method 1: Predict Same Day")
        st.caption("Predict 'Close' using Open, High, Low, Volume")
        df_res1 = run_regression_same_day(df)
        
        # FIX IS HERE: Use subset or dict to format only numeric columns
        st.dataframe(
            df_res1.style.format({'R2': '{:.4f}', 'MSE': '{:.4f}'}), 
            use_container_width=True
        )
        
    with col_m2:
        st.subheader("Method 2: Predict Next Day")
        st.caption("Predict 'Close Tomorrow' using today's data (Scaled)")
        df_res2 = run_regression_next_day(df)
        
        # FIX IS HERE: Use subset or dict to format only numeric columns
        st.dataframe(
            df_res2.style.format({'R2_NextDay': '{:.4f}', 'MSE_NextDay': '{:.4f}'})
            .background_gradient(subset=['R2_NextDay'], cmap="RdYlGn"), 
            use_container_width=True
        )

# ==============================================================================
# TAB 4: CLUSTERING & ANOMALIES (Replicating datamining_analysis.py - Part 3 & 4)
# ==============================================================================
with tab_anom:
    col_clust, col_anom = st.columns([1, 1])
    
    # --- CLUSTERING ---
    with col_clust:
        st.header("1. K-Means Clustering")
        st.caption("Cluster stocks based on Average Return and Volume.")
        
        # Data prep
        df_mining = df.copy()
        df_mining['Return'] = df_mining.groupby('Ticker')['Close'].pct_change()
        df_cluster = df_mining.groupby('Ticker')[['Return', 'Volume']].mean().reset_index().dropna()
        
        # Scaling & KMeans
        scaler_km = StandardScaler()
        X_cluster = scaler_km.fit_transform(df_cluster[['Return', 'Volume']])
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_cluster['Cluster'] = kmeans.fit_predict(X_cluster)
        
        # Visualization
        fig_km, ax_km = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=df_cluster, x='Volume', y='Return', 
            hue='Cluster', style='Ticker', s=200, palette='viridis', ax=ax_km
        )
        # Log scale for volume often helps visualization
        ax_km.set_xscale('log')
        
        # Annotate points
        for i in range(df_cluster.shape[0]):
            ax_km.text(
                x=df_cluster.Volume.iloc[i], 
                y=df_cluster.Return.iloc[i], 
                s=df_cluster.Ticker.iloc[i], 
                fontdict=dict(color='black', size=10)
            )
        st.pyplot(fig_km)
        st.dataframe(df_cluster.sort_values('Cluster'), use_container_width=True)

    # --- ANOMALIES ---
    with col_anom:
        st.header("2. Anomaly Detection")
        st.caption("Detect days where Z-Score of Close Price > 2")
        
        # Z-Score calc
        df_mining['Z_Score'] = df_mining.groupby('Ticker')['Close'].transform(lambda x: (x - x.mean()) / x.std())
        anomalies = df_mining[abs(df_mining['Z_Score']) > 2].sort_values(['Ticker', 'Date'])
        
        st.write(f"**Total Anomalies Found:** {len(anomalies)}")
        st.dataframe(anomalies[['Date', 'Ticker', 'Close', 'Z_Score']].style.format({'Close':'{:.0f}', 'Z_Score':'{:.2f}'}), use_container_width=True)
        
        # Visualize Anomaly for selected ticker
        st.subheader("Visualize Anomalies")
        target_ticker = st.selectbox("Select Ticker to view anomalies:", TICKERS, key='anom_select')
        
        data_viz = df_mining[df_mining['Ticker'] == target_ticker]
        anom_viz = anomalies[anomalies['Ticker'] == target_ticker]
        
        fig_anom, ax_anom = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=data_viz, x='Date', y='Close', color='gray', alpha=0.5, label='Normal Price', ax=ax_anom)
        if not anom_viz.empty:
            sns.scatterplot(data=anom_viz, x='Date', y='Close', color='red', s=100, label='Anomaly', zorder=5, ax=ax_anom)
        ax_anom.set_title(f"Anomalies in {target_ticker}")
        st.pyplot(fig_anom)