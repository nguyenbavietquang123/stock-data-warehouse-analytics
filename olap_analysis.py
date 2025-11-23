import pandas as pd
import matplotlib
matplotlib.use('Agg')  # MULTI-THREADING
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# --- CẤU HÌNH ---
warnings.filterwarnings("ignore")
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 11, 'figure.dpi': 300, 'savefig.dpi': 300,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--'
})

# Khóa để tránh xung đột khi lưu file
plot_lock = threading.Lock()

COLORS = {
    'VTR': '#1f77b4', 'VJC': '#2ca02c', 'HVN': '#9467bd',
    'SCS': '#ff7f0e', 'AST': '#d62728', 'NCT': '#8c564b'
}

# --- ĐỌC DỮ LIỆU ---
df = pd.read_csv("cleaned/All.csv", parse_dates=['Date'])
df = df[df['Date'].dt.year >= 2024].sort_values(['Ticker', 'Date']).reset_index(drop=True)

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['YearMonth'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)

os.makedirs("olap_results", exist_ok=True)
os.makedirs("olap_charts", exist_ok=True)

tickers = sorted(df['Ticker'].unique())

# --- BẢNG OLAP ---
pivot_quarter = df.pivot_table('Close', ['Year', 'Quarter'], 'Ticker', 'mean').round(2)
pivot_quarter.to_csv("olap_results/01_quarterly_price.csv")

yearly_avg = df.groupby(['Ticker', 'Year'])['Close'].mean().unstack()
if 2024 in yearly_avg.columns and 2025 in yearly_avg.columns:
    yearly_avg['Growth 24-25 (%)'] = ((yearly_avg[2025] / yearly_avg[2024]) - 1) * 100
yearly_avg.to_csv("olap_results/02_growth_24_25.csv")

pivot_volume_month = df.pivot_table('Volume', ['Year', 'Month'], 'Ticker', 'mean').round(0).fillna(0).astype(int)
pivot_volume_month.to_csv("olap_results/03_monthly_volume.csv")

df_2025 = df[df['Year'] == 2025]
if not df_2025.empty:
    stats_2025 = df_2025.groupby('Ticker').agg(
        Gia_TB=('Close', 'mean'),
        Volume_TB=('Volume', 'mean'),
        Bien_Dong_Std=('Close', 'std')
    )
    stats_2025['Bien_Dong_Pct (%)'] = (stats_2025['Bien_Dong_Std'] / stats_2025['Gia_TB']) * 100
    stats_2025 = stats_2025.sort_values('Volume_TB', ascending=False)
    stats_2025.to_csv("olap_results/04_stats_2025.csv")
else:
    stats_2025 = pd.DataFrame()

# --- HÀM VẼ BIỂU ĐỒ CÁ NHÂN ---
def plot_individual_ticker(ticker, data_global):
    data = data_global[data_global['Ticker'] == ticker].copy()
    path = f"olap_charts/{ticker}"
    os.makedirs(path, exist_ok=True)

    with plot_lock:
        # 1. Giá theo thời gian
        plt.figure(figsize=(12, 6))
        plt.plot(data['Date'], data['Close'], color='#1f77b4', linewidth=2.2)
        plt.title(f'[{ticker}] Diễn biến giá đóng cửa', fontsize=16, weight='bold', pad=20)
        plt.xlabel('Ngày'); plt.ylabel('Giá đóng cửa (VND)')
        plt.tight_layout()
        plt.savefig(f"{path}/01_price_trend.png", bbox_inches='tight')
        plt.close()

        # 2. Giá theo quý
        q = data.groupby(['Year', 'Quarter'])['Close'].mean()
        labels = [f"{y}-Q{q}" for y, q in q.index]
        plt.figure(figsize=(10, 5))
        bars = plt.bar(labels, q.values, color='#1f77b4', alpha=0.85, edgecolor='navy')
        plt.title(f'[{ticker}] Giá trung bình theo Quý', fontsize=16, weight='bold', pad=20)
        plt.xlabel('Quý'); plt.ylabel('Giá TB (VND)')
        plt.xticks(rotation=45)
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, h, f'{h:,.0f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{path}/02_quarterly_price_bar.png", bbox_inches='tight')
        plt.close()

        # 3. Khối lượng theo tháng
        m = data.groupby(['Year', 'Month'])['Volume'].mean() / 1e6
        labels_m = [f"{y}-{m:02d}" for y, m in m.index]
        plt.figure(figsize=(11, 5))
        bars = plt.bar(labels_m, m.values, color='#ff7f0e', alpha=0.85, edgecolor='darkred')
        plt.title(f'[{ticker}] Khối lượng giao dịch trung bình theo Tháng', fontsize=16, weight='bold', pad=20)
        plt.xlabel('Tháng'); plt.ylabel('Triệu cổ phiếu')
        plt.xticks(rotation=45)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                label = f'{h:.3f}' if h < 1 else f'{h:.2f}'
                plt.text(bar.get_x() + bar.get_width()/2, h, label, ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{path}/03_monthly_volume_bar.png", bbox_inches='tight')
        plt.close()

        # 4. Biến động theo quý
        v = data.groupby(['Year', 'Quarter'])['Close'].std()
        labels_v = [f"{y}-Q{q}" for y, q in v.index]
        plt.figure(figsize=(10, 5))
        plt.plot(labels_v, v.values, 'o-', color='#9467bd', linewidth=2.2, markersize=6)
        plt.fill_between(labels_v, v.values, alpha=0.2, color='#9467bd')
        plt.title(f'[{ticker}] Biến động giá theo Quý', fontsize=16, weight='bold', pad=20)
        plt.xlabel('Quý'); plt.ylabel('Độ lệch chuẩn (VND)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{path}/04_quarterly_volatility.png", bbox_inches='tight')
        plt.close()

        # 5. Dual Axis
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(data['Date'], data['Close'], color='#1f77b4', linewidth=2.2)
        ax1.set_ylabel('Giá đóng cửa (VND)', color='#1f77b4')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        ax2 = ax1.twinx()
        ax2.bar(data['Date'], data['Volume'], alpha=0.25, color='gray', width=1)
        ax2.set_ylabel('Khối lượng (cổ phiếu)', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        plt.title(f'[{ticker}] Giá & Khối lượng (Dual Axis)', fontsize=16, weight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{path}/05_price_volume_dual.png", bbox_inches='tight')
        plt.close()

    return f"Hoàn thành: {ticker}"

# --- CHẠY ĐA LUỒNG ---
print("Đang vẽ biểu đồ cá nhân (multi-threading)...")
with ThreadPoolExecutor(max_workers=min(6, len(tickers))) as executor:
    futures = [executor.submit(plot_individual_ticker, ticker, df) for ticker in tickers]
    for future in as_completed(futures):
        print(future.result())

# --- BIỂU ĐỒ CHUNG ---
monthly_pivot = df.pivot_table('Close', 'YearMonth', 'Ticker', 'mean')[tickers]
plt.figure(figsize=(14, 8))
for t in tickers:
    plt.plot(monthly_pivot.index, monthly_pivot[t], label=t, color=COLORS[t], linewidth=2.5, alpha=0.9)
plt.title('Diễn biến giá trung bình hàng tháng (từ 2024)', fontsize=16, weight='bold', pad=20)
plt.xlabel('Thời gian'); plt.ylabel('Giá đóng cửa (VND)')
plt.legend(title='Mã Cổ phiếu', loc='upper left', bbox_to_anchor=(1.02, 1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("olap_charts/01_combined_price_trend.png", bbox_inches='tight')
plt.close()

if not df_2025.empty:
    vol = stats_2025['Volume_TB'].sort_values(ascending=False) / 1e6
    plt.figure(figsize=(10, 6))
    bars = plt.bar(vol.index, vol.values, color=[COLORS[t] for t in vol.index], alpha=0.9, edgecolor='black')
    plt.title('Khối lượng giao dịch trung bình 2025', fontsize=16, weight='bold', pad=20)
    plt.xlabel('Mã cổ phiếu'); plt.ylabel('Triệu cổ phiếu')
    for bar in bars:
        h = bar.get_height()
        label = f'{h:.2f}' if h < 1 else f'{h:.1f}'
        plt.text(bar.get_x() + bar.get_width()/2, h, label, ha='center', va='bottom', fontsize=10, weight='bold')
    plt.tight_layout()
    plt.savefig("olap_charts/02_combined_volume_2025.png", bbox_inches='tight')
    plt.close()

plt.figure(figsize=(10, 7))
sns.heatmap(pivot_quarter, annot=True, fmt='.1f', cmap='RdYlGn',
            cbar_kws={'label': 'Giá (VND)'}, linewidths=0.5)
plt.title('Heatmap: Giá đóng cửa trung bình theo Quý', fontsize=16, weight='bold', pad=20)
plt.xlabel('Mã cổ phiếu'); plt.ylabel('Năm - Quý')
plt.tight_layout()
plt.savefig("olap_charts/03_combined_heatmap_price.png", bbox_inches='tight')
plt.close()

print(f"\nHOÀN TẤT! {len(tickers)*5} biểu đồ cá nhân + 3 biểu đồ chung.")
print("Tất cả lưu tại: olap_results/, olap_charts/")
