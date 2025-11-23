import os
import re
import glob
import pandas as pd
import numpy as np
from dateutil import parser
from sqlalchemy import create_engine

# --- Cấu hình ---
INPUT_FOLDER = "./stock_data"      # thư mục chứa file CSV gốc
OUTPUT_FOLDER = "./cleaned"     # thư mục chứa file đã xử lý
TICKERS = ["HVN", "AST", "NCT", "SCS", "VJC", "VTR"] 
# SAVE_TO_DB = False              
# DB_CONN_STR = "mysql+pymysql://user:password@localhost:3306/yourdb"  

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Hàm hỗ trợ ---
def parse_volume(v):
    """
    Chuyển từ dạng '900K' hoặc '1.5M' thành số (int).
    Xử lý '—' hoặc NaN trả về None.
    """
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s in ["", "-", "—", "nan"]:
        return None
    s = s.replace(",", "").replace(" ", "")
    m = re.match(r"^(-?[\d\.]+)([KkMmBb]?)$", s)
    if m:
        num = float(m.group(1))
        suf = m.group(2).upper()
        if suf == "K":
            return int(num * 1_000)
        elif suf == "M":
            return int(num * 1_000_000)
        elif suf == "B":
            return int(num * 1_000_000_000)
        else:
            return int(num)
    try:
        return int(float(s))
    except:
        return None

# Hàm chuyển đổi phần trăm sang số thực
def parse_percent(s):
    if pd.isna(s):
        return None
    s = str(s).strip().replace("%", "").replace(",", "")
    if s in ["", "-", "—", "nan"]:
        return None
    try:
        return float(s)
    except:
        return None

# Hàm chuyển đổi số sang số thực (xử lý dấu phẩy, dấu cách, ký tự lạ)
def parse_number(s):
    if pd.isna(s):
        return None
    s = str(s).strip().replace(",", "").replace(" ", "")
    if s in ["", "-", "—", "nan"]:
        return None
    try:
        return float(s)
    except:
        return None

# Hàm chuyển đổi ngày tháng (hỗ trợ nhiều định dạng)
def parse_date(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    try:
        # thử parse tự động (hỗ trợ nhiều định dạng)
        return pd.to_datetime(parser.parse(s, dayfirst=True)).date()
    except:
        try:
            return pd.to_datetime(s, dayfirst=True).date()
        except:
            return None

# Map tên cột tiếng Việt -> tiếng Anh
COL_MAP = {
    "Ngày": "Date",
    "Date": "Date",
    "Lần cuối": "Close",
    "Close": "Close",
    "Mở": "Open",
    "Cao": "High",
    "Thấp": "Low",
    "KL": "Volume",
    "Volume": "Volume",
    "% Thay đổi": "ChangePercent",
    "Change%": "ChangePercent",
    "% Change": "ChangePercent"
}

# Chuẩn hoá tên cột
def standardize_columns(df):
    # chuyển tên cột trông giống map
    new_cols = {}
    for c in df.columns:
        c_strip = c.strip()
        if c_strip in COL_MAP:
            new_cols[c] = COL_MAP[c_strip]
        else:
            # loại bỏ ký tự lạ, lowercase
            c_norm = c_strip.replace("\n", " ").strip()
            if c_norm in COL_MAP:
                new_cols[c] = COL_MAP[c_norm]
            else:
                # giữ nguyên nếu không chắc
                new_cols[c] = c_norm
    df = df.rename(columns=new_cols)
    return df

# --- Xử lý từng file ---
def clean_file(path, ticker=None):
    print("Processing:", path)
    # đọc csv, cố thử nhiều encoding/sep
    try:
        df = pd.read_csv(path, skipinitialspace=True)
    except Exception:
        df = pd.read_csv(path, encoding="latin1", skipinitialspace=True)

    df = standardize_columns(df)

    # Nếu file không có cột Ticker, thêm
    if ticker is None:
        # thử lấy ticker từ filename
        base = os.path.basename(path)
        ticker_guess = os.path.splitext(base)[0].upper()
        ticker = ticker_guess if ticker_guess in TICKERS else ticker_guess

    df['Ticker'] = ticker

    # Parse Date
    if 'Date' in df.columns:
        df['Date'] = df['Date'].apply(parse_date)
    else:
        # thử cột 'Ngày' trước đó đã rename
        df['Date'] = df.index.map(lambda i: None)

    # Số hoá các cột giá
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col].apply(parse_number)

    # Volume
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].apply(parse_volume)
    else:
        df['Volume'] = None

    # ChangePercent
    if 'ChangePercent' in df.columns:
        df['ChangePercent'] = df['ChangePercent'].apply(parse_percent)
    else:
        df['ChangePercent'] = None

    # Thêm các cột tính toán
    def compute_return(row):
        if pd.isna(row.get('Close')) or pd.isna(row.get('Open')) or row.get('Open') == 0:
            return None
        return (row['Close'] - row['Open']) / row['Open'] * 100
    # Return = (Close-Open)/Open * 100
    df['Return'] = df.apply(compute_return, axis=1)
    # Average = (High + Low)/2
    df['Average'] = df.apply(lambda r: (r['High'] + r['Low'])/2 if (pd.notna(r.get('High')) and pd.notna(r.get('Low'))) else None, axis=1)
    # Volatility = High - Low
    df['Volatility'] = df.apply(lambda r: (r['High'] - r['Low']) if (pd.notna(r.get('High')) and pd.notna(r.get('Low'))) else None, axis=1)
    # LogReturn = log(Close) - log(Open)
    df['LogReturn'] = df.apply(lambda r: None if (pd.isna(r.get('Close')) or pd.isna(r.get('Open')) or r.get('Open') == 0) else (np.log(r['Close']) - np.log(r['Open'])),axis=1)

    # sắp xếp theo Date nếu có
    if df['Date'].notna().any():
        df = df.sort_values(by='Date').reset_index(drop=True)

    # Lưu file đã clean riêng cho từng ticker
    out_path = os.path.join(OUTPUT_FOLDER, f"{ticker}.csv")
    df.to_csv(out_path, index=False)
    print("Saved cleaned file:", out_path)
    return df

# --- Chạy cho nhiều file (theo TICKERS hoặc tất cả file trong folder) ---
def process_all(input_folder=INPUT_FOLDER, tickers=TICKERS):
    all_dfs = []
    # nếu folder có file riêng cho từng ticker: HVN.csv, AST.csv ...
    for ticker in tickers:
        pattern = os.path.join(input_folder, f"*{ticker}*.csv")
        files = glob.glob(pattern)
        if not files:
            print("No file found for", ticker, "pattern:", pattern)
            continue
        # nếu có nhiều file cho 1 ticker, ghép chúng
        for f in files:
            df = clean_file(f, ticker=ticker)
            all_dfs.append(df)

    # Nếu không tìm file theo tickers, fallback: đọc tất cả csv trong folder
    if not all_dfs:
        for f in glob.glob(os.path.join(input_folder, "*.csv")):
            df = clean_file(f)
            all_dfs.append(df)

    # Gộp thành một bảng lớn
    if all_dfs:
        big = pd.concat(all_dfs, ignore_index=True, sort=False)
        # chuẩn hoá Date về datetime
        if 'Date' in big.columns:
            big['Date'] = pd.to_datetime(big['Date'])
        # lưu tổng hợp
        out_all = os.path.join(OUTPUT_FOLDER, "All.csv")
        big.to_csv(out_all, index=False)
        print("Saved merged file:", out_all)

        # Lưu vào DB nếu cần
        # if SAVE_TO_DB:
        #     engine = create_engine(DB_CONN_STR)
        #     # thay 'stock_fact' thành tên bảng bạn muốn
        #     big.to_sql('stock_fact', engine, if_exists='replace', index=False)
        #     print("Saved to DB table 'stock_fact'")

        return big
    else:
        print("No data processed.")
        return None

if __name__ == "__main__":
    df_all = process_all()
    print("Done.")
