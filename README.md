# Stock Data Warehouse & Analytics

This project implements a mini Data Warehouse for Vietnamese airline stocks (HVN, VJC, AST, NCT, SCS, VTR).  
The system includes ETL preprocessing, OLAP-style multidimensional analysis, and Data Mining techniques implemented using Python.

## Features

### Data Warehouse Processing (ETL)
- Clean and normalize raw stock CSV files.
- Merge all tickers into a unified dataset (`All.csv`).
- Add time-based dimensions (Year, Month, Quarter) for OLAP analysis.

### OLAP Analysis
- Pivot tables for multi-dimension analytics:
  - Average closing price by **Quarter × Ticker**
  - Average trading volume by **Month × Ticker**
  - Comparative analysis between airlines (e.g., HVN vs VJC)

### Data Mining
- **Linear Regression** for predicting stock closing prices.
- **K-Means clustering** to group stocks by return and volume patterns.
- **Z-Score detection** for identifying abnormal trading days.

### Visualization
- Time-series charts for stock prices.
- Correlation heatmaps.
- Volume comparison bar charts.
