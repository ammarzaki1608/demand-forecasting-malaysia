# 📦 E-Commerce Demand Forecasting & Inventory Optimization

A data science project that combines machine learning forecasting with 
inventory optimization to reduce supply chain costs in e-commerce operations.

---

## 📂 Dataset
**Source:** [Brazilian E-Commerce by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)  
Real-world e-commerce transaction data from Olist, a Brazilian marketplace 
platform — containing 110K orders across 2 years with product, seller, 
payment, and review information.

The demand forecasting and inventory optimization methodology demonstrated 
here is platform and geography agnostic — directly applicable to Malaysian 
e-commerce platforms such as Shopee, Lazada, and Zalora.

## 💼 Business Problem
E-commerce companies face a constant tradeoff — hold too much stock and 
capital is tied up in warehouses; hold too little and customers can't buy. 
This project builds a forecast-driven inventory system that minimizes total 
cost while maintaining target service levels.

## 📊 Key Results
- **RM 59,808 saved** (70.2%) over 180 days for a single product category
- **100% service level** achieved vs 95% target
- **71% reduction** in average inventory held (272 vs 942 units)
- Annualized savings of approximately **RM 119,600 per category**

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python, Pandas, NumPy | Data processing & feature engineering |
| XGBoost | ML-based demand forecasting |
| Scikit-learn, SciPy | Model evaluation & safety-stock statistics |
| Streamlit | Interactive dashboard |
| Matplotlib, Seaborn | Data visualization |
| Pytest | Unit tests for the inventory formulas |
| Git & GitHub | Version control |

## 📁 Project Structure

demand-forecasting-malaysia/
├── data/
│   ├── raw/           ← original data (not tracked by Git)
│   └── processed/     ← cleaned data (not tracked by Git)
├── notebooks/
│   ├── 01_eda.ipynb              ← exploratory data analysis
│   ├── 02_forecasting.ipynb      ← model building & comparison
│   └── 03_inventory_optimization.ipynb  ← inventory policy & cost analysis
├── src/
│   ├── data.py         ← shared data-loading helpers
│   ├── forecasting.py  ← shared feature engineering & model training
│   ├── inventory.py    ← shared safety stock / reorder point / simulation logic
│   └── dashboard.py    ← Streamlit interactive dashboard
├── tests/
│   └── test_inventory.py  ← regression tests for the safety-stock formula
├── output/             ← saved charts and visualizations
└── requirements.txt    ← package dependencies

Notebooks and the dashboard both import their forecasting and inventory logic
from `src/` rather than duplicating it, so a fix in one place applies
everywhere.

## 🚀 How to Reproduce
1. Clone this repository
2. Create and activate a virtual environment:
```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
```
3. Install dependencies:
```bash
   pip install -r requirements.txt
```
4. Download the dataset from [Kaggle — Brazilian E-Commerce by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
   and place all CSV files into `data/raw/`
5. Run notebooks in order (01 → 02 → 03)
6. Run the tests:
```bash
   pytest tests/
```
7. Launch the dashboard:
```bash
   streamlit run src/dashboard.py
```

## 📓 Project Walkthrough
| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Data cleaning, merging, time-series EDA across 110K orders |
| `02_forecasting.ipynb` | Baseline, XGBoost models with feature importance analysis |
| `03_inventory_optimization.ipynb` | Safety stock, reorder point, cost simulation |

## 💡 Key Insights
- **Rolling averages dominate feature importance** — recent demand history 
  is the strongest predictor of future demand
- **Model complexity doesn't always win** — naive baseline outperformed 
  XGBoost on MAE with limited data, highlighting the importance of baselines
- **Forecast-driven inventory beats fixed policies** — optimized policy 
  matched the naive policy's service level at 70% lower cost
- **Unit tests catch statistics bugs code review won't** — an earlier version
  of the safety-stock calculation used `norm.pdf()` instead of `norm.ppf()`,
  a one-character mix-up that understated safety stock by ~6x while still
  running and producing plausible-looking numbers. It's now covered by
  [`tests/test_inventory.py`](tests/test_inventory.py) and the calculation
  lives in one shared module (`src/inventory.py`) instead of being
  duplicated across notebooks and the dashboard

## 👤 Author
**Ammar Zaki** 
Skills: Python, SQL, Power BI, Machine Learning, Supply Chain Analytics