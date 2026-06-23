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
- **RM 60,515 saved** over 180 days for a single product category
- **97.2% service level** achieved vs 95% target
- **73% reduction** in average inventory held (253 vs 942 units)
- Annualized savings of approximately **RM 121,000 per category**

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python, Pandas, NumPy | Data processing & feature engineering |
| XGBoost | ML-based demand forecasting |
| Scikit-learn | Model evaluation |
| Streamlit | Interactive dashboard |
| Matplotlib, Seaborn | Data visualization |
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
│   └── dashboard.py   ← Streamlit interactive dashboard
├── output/            ← saved charts and visualizations
└── requirements.txt   ← package dependencies

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
6. Launch the dashboard:
```bash
   cd src
   streamlit run dashboard.py
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
  achieved comparable service levels at 71% lower cost

## 👤 Author
**Ammar Zaki** 
Skills: Python, SQL, Power BI, Machine Learning, Supply Chain Analytics