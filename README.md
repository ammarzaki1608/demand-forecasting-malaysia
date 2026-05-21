# E-Commerce Demand Forecasting & Inventory Optimization

Demamd forecating and inventory optimizatiopn system for e-commerce using machine learning to predict product demand and  recommend optimal stock levels.

## tech Stack 
- Python (pandas, scikit-learn, XGBoost)
- Streamlit (interactive dashboard)
- SQL-style data manipulation

## Data Source
Dataset: [Brazilian E-Commerce by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
Download and place all CSV files into `data/raw/`

## How to Reproduce
1. Download the dataset from Kaggle (link above) into `data/raw/`
2. Run `notebooks/01_eda.ipynb` to generate `data/processed/forecast_ready.csv`
3. Run `notebooks/02_forecasting.ipynb` for model training and evaluation

## Project Status
Phase 1: Data Engineering & EDA (in progress)

## Author
Ammar 