import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SRC_DIR)

from data import daily_demand_for_category
from forecasting import create_features, FEATURE_COLS, train_xgboost
from inventory import compute_safety_stock, compute_reorder_point

st.set_page_config(
    page_title="E-Commerce Demand Forecasting",
    page_icon="📦",
    layout="wide"
)

@st.cache_data
def load_data():
    csv_path = os.path.join(SRC_DIR, '..', 'data', 'processed', 'forecast_ready.csv')
    df = pd.read_csv(csv_path)
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    return df

df = load_data()

st.sidebar.title("📦 Dashboard Controls")
st.sidebar.markdown("---")

# Category selector
top_categories = (df['product_category_name_english']
                  .value_counts()
                  .head(10)
                  .index.tolist())

selected_category = st.sidebar.selectbox(
    "Select Product Category",
    options=top_categories
)

# Inventory parameters
st.sidebar.markdown("### Inventory Parameters")
lead_time = st.sidebar.slider("Lead Time (days)", min_value=1, max_value=14, value=3)
service_level = st.sidebar.slider("Target Service Level", 
                                   min_value=0.80, max_value=0.99, 
                                   value=0.95, step=0.01,
                                   format="%.0f%%")

st.title("📦 E-Commerce Demand Forecasting & Inventory Optimization")
st.markdown("Demand forecasting and inventory optimization using real-world e-commerce transaction data")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📈 Demand Overview", "🔮 Forecast", "📦 Inventory Policy"])

with tab1:
    st.subheader(f"Demand Overview — {selected_category}")
    
    # Daily demand for selected category
    cat_data = daily_demand_for_category(df, selected_category)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Orders", f"{cat_data['order_count'].sum():,}")
    with col2:
        st.metric("Avg Daily Orders", f"{cat_data['order_count'].mean():.1f}")
    with col3:
        st.metric("Peak Day Orders", f"{cat_data['order_count'].max():,}")
    
    # Demand trend chart
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(cat_data['order_date'], cat_data['order_count'], 
            alpha=0.5, color='steelblue', linewidth=0.8)
    rolling = cat_data['order_count'].rolling(30).mean()
    ax.plot(cat_data['order_date'], rolling, 
            color='darkred', linewidth=2, label='30-day average')
    ax.set_title(f'Daily Order Volume — {selected_category}')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab2:
    st.subheader(f"Demand Forecast — {selected_category}")

    @st.cache_data
    def run_forecast(category):
        cat = daily_demand_for_category(df, category)
        cat_featured = create_features(cat).dropna()

        split = cat_featured['order_date'].max() - pd.Timedelta(days=60)
        train = cat_featured[cat_featured['order_date'] <= split]
        test = cat_featured[cat_featured['order_date'] > split]

        _, forecast = train_xgboost(train, test, FEATURE_COLS)

        test = test.copy()
        test['forecast'] = forecast
        mae = mean_absolute_error(test['order_count'], test['forecast'])

        return test, mae

    with st.spinner("Training forecast model..."):
        forecast_df, mae = run_forecast(selected_category)

    st.metric("Model MAE", f"{mae:.2f} orders/day")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(forecast_df['order_date'], forecast_df['order_count'],
            label='Actual', color='steelblue', linewidth=1.5)
    ax.plot(forecast_df['order_date'], forecast_df['forecast'],
            label='Forecast', color='coral', linestyle='--', linewidth=1.5)
    ax.set_title(f'Forecast vs Actual — Last 60 Days')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab3:
    st.subheader(f"Inventory Policy — {selected_category}")

    cat_inv = daily_demand_for_category(df, selected_category)

    avg_demand = cat_inv['order_count'].mean()
    std_demand = cat_inv['order_count'].std()

    safety_stock = compute_safety_stock(std_demand, lead_time, service_level)
    reorder_point = compute_reorder_point(avg_demand, lead_time, safety_stock)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Safety Stock", f"{safety_stock:.0f} units")
    with col2:
        st.metric("Reorder Point", f"{reorder_point:.0f} units")
    with col3:
        st.metric("Avg Daily Demand", f"{avg_demand:.1f} orders")

    st.markdown("### Cost Comparison")
    
    # Quick cost estimate (holding cost only; see notebooks/03 for the
    # full simulation-based cost comparison including stockout/ordering cost)
    holding_cost = 0.50

    optimized_inv = reorder_point
    naive_inv = avg_demand * 30
    
    optimized_cost = optimized_inv * holding_cost * 180
    naive_cost = naive_inv * holding_cost * 180
    savings = naive_cost - optimized_cost

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Optimized Policy Cost", f"RM {optimized_cost:,.0f}")
    with col2:
        st.metric("Naive Policy Cost", f"RM {naive_cost:,.0f}")
    with col3:
        st.metric("Estimated Savings", f"RM {savings:,.0f}", 
                  delta=f"{savings/naive_cost:.0%} reduction")

    st.info(f"""
    **Recommendation for {selected_category}:**
    - Place a new order when stock drops to **{reorder_point:.0f} units**
    - Maintain **{safety_stock:.0f} units** of safety stock
    - This policy targets **{service_level:.0%}** service level with **{lead_time}-day** lead time
    """)

