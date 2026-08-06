"""Shared data-loading helpers for notebooks and the dashboard."""

import pandas as pd


def daily_demand_for_category(df, category, category_col='product_category_name_english',
                               date_col='order_date', order_id_col='order_id'):
    """Aggregate order counts to a complete daily series (zero-filled) for one category."""
    cat = (df[df[category_col] == category]
           .groupby(date_col)
           .agg(order_count=(order_id_col, 'nunique'))
           .reset_index())

    date_range = pd.date_range(cat[date_col].min(), cat[date_col].max(), freq='D')
    cat = (cat.set_index(date_col)
           .reindex(date_range)
           .fillna({'order_count': 0})
           .rename_axis(date_col)
           .reset_index())

    return cat
