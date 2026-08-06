"""Shared demand-forecasting logic for notebooks and the dashboard."""

from xgboost import XGBRegressor

FEATURE_COLS = [
    'dayofweek', 'month', 'day', 'weekofyear', 'is_weekend',
    'is_month_start', 'is_month_end',
    'lag_7', 'lag_14', 'lag_28',
    'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30',
]


def create_features(data, date_col='order_date', target_col='order_count'):
    """Create time-based, lag, and rolling-average features for the ML model."""
    df = data.copy()
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['weekofyear'] = df[date_col].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)

    for lag in [7, 14, 28]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)

    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()

    return df


def naive_baseline(train, test, date_col='order_date', target_col='order_count', weeks=4):
    """Predict tomorrow's demand as the average of the same weekday over the last N weeks."""
    predictions = []
    for _, row in test.iterrows():
        day_of_week = row[date_col].dayofweek
        same_day = train[train[date_col].dt.dayofweek == day_of_week].tail(weeks)
        predictions.append(same_day[target_col].mean())
    return predictions


def train_xgboost(train_feat, test_feat, feature_cols=FEATURE_COLS, target_col='order_count',
                   n_estimators=200, max_depth=5, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, random_state=42):
    """Fit an XGBoost regressor and return the model plus clipped test predictions."""
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
    )
    model.fit(
        train_feat[feature_cols],
        train_feat[target_col],
        eval_set=[(test_feat[feature_cols], test_feat[target_col])],
        verbose=False,
    )
    predictions = model.predict(test_feat[feature_cols]).clip(min=0)
    return model, predictions
