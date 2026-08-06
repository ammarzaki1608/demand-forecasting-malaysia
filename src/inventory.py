"""Shared inventory-optimization logic for notebooks and the dashboard."""

import numpy as np
from scipy.stats import norm


def service_level_z_score(service_level):
    """Z-score for a target service level, e.g. 0.95 -> ~1.645.

    Uses the inverse CDF (percent point function). norm.pdf() is the
    density at a point and is NOT the right function here -- it was a
    bug in an earlier version of this project that understated safety
    stock by ~6x. See tests/test_inventory.py.
    """
    return norm.ppf(service_level)


def compute_safety_stock(std_demand, lead_time, service_level):
    z = service_level_z_score(service_level)
    return z * std_demand * np.sqrt(lead_time)


def compute_reorder_point(avg_demand, lead_time, safety_stock):
    return (avg_demand * lead_time) + safety_stock


def simulate_inventory(demand_series, reorder_point, order_quantity, lead_time, initial_stock):
    """Simulate daily inventory levels under a reorder-point/order-quantity policy."""
    n_days = len(demand_series)
    inventory = np.zeros(n_days)
    stockouts = np.zeros(n_days)
    orders_placed = []
    pending_orders = []  # (delivery_day, quantity)

    inventory[0] = initial_stock

    for day in range(1, n_days):
        received = sum(qty for del_day, qty in pending_orders if del_day == day)

        inventory[day] = inventory[day - 1] + received - demand_series.iloc[day - 1]

        if inventory[day] < 0:
            stockouts[day] = abs(inventory[day])
            inventory[day] = 0

        if inventory[day] <= reorder_point:
            pending_orders.append((day + lead_time, order_quantity))
            orders_placed.append(day)

        pending_orders = [(d, q) for d, q in pending_orders if d > day]

    return inventory, stockouts, orders_placed


def compute_costs(avg_inventory, total_stockout_units, num_orders, days,
                   holding_cost_per_unit_per_day=0.50, stockout_cost_per_unit=15.00,
                   order_fixed_cost=50.00):
    holding = avg_inventory * holding_cost_per_unit_per_day * days
    stockout = total_stockout_units * stockout_cost_per_unit
    ordering = num_orders * order_fixed_cost
    return {
        'holding_cost': holding,
        'stockout_cost': stockout,
        'ordering_cost': ordering,
        'total_cost': holding + stockout + ordering,
    }
