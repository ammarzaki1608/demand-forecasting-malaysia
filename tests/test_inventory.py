import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from inventory import service_level_z_score, compute_safety_stock, compute_reorder_point


@pytest.mark.parametrize('service_level,expected_z', [
    (0.90, 1.2816),
    (0.95, 1.6449),
    (0.99, 2.3263),
])
def test_service_level_z_score(service_level, expected_z):
    assert service_level_z_score(service_level) == pytest.approx(expected_z, abs=1e-3)


def test_z_score_is_not_the_density_function():
    # Regression test for a past bug: norm.pdf(0.95) ~= 0.25, which was
    # mistakenly used in place of norm.ppf(0.95) ~= 1.645.
    z = service_level_z_score(0.95)
    assert z > 1.0


def test_safety_stock_scales_with_std_and_lead_time():
    low = compute_safety_stock(std_demand=10, lead_time=3, service_level=0.95)
    high_std = compute_safety_stock(std_demand=20, lead_time=3, service_level=0.95)
    high_lead = compute_safety_stock(std_demand=10, lead_time=12, service_level=0.95)

    assert high_std == pytest.approx(2 * low)
    assert high_lead == pytest.approx(2 * low)


def test_reorder_point_combines_demand_and_safety_stock():
    safety_stock = compute_safety_stock(std_demand=10, lead_time=3, service_level=0.95)
    reorder_point = compute_reorder_point(avg_demand=13.6, lead_time=3, safety_stock=safety_stock)
    assert reorder_point == pytest.approx((13.6 * 3) + safety_stock)
