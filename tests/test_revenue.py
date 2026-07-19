from datetime import date
from app.schemas import ForecastRequest
from app.services.revenue_service import RevenueService


def test_forecast_and_pricing():
    service = RevenueService()
    result = service.forecast(ForecastRequest(
        hotel_id="DEL-001", forecast_date=date(2026, 8, 15), lead_time=14,
        booking_pickup_7d=31, rolling_demand_14d=0.73,
        cancellation_rate_28d=0.11, available_inventory=26,
        base_adr=3200, event_intensity=0.7,
    ))
    assert 0.03 <= result.predicted_occupancy <= 0.99
    assert result.recommended_adr > 0
    assert result.estimated_revpar > 0
