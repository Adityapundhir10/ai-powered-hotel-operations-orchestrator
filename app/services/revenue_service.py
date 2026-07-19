from datetime import date
from app.config import settings
from app.ml.occupancy_forecaster import OccupancyForecaster
from app.ml.pricing_engine import DynamicPricingEngine
from app.schemas import ForecastRequest, ForecastResponse


class RevenueService:
    def __init__(self):
        self.forecaster = OccupancyForecaster(settings.resolve(settings.forecast_model_path))
        self.pricing = DynamicPricingEngine()

    def forecast(self, request: ForecastRequest) -> ForecastResponse:
        feature_row = {
            "forecast_date": request.forecast_date.isoformat(),
            "lead_time": request.lead_time,
            "booking_pickup_7d": request.booking_pickup_7d,
            "rolling_demand_14d": request.rolling_demand_14d,
            "cancellation_rate_28d": request.cancellation_rate_28d,
            "available_inventory": request.available_inventory,
            "event_intensity": request.event_intensity,
            "lag_occupancy_7d": request.lag_occupancy_7d,
            "lag_occupancy_28d": request.lag_occupancy_28d,
        }
        predicted = self.forecaster.predict(feature_row)
        decision = self.pricing.recommend(
            base_adr=request.base_adr,
            predicted_occupancy=predicted,
            forecast_date=request.forecast_date,
            event_intensity=request.event_intensity,
            lead_time=request.lead_time,
            cancellation_rate=request.cancellation_rate_28d,
        )
        return ForecastResponse(
            hotel_id=request.hotel_id,
            forecast_date=request.forecast_date,
            predicted_occupancy=round(predicted, 4),
            recommended_adr=decision.recommended_adr,
            estimated_revpar=decision.estimated_revpar,
            demand_multiplier=decision.demand_multiplier,
            model_backend=self.forecaster.backend,
            feature_contributions=self.forecaster.feature_contributions(feature_row),
        )


revenue_service = RevenueService()
