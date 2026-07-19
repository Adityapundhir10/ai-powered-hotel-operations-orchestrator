from __future__ import annotations
from dataclasses import dataclass
from datetime import date
import math


@dataclass(frozen=True)
class PricingDecision:
    recommended_adr: float
    estimated_revpar: float
    demand_multiplier: float


class DynamicPricingEngine:
    """Converts occupancy demand into bounded ADR and RevPAR recommendations."""

    def __init__(self, floor_multiplier: float = 0.72, ceiling_multiplier: float = 1.65):
        self.floor_multiplier = floor_multiplier
        self.ceiling_multiplier = ceiling_multiplier

    def recommend(
        self,
        *,
        base_adr: float,
        predicted_occupancy: float,
        forecast_date: date,
        event_intensity: float,
        lead_time: int,
        cancellation_rate: float,
    ) -> PricingDecision:
        occupancy_pressure = 0.72 + 0.95 * predicted_occupancy
        weekend = 1.06 if forecast_date.weekday() >= 5 else 1.0
        event = 1.0 + 0.22 * event_intensity
        urgency = 1.0 + max(0, 14 - lead_time) * 0.008
        cancellation_penalty = 1.0 - 0.12 * cancellation_rate
        raw_multiplier = occupancy_pressure * weekend * event * urgency * cancellation_penalty
        multiplier = min(self.ceiling_multiplier, max(self.floor_multiplier, raw_multiplier))
        recommended_adr = round(base_adr * multiplier, 2)
        revpar = round(recommended_adr * predicted_occupancy, 2)
        return PricingDecision(recommended_adr, revpar, round(multiplier, 4))
