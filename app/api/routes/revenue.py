from fastapi import APIRouter
from app.schemas import ForecastRequest, ForecastResponse
from app.services.revenue_service import revenue_service

router = APIRouter()


@router.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    return revenue_service.forecast(request)
