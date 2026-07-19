import pytest
from app.schemas import ComplaintRequest
from app.workflows.hotel_ops_graph import HotelOperationsWorkflow


@pytest.mark.asyncio
async def test_workflow_routes_maintenance():
    engine = HotelOperationsWorkflow()
    result = await engine.run(ComplaintRequest(
        complaint_id="TEST-1", hotel_id="DEL-001", room_number="408",
        language="en", text="Water is leaking from the bathroom ceiling"
    ))
    assert result.assigned_team == "Engineering & Maintenance"
    assert result.sla_minutes <= 20
    assert result.retrieved_sops
    assert result.event_id
