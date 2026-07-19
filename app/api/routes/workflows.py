from fastapi import APIRouter, HTTPException
from app.schemas import ComplaintRequest, WorkflowResult
from app.workflows.hotel_ops_graph import workflow_engine
from app.infrastructure.state_store import state_store

router = APIRouter()


@router.post("/complaints", response_model=WorkflowResult)
async def process_complaint(request: ComplaintRequest):
    return await workflow_engine.run(request)


@router.get("/complaints/{complaint_id}")
def get_workflow_state(complaint_id: str):
    state = state_store.get(f"complaint:{complaint_id}")
    if state is None:
        raise HTTPException(status_code=404, detail="Workflow state not found")
    return state
