from fastapi import APIRouter

from app.worker.inference import run_inference
from app.api.config import Request, Response

router = APIRouter(
    prefix="/api",
    tags=["inference"]
)

@router.post("/recognize")
async def recognize(item: Request) -> Response:
    result = run_inference(item)
    return result