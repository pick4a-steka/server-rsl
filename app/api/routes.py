import json
import uuid
import time
import logging

from fastapi import APIRouter, HTTPException

from app.worker.inference import run_inference
from app.common.redis_client import redis_client
from app.common.config import (
    Request, Response,
    TASK_STREAM_NAME, TASK_GROUP_NAME,
    RESULT_TTL_SECONDS
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["inference"]
)

@router.get("/health")
def health_check():
    return { "status": "ok" }

@router.post("/recognize")
async def recognize(item: Request) -> Response:
    task_id = str(uuid.uuid4())

    # отладка: сохраняем пришедший большой тензор в лог, чтобы он полностью отображался
    logger.debug(f"Received tensor for task {task_id}: {json.dumps(item.tensor)})")

    try:
        message_id = redis_client.xadd(
            TASK_STREAM_NAME,
            {
                "task_id": task_id,
                "tensor": json.dumps(item.tensor),
            },
            maxlen=1000,
            approximate=True
        )
    except redis.exceptions.RedisError:
        raise HTTPException(
            status_code=503,
            detail="Redis service is unavailable"
        )

    logger.info(f"Task {task_id} added to stream with message ID {message_id}")
    result_key = f"recognition_result:{task_id}"

    timeout_sec = 10
    start_time = time.time()

    while time.time() - start_time < timeout_sec:
        result_json = redis_client.get(result_key)

        if result_json is not None:
            redis_client.delete(result_key)
            return json.loads(result_json)
        time.sleep(0.05)

    raise HTTPException(status_code=504, detail="Recognition timedout")