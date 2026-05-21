import json
import logging
import os
import time
import socket

import redis

from app.common.redis_client import redis_client
from app.common.config import (
    TASK_STREAM_NAME,
    TASK_GROUP_NAME,
    RESULT_TTL_SECONDS,
    TASK_WAIT_TIMEOUT_MS,
)
from app.worker.inference import run_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_consumer_group():
    try:
        redis_client.xgroup_create(
            name=TASK_STREAM_NAME,
            groupname=TASK_GROUP_NAME,
            id="0",
            mkstream=True
        )
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.info("Redis consumer group already exists")
        else:
            raise

def main():
    consumer_name = os.getenv("WORKER_NAME", f"worker-{socket.gethostname()}")

    create_consumer_group()

    logger.info(f"Worker started: {consumer_name}")

    while True:
        try:
            messages = redis_client.xreadgroup(
                groupname=TASK_GROUP_NAME,
                consumername=consumer_name,
                streams={TASK_STREAM_NAME: ">"},
                count=1,
                block=TASK_WAIT_TIMEOUT_MS,
            )

            if not messages:
                continue

            for stream_name, stream_messages in messages:
                for message_id, fields in stream_messages:
                    task_id = fields["task_id"]
                    tensor = json.loads(fields["tensor"])

                    logger.info(
                        f"Worker {consumer_name} received task: {task_id}, "
                        f"message_id={message_id}"
                    )

                    try:
                        result = run_inference(tensor)

                        result_key = f"recognition_result:{task_id}"

                        redis_client.setex(
                            result_key,
                            RESULT_TTL_SECONDS,
                            json.dumps(result)
                        )

                        redis_client.xack(
                            TASK_STREAM_NAME,
                            TASK_GROUP_NAME,
                            message_id
                        )

                        logger.info(f"Worker {consumer_name} completed task: {task_id}")
                    except Exception as e:
                        logger.exception(f"Worker {consumer_name} failed task: {task_id}, error: {e}"
                        )

                        error_result = {
                            "status": False,
                            "message": str(e)
                        }

                        redis_client.setex(
                            f"recognition_result:{task_id}",
                            RESULT_TTL_SECONDS,
                            json.dumps(error_result)
                        )

                        redis_client.xack(
                            TASK_STREAM_NAME,
                            TASK_GROUP_NAME,
                            message_id
                        )
        except redis.exceptions.RedisError as e:
            logger.exception(f"Worker {consumer_name} loop error:-- {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()