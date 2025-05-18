import os
from dotenv import load_dotenv
import redis
from rq import Worker, Queue
from rq.worker import SpawnWorker

load_dotenv()

env = os.getenv("PYTHON_ENV")

redis_url = os.getenv("REDIS_URL")
listen = ["high", "default", "low"]

if env == "production":
    conn = redis.from_url(redis_url, ssl_cert_reqs=None)
else:
    conn = redis.from_url(redis_url)

if __name__ == "__main__":
    queues = [Queue(name, connection=conn) for name in listen]
    if env == "production":
        worker = Worker(queues, connection=conn)
    else:
        worker = SpawnWorker(queues, connection=conn)
    worker.work()
