from celery import Celery
import os

# Configure Celery
celery = Celery(
    "tasks",
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)

# Configure task routes
celery.conf.task_routes = {
    "src.tasks.classify_file_task": {"queue": "classify_file"},
    "src.tasks.classify_batch_task": {"queue": "classify_batch"}
}

# Configure task settings
celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes
    worker_max_tasks_per_child=100,
    worker_prefetch_multiplier=1
)