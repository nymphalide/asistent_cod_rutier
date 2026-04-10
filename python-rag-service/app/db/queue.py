import procrastinate
from app.core.config import settings

# Infrastructure Layer: Set up the DSN
dsn = settings.DATABASE_URL.replace("+asyncpg", "")

# SOTA Setup: We tell the App which modules contain our tasks.
# Procrastinate is smart—it won't load these until the worker starts,
# which prevents the circular import 'terci' we were worried about.
task_app = procrastinate.App(
    connector=procrastinate.PsycopgConnector(conninfo=dsn),
    import_paths=["app.pipeline.ingestion.tasks"]
)