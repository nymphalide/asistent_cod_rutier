import procrastinate # type: ignore
from src.app.core.config import settings

# Infrastructure Layer: Set up the DSN
dsn = settings.DATABASE_URL.replace("+psycopg", "")

# Registry Pattern: App initialization is isolated here.
# Procrastinate lazily loads the paths to prevent circular logic.
task_app = procrastinate.App(
    connector=procrastinate.PsycopgConnector(conninfo=dsn),
    import_paths=["src.app.pipeline.ingestion.tasks"]
)