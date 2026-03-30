import procrastinate
from app.core.config import settings

# Infrastructure Layer: Doar conexiunea, fără import_paths!
dsn = settings.DATABASE_URL.replace("+asyncpg", "")

task_app = procrastinate.App(
    connector=procrastinate.PsycopgConnector(conninfo=dsn)
)