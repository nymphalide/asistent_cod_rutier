import typer
from src.cli import ingest, wipe, system, search

app = typer.Typer(
    name="RAG Dev Tool",
    help="Modular Developer CLI for managing the backend.",
    no_args_is_help=True
)

# Bind the submodules to the main application
app.add_typer(ingest.app, name="ingest")
app.add_typer(wipe.app, name="wipe")
app.add_typer(system.app, name="system")
app.add_typer(search.app, name="search")

def run():
    """Entry point for the pyproject.toml executable."""
    app()

if __name__ == "__main__":
    run()