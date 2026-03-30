import sys
import os

# Adaugă folderul curent în path
sys.path.append(os.getcwd())

try:
    print("🔍 Testăm importul pentru task_app...")
    from app.db.queue import task_app

    print("✅ task_app a fost importat.")

    print("🔍 Testăm importul pentru tasks...")
    import app.pipeline.ingestion.tasks

    print("✅ Toate task-urile au fost importate cu succes.")

except Exception:
    import traceback

    print("\n❌ EROARE GĂSITĂ:\n")
    traceback.print_exc()