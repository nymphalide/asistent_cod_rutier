import asyncio
import json
from app.schemas.law_unit import LawUnitCreate
from app.core.custom_types import UnitType
from app.pipeline.ingestion.enricher import EnricherService


async def run_test():
    print("🚦 --- Starting GPU Enricher Test --- 🚦\n")

    # 1. Initialize the Service
    try:
        enricher = EnricherService()
        print("[OK] EnricherService initialized successfully.\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Enricher. Check config. Error: {e}")
        return

    # 2. Create Dummy Data (Simulating the output from your Parser)
    # We test a CHAPTER (should skip QuOTE) and an ARTICLE (should trigger QuOTE)
    chapter_unit = LawUnitCreate(
        id="cap_2",
        content="CAPITOLUL II: Vehiculele. Secțiunea 1: Condițiile privind circulația vehiculelor.",
        unit_type=UnitType.CHAPTER,
        meta_info={"source": "test_script"}
    )

    article_unit = LawUnitCreate(
        id="art_14",
        content="Tramvaiele, tractoarele agricole sau forestiere, remorcile destinate a fi tractate de acestea, precum și troleibuzele se înregistrează la nivelul primăriilor comunelor, ale orașelor, ale municipiilor, ale sectoarelor municipiului București.",
        unit_type=UnitType.ARTICLE,
        meta_info={"source": "test_script"}
    )

    units_to_test = [chapter_unit, article_unit]

    # 3. Run the Enricher
    print(f"🧠 Sending {len(units_to_test)} units to local Ollama models (Watch your GPU VRAM!)...\n")

    try:
        enriched_units = await enricher.enrich_batch(units_to_test)
    except Exception as e:
        print(f"[ERROR] Batch enrichment failed: {e}")
        return

    # 4. Display Results Visually
    print("\n✅ --- Enrichment Complete. Results: --- ✅\n")
    for eu in enriched_units:
        print(f"=== ID: {eu.id} | Type: {eu.unit_type.name} ===")
        print(f"Text: {eu.content}\n")

        # Print generated questions
        print("Generated Questions:")
        if eu.hypothetical_questions:
            print(json.dumps(eu.hypothetical_questions, indent=2, ensure_ascii=False))
        else:
            print("  [None - Skipped as expected for this unit type]")

        # Verify vector dimensions
        c_vec_len = len(eu.content_vector) if eu.content_vector else 0
        q_vec_count = len(eu.question_vectors) if eu.question_vectors else 0

        print(f"\nDiagnostics:")
        print(f" - Content Vector Dimension: {c_vec_len} (Should be 768 for nomic-embed-text)")
        print(f" - Number of Question Vectors: {q_vec_count} (Should match number of questions)")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    # Windows/WSL2 specific fix for asyncio loop policies if needed
    import sys

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_test())