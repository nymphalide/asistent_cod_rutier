from typing import List, Optional, Sequence
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from src.app.db.models import LawUnit
from src.app.schemas.law_unit import LawUnitCreate

class LawUnitRepository:
    """
    Repository for handling LawUnit (Traffic Code) database operations.
    Acts as the 'Librarian', abstracting SQL away from the business logic.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, unit_in: LawUnitCreate) -> LawUnit:
        """
        Creates a single LawUnit record by converting the Pydantic
        schema into a SQLAlchemy model instance.
        """
        values = unit_in.model_dump(by_alias=False)
        db_obj = LawUnit(**values)

        self.db.add(db_obj)
        # We use flush() instead of commit() to respect the Unit of Work pattern.
        # This pushes the query to the DB to get generated fields without finalizing the transaction.
        await self.db.flush()
        await self.db.refresh(db_obj)
        return db_obj

    async def get(self, unit_id: str) -> Optional[LawUnit]:
        """
        Retrieves a LawUnit by its ID using async execute.
        """
        stmt = select(LawUnit).where(LawUnit.id == unit_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_children(self, parent_id: str) -> Sequence[LawUnit]:
        """
        Retrieves all direct children of a specific unit using async execute.
        """
        stmt = select(LawUnit).where(LawUnit.parent_id == parent_id)
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def bulk_upsert(self, units_in: List[LawUnitCreate]) -> int:
        if not units_in:
            return 0

        #extracts only the lawcreateunitcreate fields from the enriched unit for the postgres upsert
        postgres_fields = LawUnitCreate.model_fields.keys()

        values = [
            obj.model_dump(by_alias=False, include=postgres_fields)
            for obj in units_in
        ]

        stmt = insert(LawUnit).values(values)

        # Grab every column from the schema EXCEPT id to update on conflict
        update_dict = {
            col.name: col
            for col in stmt.excluded
            if col.name != "id"
        }

        stmt = stmt.on_conflict_do_update(
            index_elements=[LawUnit.id],
            set_=update_dict
        )

        await self.db.execute(stmt)
        return len(values)