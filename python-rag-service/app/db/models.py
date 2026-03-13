from typing import Optional, List, Any, Dict
from sqlalchemy import String, Text, ForeignKey, Computed
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB, ENUM

from app.db.session import Base
# We will define the Enum in schemas next, but we reference it here for the DB type.
# Ensure your schema file has the UnitType enum defined as verified in the next step.
from app.core.custom_types import UnitType
from app.core.config import settings


class LawUnit(Base):
    """
    SQLAlchemy model representing the 'law_units' table.
    Stores the graph nodes (Articles, Paragraphs) and their vector embeddings.
    """
    __tablename__ = "law_units"

    # --- Primary Keys & Identifiers ---
    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)

    # --- Hierarchy & Graph Relationships ---
    # Structural Parent (e.g., Art 5 -> Chapter II)
    parent_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("law_units.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )

    # --- Content ---
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # We use the Python Enum 'UnitType' to strictly type this column in Postgres
    unit_type: Mapped[UnitType] = mapped_column(
        # FIX: Ensure the enum inherits schema metadata so Alembic can track it
        ENUM(UnitType, name="unit_type_enum", create_type=True, metadata=Base.metadata),
        nullable=False
    )

    # --- Metadata ---
    # Stores flexible data like {"chapter_title": "...", "breadcrumbs": [...]}
    # We map it to the SQL column 'meta_info' (requires a migration!)
    meta_info: Mapped[Dict[str, Any]] = mapped_column("meta_info", JSONB, server_default='{}')


    # --- ORM Relationships (Optional helper for Python navigation) ---
    # Allows accessing children via parent.children
    children: Mapped[List["LawUnit"]] = relationship(
        "LawUnit",
        back_populates="parent",
        foreign_keys=[parent_id],
        cascade="all, delete-orphan"
    )

    parent: Mapped[Optional["LawUnit"]] = relationship(
        "LawUnit",
        remote_side=[id],
        foreign_keys=[parent_id],
        back_populates="children"
    )

    def __repr__(self):
        return f"<LawUnit(id='{self.id}', type='{self.unit_type}', parent='{self.parent_id}')>"