# app/core/custom_types.py
from enum import Enum

class UnitType(str, Enum):
    CHAPTER = 'chapter'         # Top-level division (e.g., "CAPITOLUL I: DISPOZITII GENERALE")
    SECTION = 'section'         # Sub-division of a chapter (e.g., "SECŢIUNEA 1")
    ARTICLE = 'article'         # The main legal node (e.g., "Art. 5", "Art. 11.2")
    PROLOGUE = 'prologue'       # Unnumbered intro text directly under an Article, before paragraphs start (e.g., the first sentence in Art. 6)
    PARAGRAPH = 'paragraph'     # Numbered paragraphs using parentheses (e.g., "(1)", "(1.1)", "(2)")
    LETTER_ITEM = 'letter_item' # Lower-level lists using letters (e.g., "a)", "b)", "c)") inside paragraphs or annexes
    NUMBERED_ITEM = 'num_item'  # Numbered lists without parentheses (e.g., definitions like "1.", "35.1." or Annex items)