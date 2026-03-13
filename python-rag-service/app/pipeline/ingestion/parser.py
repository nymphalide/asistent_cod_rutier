import re
from typing import List, Optional, Dict, Any, Tuple
from app.schemas.law_unit import LawUnitCreate
from app.core.custom_types import UnitType


class TrafficCodeParser:
    REGEX_CHAPTER = re.compile(r"^(CAPITOLUL|TITLUL)\s+([IVXLCDM]+)(?::\s*(.*))?", re.IGNORECASE)
    REGEX_SECTION = re.compile(r"^SEC[TȚ]IUNEA\s+(?:nr\.\s*)?(\d+|a\s+[a-z0-9-]+-a)(?:\s*(.*))?", re.IGNORECASE)
    REGEX_ART = re.compile(r"^Art\.\s*(\d+(?:\.\d+)?)\.?\s*(.*)", re.IGNORECASE)
    REGEX_PARA = re.compile(r"^\((\d+(?:\.\d+)?)\)\s*(.*)")
    REGEX_LETTER = re.compile(r"^([a-z])\)\s*(.*)")
    REGEX_NUM_ITEM = re.compile(r"^(\d+(?:\.\d+)?)\.\s*(.*?)(?:\s*-\s*(.*))?$")

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.results: List[LawUnitCreate] = []

        self.active_chapter: Optional[str] = None
        self.active_section: Optional[str] = None
        self.active_article: Optional[str] = None
        self.active_paragraph: Optional[str] = None
        self.active_list_parent: Optional[str] = None

        self.current_buffer: List[str] = []
        self.pending_meta: Optional[Dict[str, Any]] = None

    def _read_file_safely(self, file_path: str) -> List[str]:
        encodings = ['utf-8-sig', 'iso-8859-2', 'cp1252', 'utf-8']
        for enc in encodings:
            try:
                # Adding newline=None forces Python to translate all \r\n to \n automatically
                with open(file_path, 'r', encoding=enc, newline=None) as f:
                    # strip('\r\n ') guarantees no Windows ghosts survive
                    return [line.strip('\r\n ') for line in f.readlines() if line.strip('\r\n ')]
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Encoding failed for {file_path}")

    def _flush_buffer(self):
        if self.pending_meta and self.current_buffer:
            content = " ".join(self.current_buffer)
            self.results.append(LawUnitCreate(content=content, **self.pending_meta))

        self.current_buffer = []
        self.pending_meta = None

    def _tokenize(self, line: str) -> Tuple[str, Any]:
        """The Lexer: Returns a structural tuple (Token_Type, Match_Data)"""
        if match := self.REGEX_CHAPTER.match(line): return ('CHAPTER', match)
        if match := self.REGEX_SECTION.match(line): return ('SECTION', match) # <-- ADDED
        if match := self.REGEX_ART.match(line): return ('ARTICLE', match)
        if match := self.REGEX_PARA.match(line): return ('PARAGRAPH', match)
        if match := self.REGEX_LETTER.match(line): return ('LETTER', match)
        if match := self.REGEX_NUM_ITEM.match(line): return ('NUMBERED', match)
        return ('TEXT', line)

    def parse(self) -> List[LawUnitCreate]:
        lines = self._read_file_safely(self.file_path)

        for line in lines:
            token_type, match_data = self._tokenize(line)

            # Universal Pre-processing: Flush if a NEW structural unit begins
            if token_type != 'TEXT':
                self._flush_buffer()

            match token_type:
                case 'CHAPTER':
                    num, title = match_data.group(2), match_data.group(3) or line
                    self.active_chapter = f"cap_{num}"
                    self.active_section = self.active_article = self.active_paragraph = self.active_list_parent = None
                    self.pending_meta = {"id": self.active_chapter, "parent_id": None, "unit_type": UnitType.CHAPTER, "metadata": {"title": title}}

                case 'SECTION':
                    sec_num = match_data.group(1).replace(' ', '_')
                    title = match_data.group(2) or line
                    unit_id = f"{self.active_chapter}_sec_{sec_num}" if self.active_chapter else f"sec_{sec_num}"
                    self.active_section = unit_id
                    self.active_article = self.active_paragraph = self.active_list_parent = None
                    self.pending_meta = {"id": self.active_section, "parent_id": self.active_chapter, "unit_type": UnitType.SECTION, "metadata": {"title": title, "number": sec_num}}

                case 'ARTICLE':
                    art_num = match_data.group(1)
                    self.active_article = f"art_{art_num.replace('.', '_')}"
                    self.active_paragraph = self.active_list_parent = None
                    self.pending_meta = {"id": self.active_article, "parent_id": self.active_section or self.active_chapter, "unit_type": UnitType.ARTICLE, "metadata": {"number": art_num}}

                case 'PARAGRAPH':
                    para_num = match_data.group(1)
                    self.active_paragraph = f"{self.active_article}_alin_{para_num.replace('.', '_')}"
                    self.active_list_parent = self.active_paragraph
                    self.pending_meta = {"id": self.active_paragraph, "parent_id": self.active_article, "unit_type": UnitType.PARAGRAPH, "metadata": {"number": para_num}}

                case 'LETTER':
                    letter = match_data.group(1)
                    unit_id = f"{self.active_list_parent or self.active_article}_lit_{letter}"
                    self.pending_meta = {"id": unit_id, "parent_id": self.active_list_parent or self.active_article, "unit_type": UnitType.LETTER_ITEM, "metadata": {"letter": letter}}

                case 'NUMBERED':
                    num = match_data.group(1)
                    unit_id = f"{self.active_article or 'anexa'}_pct_{num.replace('.', '_')}"
                    self.pending_meta = {"id": unit_id, "parent_id": self.active_list_parent or self.active_article, "unit_type": UnitType.NUMBERED_ITEM, "metadata": {"number": num}}

                case 'TEXT':
                    # Edge case: Catching unnumbered prologue text before paragraphs start
                    if not self.current_buffer and self.active_article and not self.active_paragraph:
                        self.pending_meta = {"id": f"{self.active_article}_prologue", "parent_id": self.active_article, "unit_type": UnitType.PROLOGUE, "metadata": {}}

            # Universal Post-processing: Always append the current line to the active buffer
            self.current_buffer.append(line)

        self._flush_buffer()
        return self.results