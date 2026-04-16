import re
import logging
from typing import Optional
from pydantic import BaseModel, Field

from app.schemas.graph import ReferenceEdge, ExternalLawNode, RefersToExternalEdge
from app.core.patterns import SingletonMeta

logger = logging.getLogger(__name__)


# ==========================================
# DTO PATTERN: EXTRACTOR OUTPUT
# ==========================================

class DeterministicResult(BaseModel):
    """
    Packages the Layer 1 graph boundaries for the Orchestrator.
    Prevents the Orchestrator from needing to know how the regex works.
    """
    internal_edges: list[ReferenceEdge] = Field(default_factory=list)
    external_nodes: list[ExternalLawNode] = Field(default_factory=list)
    external_edges: list[RefersToExternalEdge] = Field(default_factory=list)


# ==========================================
# STRATEGY PATTERN: REGEX PARSING
# ==========================================

class DeterministicExtractor(metaclass=SingletonMeta):
    """
    Parses legal text to deterministically build the Layer 1
    structural Knowledge Graph skeleton.
    """

    # 1. External Legislation Regex
    # Captures: "Legea nr. 286/2009", "Ordonanța de urgență nr. 195/2002", "Regulamentul (UE) nr. 168/2013"
    REGEX_EXTERNAL = re.compile(
        r"(?P<type>Legea|Ordonan[tț][aă](?:\s+Guvernului|\s+de\s+urgen[tț][aă])?|Regulamentul\s*\(UE\))\s+nr\.\s*(?P<num>\d+/\d{4})",
        re.IGNORECASE
    )

    # 2. Internal Cross-References Regex (Unified with Named Groups)
    # The order of these branches matters; it evaluates the most complex first.
    REGEX_INTERNAL = re.compile(
        r"(?P<abs_art_range>art\.\s*(?P<r_start>\d+(?:\.\d+)?)\s*-\s*(?P<r_end>\d+(?:\.\d+)?))|"
        r"(?P<abs_art_alin>art\.\s*(?P<aa_art>\d+(?:\.\d+)?)\s*alin\.\s*\((?P<aa_alin>[a-z0-9.]+)\))|"
        r"(?P<abs_art>art\.\s*(?P<a_art>\d+(?:\.\d+)?))|"
        r"(?P<rel_alin>alin\.\s*\((?P<r_alin>[a-z0-9.]+)\))",
        re.IGNORECASE
    )

    def extract_references(self, source_id: str, content: str) -> DeterministicResult:
        """Main execution method for deterministic parsing."""
        result = DeterministicResult()

        self._parse_external(source_id, content, result)
        self._parse_internal(source_id, content, result)

        return result

    def _parse_external(self, source_id: str, content: str, result: DeterministicResult) -> None:
        """Finds boundaries triggering external routing strategies."""
        for match in self.REGEX_EXTERNAL.finditer(content):
            law_type = match.group('type').strip().title()
            num = match.group('num').strip()

            # Normalize ID: "Legea nr. 286/2009" -> "legea_286_2009"
            ext_id = f"{law_type.split()[0].lower()}_{num.replace('/', '_')}"
            ext_name = f"{law_type} nr. {num}"

            result.external_nodes.append(
                ExternalLawNode(id=ext_id, name=ext_name, law_type=law_type)
            )
            result.external_edges.append(
                RefersToExternalEdge(source_unit_id=source_id, target_external_id=ext_id)
            )

    def _parse_internal(self, source_id: str, content: str, result: DeterministicResult) -> None:
        """Finds internal nodes to build standard traversal paths."""
        base_article = self._get_base_article(source_id)

        for match in self.REGEX_INTERNAL.finditer(content):
            # Clean the dictionary of None values to enable exact structural pattern matching
            valid_groups = {k: v for k, v in match.groupdict().items() if v is not None}

            match valid_groups:
                # Format: "art. 11.2 - 11.4"
                case {"abs_art_range": _, "r_start": start, "r_end": end}:
                    start_id = f"art_{start.replace('.', '_')}"
                    end_id = f"art_{end.replace('.', '_')}"
                    result.internal_edges.append(ReferenceEdge(source_id=source_id, target_id=start_id))
                    result.internal_edges.append(ReferenceEdge(source_id=source_id, target_id=end_id))

                # Format: "art. 13 alin. (3)"
                case {"abs_art_alin": _, "aa_art": art, "aa_alin": alin}:
                    target_id = f"art_{art.replace('.', '_')}_alin_{alin.replace('.', '_')}"
                    result.internal_edges.append(ReferenceEdge(source_id=source_id, target_id=target_id))

                # Format: "art. 102"
                case {"abs_art": _, "a_art": art}:
                    target_id = f"art_{art.replace('.', '_')}"
                    result.internal_edges.append(ReferenceEdge(source_id=source_id, target_id=target_id))

                # Format: "alin. (3)" -> Requires Relative Context
                case {"rel_alin": _, "r_alin": alin}:
                    if base_article:
                        target_id = f"{base_article}_alin_{alin.replace('.', '_')}"
                        result.internal_edges.append(ReferenceEdge(source_id=source_id, target_id=target_id))

                # Fallback for safe degradation
                case _:
                    logger.debug(f"Regex matched but fell through structural matching on ID {source_id}")

    def _get_base_article(self, source_id: str) -> Optional[str]:
        """
        Extracts the parent article ID to resolve relative references.
        Example: 'art_102_alin_3_lit_a' -> 'art_102'
        """
        match = re.match(r"^(art_\d+(?:_\d+)?)", source_id)
        if match:
            return match.group(1)
        return None