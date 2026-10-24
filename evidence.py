"""Evidence validation, normalization, formatting, and citation verification.

Treats all user-supplied evidence as passive, untrusted text data. Enforces ID uniqueness,
bounds text lengths, encapsulates evidence in injection-resistant XML blocks, and
validates citations to prevent model fabrication.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from config import Settings
from schemas import EvidenceItem


class EvidenceValidationError(ValueError):
    """Raised when supplied evidence violates uniqueness, format, or size constraints."""
    pass


class EvidenceManager:
    """Manages reference evidence items with integrity checks and prompt packaging."""

    def __init__(self, evidence_items: List[EvidenceItem], settings: Settings):
        self.settings = settings
        self._evidence_map: Dict[str, EvidenceItem] = {}
        self._validate_and_index(evidence_items)

    def _validate_and_index(self, items: List[EvidenceItem]) -> None:
        """Validate evidence count, ID uniqueness, and character length limits."""
        if not items:
            raise EvidenceValidationError("At least one evidence item must be supplied.")

        if len(items) > self.settings.max_evidence_items:
            raise EvidenceValidationError(
                f"Too many evidence items: received {len(items)}, "
                f"maximum allowed is {self.settings.max_evidence_items}."
            )

        seen_ids: Set[str] = set()
        total_evidence_chars = 0

        for item in items:
            clean_id = item.id.strip()
            if not clean_id:
                raise EvidenceValidationError("Evidence ID cannot be empty or whitespace.")

            if clean_id in seen_ids:
                raise EvidenceValidationError(
                    f"Duplicate evidence ID detected: '{clean_id}'. Evidence IDs must be unique."
                )
            seen_ids.add(clean_id)

            text_len = len(item.text)
            if text_len > self.settings.max_evidence_text_chars:
                raise EvidenceValidationError(
                    f"Evidence item '{clean_id}' exceeds maximum character limit: "
                    f"{text_len} > {self.settings.max_evidence_text_chars} chars."
                )

            total_evidence_chars += text_len
            self._evidence_map[clean_id] = item

    @property
    def item_count(self) -> int:
        """Return total number of loaded evidence items."""
        return len(self._evidence_map)

    @property
    def all_ids(self) -> Set[str]:
        """Return the set of all valid evidence IDs."""
        return set(self._evidence_map.keys())

    def get(self, evidence_id: str) -> Optional[EvidenceItem]:
        """Retrieve evidence item by ID."""
        return self._evidence_map.get(evidence_id)

    def filter_valid_ids(self, cited_ids: List[str]) -> Tuple[List[str], List[str]]:
        """Separate cited IDs into valid (supplied) and fabricated/hallucinated IDs.
        
        Returns:
            Tuple of (valid_ids, invalid_ids)
        """
        valid: List[str] = []
        invalid: List[str] = []
        seen: Set[str] = set()

        for raw_id in cited_ids:
            clean_id = str(raw_id).strip() if raw_id is not None else ""
            if not clean_id or clean_id in seen:
                continue
            seen.add(clean_id)
            if clean_id in self._evidence_map:
                valid.append(clean_id)
            else:
                invalid.append(clean_id)

        return valid, invalid

    def format_for_prompt(self) -> str:
        """Format evidence into structured XML-like tags to resist prompt injection.
        
        Treats the content strictly as data within bounded tags.
        """
        lines = ["<reference_evidence_store>"]
        for ev_id, item in self._evidence_map.items():
            if item.source:
                encoded_source = (
                    item.source.replace("&", "&amp;")
                    .replace('"', "&quot;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                source_attr = f' source="{encoded_source}"'
            else:
                source_attr = ""

            # Escape all closing delimiters case-insensitively with whitespace tolerance
            escaped_text = re.sub(
                r"</\s*(evidence|reference_evidence_store)\s*>",
                r"&lt;/\1&gt;",
                item.text,
                flags=re.IGNORECASE,
            )
            lines.append(f'  <evidence id="{ev_id}"{source_attr}>')
            lines.append(f"    {escaped_text}")
            lines.append("  </evidence>")
        lines.append("</reference_evidence_store>")
        return "\n".join(lines)
