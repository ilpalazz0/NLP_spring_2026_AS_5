from __future__ import annotations

from rag_system.schemas import DocumentRecord
from rag_system.utils.text import normalize_whitespace, strip_html


class TextCleaner:
    def __init__(
        self,
        lowercase: bool = False,
        strip_html_tags: bool = True,
        remove_extra_whitespace: bool = True,
    ) -> None:
        self.lowercase = lowercase
        self.strip_html_tags = strip_html_tags
        self.remove_extra_whitespace = remove_extra_whitespace

    def clean_text(self, text: str) -> str:
        value = text
        if self.strip_html_tags:
            value = strip_html(value)
        if self.lowercase:
            value = value.lower()
        if self.remove_extra_whitespace:
            value = normalize_whitespace(value)
        return value

    def clean_document(self, document: DocumentRecord) -> DocumentRecord:
        return DocumentRecord(
            doc_id=document.doc_id,
            title=self.clean_text(document.title) or document.doc_id,
            text=self.clean_text(document.text),
            metadata=document.metadata,
        )
