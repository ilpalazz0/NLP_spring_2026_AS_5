from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT_DIR / "code"
RAG_CODE_DIR = CODE_DIR / "rag_system"

if str(RAG_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_CODE_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(1, str(CODE_DIR))
