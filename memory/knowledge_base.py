"""
Phase 10: Document and user-profile storage outside model weights.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class KnowledgeBase:
    """Store documents and user profiles as JSON files on disk."""

    def __init__(self, base_dir: str = "memory/store"):
        self.base_dir = Path(base_dir)
        self.docs_dir = self.base_dir / "documents"
        self.profiles_dir = self.base_dir / "profiles"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def add_document(self, doc_id: str, content: str, metadata: Optional[dict] = None) -> None:
        data = {
            "id": doc_id,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
        }
        path = self.docs_dir / f"{doc_id}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_document(self, doc_id: str) -> Optional[dict]:
        path = self.docs_dir / f"{doc_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_documents(self) -> List[str]:
        return [p.stem for p in self.docs_dir.glob("*.json")]

    def set_user_profile(self, user_id: str, profile: dict) -> None:
        path = self.profiles_dir / f"{user_id}.json"
        profile["updated_at"] = datetime.utcnow().isoformat()
        path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    def get_user_profile(self, user_id: str) -> Optional[dict]:
        path = self.profiles_dir / f"{user_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def load_all_documents(self) -> List[dict]:
        return [
            json.loads(p.read_text(encoding="utf-8"))
            for p in self.docs_dir.glob("*.json")
        ]
