"""
Phase 10: Persistent chat history per conversation.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import uuid


class ChatHistory:
    def __init__(self, base_dir: str = "memory/store/conversations"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_conversation(self, title: str = "New Chat") -> str:
        conv_id = str(uuid.uuid4())[:8]
        data = {
            "id": conv_id,
            "title": title,
            "messages": [],
            "created_at": datetime.utcnow().isoformat(),
        }
        self._save(conv_id, data)
        return conv_id

    def add_message(self, conv_id: str, role: str, content: str) -> None:
        data = self.get_conversation(conv_id)
        if data is None:
            raise ValueError(f"Conversation {conv_id} not found")
        data["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self._save(conv_id, data)

    def get_conversation(self, conv_id: str) -> Optional[dict]:
        path = self.base_dir / f"{conv_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_conversations(self) -> List[Dict[str, str]]:
        convs = []
        for p in self.base_dir.glob("*.json"):
            data = json.loads(p.read_text(encoding="utf-8"))
            convs.append({"id": data["id"], "title": data.get("title", p.stem)})
        return sorted(convs, key=lambda c: c["id"], reverse=True)

    def get_messages(self, conv_id: str) -> List[dict]:
        data = self.get_conversation(conv_id)
        return data["messages"] if data else []

    def _save(self, conv_id: str, data: dict) -> None:
        path = self.base_dir / f"{conv_id}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
