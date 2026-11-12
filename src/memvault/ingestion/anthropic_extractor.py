"""
Anthropic extractor 

Uses claude-haiku to intelligently extract memorable facts from 
conversations. 
Install: pip install anthropic
Set: ANTHROPIC_API_KEY = your-key
"""
from __future__ import annotations

import json
import os

from memvault.ingestion.base import BaseExtractor, ExtractedFact

EXTRACTION_PROMPT = """You are a memory extraction system for an AI agent

Given a conversation, extract facts worth remembering long term about the user
Focus on: preferences, background, goals, skills, constraits, relationships,
identity, opinions
Ignore: questions, greetings, temporary context, assistant responses.

Return ONLY JSON array of objects. Each object must have:
 - "content": the fact is a clear, self-contained statement (string)
 - "memory_type": one of the "episodic", "semantic", "procedural", "working" (string)
 - "importance": float between 0.0 and 1.0 based on how important the memory is
 
 Example output:
 [
    {"content": "User prefers Javascript", "memory_type": "semantic", "importance": 0.8},
    {"content": "User is interested in AI CRUD projects", "memory_type": "episodic",
      "importance": 0.9}
]
If no memorable facts exists, return an empty array: []
Return only the JSON array no other text."""

class AnthropicExtractor(BaseExtractor):
    """
    LLM Powered extractor using Claude Haiku
    
    Requires: pip install anthropic + API_KEY
    """

    def __init__(
            self,
            api_key: str | None = None,
            model: str = "claude-haiku-3",
            max_facts: int = 20,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.max_facts = max_facts

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found, key required" \
                "or pass api_key= to AnthropicExtractor()."
            )

    def extract(self,
                messages: list[dict]
                )-> list[ExtractedFact]:

        """Extract facts using claude haiku"""

        try:
            import anthropic
        except ImportError as err:
            raise ImportError(
                "anthropic package required, install with pip install anthropic"
            ) from err

        client = anthropic.Anthropic(api_key=self.api_key)

        conversation_text = "\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
            for m in messages
            if m.get("content")
        )

        response = client.messages.create(
            model = self.model,
            max_tokens = 1024,
            messages=[
                {
                    "role": "user",
                    "content": f"{EXTRACTION_PROMPT}\n\nConversation:\n{conversation_text}",
                }
            ],
        )

        raw = response.content[0].text.strip()

        try:
            facts_data = json.loads(raw)
        except json.JSONDecodeError: #Model returns other tahn json
            return []

        facts = []
        for item in facts_data[: self.max_facts]:
            try:
                facts.append(
                    ExtractedFact(
                        content=item["content"],
                        memory_type=item.get("memory_type", "semantic"),
                        importance=float(item.get("importance", 0.6)),
                        tags=["auto-ingested", "anthropic"],
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue

        return facts