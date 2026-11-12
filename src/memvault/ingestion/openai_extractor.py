"""
OpenAI extractor

Uses an OpenAI model to intelligently extract memorable facts from
conversations.
Install: pip install openai
Set: OPENAI_API_KEY = your-key
"""
from __future__ import annotations

import json
import os

from memvault.ingestion.base import BaseExtractor, ExtractedFact

EXTRACTION_PROMPT = """You are a memory extraction system for an AI agent.

Given a conversation, extract facts worth remembering long term about the user.
Focus on: preferences, background, goals, skills, constraints, relationships,
identity, opinions.
Ignore: questions, greetings, temporary context, assistant responses.

Return ONLY a JSON array of objects. Each object must have:
 - "content": the fact as a clear, self-contained statement (string)
 - "memory_type": one of "episodic", "semantic", "procedural", "working" (string)
 - "importance": float between 0.0 and 1.0

Example output:
[
  {"content": "User prefers Python", "memory_type": "semantic", "importance": 0.8},
  {"content": "User is building a CRM", "memory_type": "episodic", "importance": 0.9}
]
If no memorable facts exist, return an empty array: []
Return only the JSON array, no other text.
"""


class OpenAIExtractor(BaseExtractor):
    """
    LLM-powered extractor using OpenAI.

    Requires: pip install openai + OPENAI_API_KEY
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_facts: int = 20,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.max_facts = max_facts

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found, key required "
                "or pass api_key= to OpenAIExtractor()."
            )

    def extract(self, messages: list[dict]) -> list[ExtractedFact]:
        """Extract facts using an OpenAI model."""

        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError(
                "openai package required, install with pip install openai"
            ) from err

        client = OpenAI(api_key=self.api_key)

        conversation_text = "\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
            for m in messages
            if m.get("content")
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": EXTRACTION_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"Conversation:\n{conversation_text}",
                },
            ],
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()

        try:
            facts_data = json.loads(raw)
        except (json.JSONDecodeError, AttributeError):
            return []

        facts = []
        for item in facts_data[: self.max_facts]:
            try:
                facts.append(
                    ExtractedFact(
                        content=item["content"],
                        memory_type=item.get("memory_type", "semantic"),
                        importance=float(item.get("importance", 0.6)),
                        tags=["auto-ingested", "openai"],
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue

        return facts
