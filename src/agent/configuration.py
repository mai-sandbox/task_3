"""Configuration management for the company research agent.

This module defines the Configuration class that manages all configurable parameters
for the research agent, including conversation management settings and API configurations.
"""
import os
from dataclasses import dataclass, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""

    max_search_queries: int = 3  # Max search queries per company
    max_search_results: int = 3  # Max search results per query
    max_reflection_steps: int = 0  # Max reflection steps
    include_search_results: bool = (
        False  # Whether to include search results in the output
    )

    # Conversation management settings
    max_conversation_tokens: int = (
        20000  # Maximum tokens allowed in conversation history
    )
    summarization_trigger_tokens: int = (
        18000  # Token threshold to trigger summarization
    )
    messages_to_keep_after_summary: int = (
        5  # Number of recent messages to preserve after summarization
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})

