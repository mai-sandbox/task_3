"""State management for the company research agent.

This module defines the state classes used by the LangGraph agent, including
input/output states and the main OverallState that tracks conversation history,
research progress, and token usage for dynamic summarization.
"""
import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional

from langchain_core.messages import BaseMessage

DEFAULT_EXTRACTION_SCHEMA = {
    "title": "CompanyInfo",
    "description": "Basic information about a company",
    "type": "object",
    "properties": {
        "company_name": {
            "type": "string",
            "description": "Official name of the company",
        },
        "founding_year": {
            "type": "integer",
            "description": "Year the company was founded",
        },
        "founder_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names of the founding team members",
        },
        "product_description": {
            "type": "string",
            "description": "Brief description of the company's main product or service",
        },
        "funding_summary": {
            "type": "string",
            "description": "Summary of the company's funding history",
        },
    },
    "required": ["company_name"],
}


@dataclass(kw_only=True)
class InputState:
    """Input state defines the interface between the graph and the user (external API)."""

    company: str
    "Company to research provided by the user."

    extraction_schema: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_EXTRACTION_SCHEMA
    )
    "The json schema defines the information the agent is tasked with filling out."

    user_notes: Optional[dict[str, Any]] = field(default=None)
    "Any notes from the user to start the research process."


@dataclass(kw_only=True)
class OverallState:
    """Input state defines the interface between the graph and the user (external API)."""

    company: str
    "Company to research provided by the user."

    extraction_schema: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_EXTRACTION_SCHEMA
    )
    "The json schema defines the information the agent is tasked with filling out."

    user_notes: str = field(default=None)
    "Any notes from the user to start the research process."

    search_queries: list[str] = field(default=None)
    "List of generated search queries to find relevant information"

    search_results: list[dict] = field(default=None)
    "List of search results"

    completed_notes: Annotated[list, operator.add] = field(default_factory=list)
    "Notes from completed research related to the schema"

    info: dict[str, Any] = field(default=None)
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """

    is_satisfactory: bool = field(default=None)
    "True if all required fields are well populated, False otherwise"

    reflection_steps_taken: int = field(default=0)
    "Number of times the reflection node has been executed"

    messages: Annotated[list[BaseMessage], operator.add] = field(default_factory=list)
    "Conversation history tracking all messages exchanged during research"

    summary: str = field(default="")
    "Summary of the conversation history to preserve context when messages are trimmed"

    total_tokens: int = field(default=0)
    "Current token count of the conversation history for managing token limits"


@dataclass(kw_only=True)
class OutputState:
    """The response object for the end user.

    This class defines the structure of the output that will be provided
    to the user after the graph's execution is complete.
    """

    info: dict[str, Any]
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """

    search_results: list[dict] = field(default=None)
    "List of search results"

