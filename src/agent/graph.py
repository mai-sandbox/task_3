import asyncio
from typing import cast, Any, Literal
import json

from tavily import AsyncTavilyClient
from langchain_anthropic import ChatAnthropic
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field

from agent.configuration import Configuration
from agent.state import InputState, OutputState, OverallState
from agent.utils import deduplicate_sources, format_sources, format_all_notes, count_tokens, should_summarize
from agent.prompts import (
    EXTRACTION_PROMPT,
    REFLECTION_PROMPT,
    INFO_PROMPT,
    QUERY_WRITER_PROMPT,
    CONVERSATION_SUMMARIZATION_PROMPT,
)

# LLMs

rate_limiter = InMemoryRateLimiter(
    requests_per_second=4,
    check_every_n_seconds=0.1,
    max_bucket_size=10,  # Controls the maximum burst size.
)
claude_3_5_sonnet = ChatAnthropic(
    model="claude-3-5-sonnet-latest", temperature=0, rate_limiter=rate_limiter
)

# Search

tavily_async_client = AsyncTavilyClient()


class Queries(BaseModel):
    queries: list[str] = Field(
        description="List of search queries.",
    )


class ReflectionOutput(BaseModel):
    is_satisfactory: bool = Field(
        description="True if all required fields are well populated, False otherwise"
    )
    missing_fields: list[str] = Field(
        description="List of field names that are missing or incomplete"
    )
    search_queries: list[str] = Field(
        description="If is_satisfactory is False, provide 1-3 targeted search queries to find the missing information"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")


class ConversationSummary(BaseModel):
    summary: str = Field(
        description="Concise summary of the older messages (first 60%) that preserves key company research findings, insights, and important context"
    )
    preserved_messages: list[dict] = Field(
        description="The most recent messages (last 40%) to keep intact for immediate context"
    )
    key_findings: list[str] = Field(
        description="List of key company research findings and insights that must be preserved"
    )
    total_original_messages: int = Field(
        description="Total number of messages in the original conversation history"
    )
    summary_token_count: int = Field(
        description="Approximate token count of the summary"
    )


def generate_queries(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Generate search queries based on the user input and extraction schema."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_search_queries = configurable.max_search_queries

    # Generate search queries
    structured_llm = claude_3_5_sonnet.with_structured_output(Queries)

    # Format system instructions
    query_instructions = QUERY_WRITER_PROMPT.format(
        company=state.company,
        info=json.dumps(state.extraction_schema, indent=2),
        user_notes=state.user_notes,
        max_search_queries=max_search_queries,
    )

    # Generate queries
    results = cast(
        Queries,
        structured_llm.invoke(
            [
                {"role": "system", "content": query_instructions},
                {
                    "role": "user",
                    "content": "Please generate a list of search queries related to the schema that you want to populate.",
                },
            ]
        ),
    )

    # Queries
    query_list = [query for query in results.queries]
    return {"search_queries": query_list}


async def research_company(
    state: OverallState, config: RunnableConfig
) -> dict[str, Any]:
    """Execute a multi-step web search and information extraction process.

    This function performs the following steps:
    1. Executes concurrent web searches using the Tavily API
    2. Deduplicates and formats the search results
    """

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_search_results = configurable.max_search_results

    # Search tasks
    search_tasks = []
    for query in state.search_queries:
        search_tasks.append(
            tavily_async_client.search(
                query,
                max_results=max_search_results,
                include_raw_content=True,
                topic="general",
            )
        )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)

    # Deduplicate and format sources
    deduplicated_search_docs = deduplicate_sources(search_docs)
    source_str = format_sources(
        deduplicated_search_docs, max_tokens_per_source=1000, include_raw_content=True
    )

    # Generate structured notes relevant to the extraction schema
    p = INFO_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2),
        content=source_str,
        company=state.company,
        user_notes=state.user_notes,
    )
    result = await claude_3_5_sonnet.ainvoke(p)
    state_update = {
        "completed_notes": [str(result.content)],
    }
    if configurable.include_search_results:
        state_update["search_results"] = deduplicated_search_docs

    return state_update


def gather_notes_extract_schema(state: OverallState) -> dict[str, Any]:
    """Gather notes from the web search and extract the schema fields."""

    # Format all notes
    notes = format_all_notes(state.completed_notes)

    # Extract schema fields
    system_prompt = EXTRACTION_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2), notes=notes
    )
    structured_llm = claude_3_5_sonnet.with_structured_output(state.extraction_schema)
    result = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Produce a structured output from these notes.",
            },
        ]
    )
    return {"info": result}


def reflection(state: OverallState) -> dict[str, Any]:
    """Reflect on the extracted information and generate search queries to find missing information."""
    structured_llm = claude_3_5_sonnet.with_structured_output(ReflectionOutput)

    # Format reflection prompt
    system_prompt = REFLECTION_PROMPT.format(
        schema=json.dumps(state.extraction_schema, indent=2),
        info=state.info,
    )

    # Invoke
    result = cast(
        ReflectionOutput,
        structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Produce a structured reflection output."},
            ]
        ),
    )

    if result.is_satisfactory:
        return {"is_satisfactory": result.is_satisfactory}
    else:
        return {
            "is_satisfactory": result.is_satisfactory,
            "search_queries": result.search_queries,
            "reflection_steps_taken": state.reflection_steps_taken + 1,
        }


def summarize_conversation(state: OverallState) -> dict[str, Any]:
    """Summarize conversation history when token limits are approached.
    
    Compresses the first 60% of messages into a concise summary while preserving
    the most recent 40% intact. Maintains key company research findings and insights.
    """
    if not state.conversation_history:
        return {}
    
    # Calculate split point: first 60% to summarize, last 40% to preserve
    total_messages = len(state.conversation_history)
    split_point = int(total_messages * 0.6)
    
    messages_to_summarize = state.conversation_history[:split_point]
    messages_to_preserve = state.conversation_history[split_point:]
    
    if not messages_to_summarize:
        # If no messages to summarize, return unchanged
        return {}
    
    # We'll need the CONVERSATION_SUMMARIZATION_PROMPT from the next task
    # For now, create a basic prompt inline
    summarization_prompt = """You are tasked with summarizing a conversation history to reduce token usage while preserving critical information.

Your task:
1. Create a concise summary of the provided messages that preserves:
   - Key company research findings and insights
   - Important search queries and results
   - Critical data points and facts discovered
   - Overall conversation flow and context

2. Extract and list the most important research findings separately

3. Preserve the recent messages exactly as provided

Messages to summarize:
{messages_to_summarize}

Recent messages to preserve:
{messages_to_preserve}

Company being researched: {company}

Create a structured summary that maintains all critical research context while significantly reducing token count."""
    
    # Format messages for the prompt
    messages_text = ""
    for i, msg in enumerate(messages_to_summarize):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        messages_text += f"[{i+1}] {role} ({timestamp}): {content}\n\n"
    
    preserve_text = ""
    for i, msg in enumerate(messages_to_preserve):
        role = msg.get("role", "unknown") 
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        preserve_text += f"[{i+1}] {role} ({timestamp}): {content}\n\n"
    
    # Use structured output for consistent results
    structured_llm = claude_3_5_sonnet.with_structured_output(ConversationSummary)
    
    # Format the prompt
    system_prompt = summarization_prompt.format(
        messages_to_summarize=messages_text,
        messages_to_preserve=preserve_text,
        company=state.company
    )
    
    # Invoke the LLM
    result = cast(
        ConversationSummary,
        structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Please provide a structured conversation summary."}
        ])
    )
    
    # Create the new conversation history with summary + preserved messages
    summary_message = {
        "role": "system",
        "content": f"[CONVERSATION SUMMARY] {result.summary}",
        "timestamp": "summarized",
        "type": "summary",
        "key_findings": result.key_findings,
        "original_message_count": result.total_original_messages
    }
    
    # Build new conversation history: summary + preserved messages
    new_conversation_history = [summary_message] + result.preserved_messages
    
    # Calculate new token count
    new_token_count = count_tokens(new_conversation_history)
    
    return {
        "conversation_history": new_conversation_history,
        "total_tokens": new_token_count
    }


def route_from_reflection(
    state: OverallState, config: RunnableConfig
) -> Literal[END, "research_company"]:  # type: ignore
    """Route the graph based on the reflection output."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # If we have satisfactory results, end the process
    if state.is_satisfactory:
        return END

    # If results aren't satisfactory but we haven't hit max steps, continue research
    if state.reflection_steps_taken <= configurable.max_reflection_steps:
        return "research_company"

    # If we've exceeded max steps, end even if not satisfactory
    return END


# Add nodes and edges
builder = StateGraph(
    OverallState,
    input=InputState,
    output=OutputState,
    config_schema=Configuration,
)
builder.add_node("gather_notes_extract_schema", gather_notes_extract_schema)
builder.add_node("generate_queries", generate_queries)
builder.add_node("research_company", research_company)
builder.add_node("reflection", reflection)
builder.add_node("summarize_conversation", summarize_conversation)

builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", "research_company")
builder.add_edge("research_company", "gather_notes_extract_schema")
builder.add_edge("gather_notes_extract_schema", "reflection")
builder.add_conditional_edges("reflection", route_from_reflection)

# Compile
graph = builder.compile()





