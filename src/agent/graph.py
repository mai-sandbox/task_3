import asyncio
from typing import cast, Any, Literal
import json

from tavily import AsyncTavilyClient
from langchain_anthropic import ChatAnthropic
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field

from .configuration import Configuration
from .state import InputState, OutputState, OverallState
from .utils import (
    deduplicate_sources, 
    format_sources, 
    format_all_notes,
    should_summarize,
    prepare_conversation_for_summary,
    count_conversation_tokens
)
from .prompts import (
    EXTRACTION_PROMPT,
    REFLECTION_PROMPT,
    INFO_PROMPT,
    QUERY_WRITER_PROMPT,
    SUMMARIZATION_PROMPT,
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
    """Summarize conversation history when it becomes too long."""
    # Prepare conversation for summarization
    messages_to_summarize, messages_to_keep = prepare_conversation_for_summary(
        state.conversation_history, keep_recent=5
    )
    
    if not messages_to_summarize:
        return {}
    
    # Format conversation history for summarization
    conversation_text = ""
    for msg in messages_to_summarize:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        conversation_text += f"{role}: {content}\n\n"
    
    # Create summary using LLM
    summary_prompt = SUMMARIZATION_PROMPT.format(
        conversation_history=conversation_text,
        company=state.company
    )
    
    result = claude_3_5_sonnet.invoke(summary_prompt)
    new_summary = str(result.content)
    
    # Combine with existing summary if present
    if state.conversation_summary:
        combined_summary = f"Previous Summary:\n{state.conversation_summary}\n\nRecent Summary:\n{new_summary}"
        final_summary = claude_3_5_sonnet.invoke(
            f"Combine these summaries into a cohesive research summary:\n\n{combined_summary}"
        )
        new_summary = str(final_summary.content)
    
    # Update state with summary and reduced conversation history
    return {
        "conversation_summary": new_summary,
        "conversation_history": messages_to_keep,
        "total_tokens": count_conversation_tokens(messages_to_keep)
    }


def check_and_update_conversation(state: OverallState) -> dict[str, Any]:
    """Check if conversation needs summarization and update tracking."""
    updates = {}
    
    # Add current research step to conversation history
    if state.completed_notes:
        latest_note = state.completed_notes[-1]
        conversation_msg = {
            "role": "assistant", 
            "content": f"Research findings: {latest_note}",
            "timestamp": "current"
        }
        updates["conversation_history"] = [conversation_msg]
    
    # Update token count
    current_tokens = count_conversation_tokens(state.conversation_history + updates.get("conversation_history", []))
    updates["total_tokens"] = current_tokens
    
    return updates


def route_from_reflection(
    state: OverallState, config: RunnableConfig
) -> Literal[END, "research_company", "summarize_conversation"]:  # type: ignore
    """Route the graph based on the reflection output and conversation length."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    
    # Check if conversation needs summarization first
    if should_summarize(state.conversation_history):
        return "summarize_conversation"

    # If we have satisfactory results, end the process
    if state.is_satisfactory:
        return END

    # If results aren't satisfactory but we haven't hit max steps, continue research
    if state.reflection_steps_taken <= configurable.max_reflection_steps:
        return "research_company"

    # If we've exceeded max steps, end even if not satisfactory
    return END


def route_after_summarization(
    state: OverallState, config: RunnableConfig
) -> Literal[END, "research_company"]:  # type: ignore
    """Route after summarization based on original reflection logic."""
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
builder.add_node("check_and_update_conversation", check_and_update_conversation)
builder.add_node("summarize_conversation", summarize_conversation)

builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", "research_company")
builder.add_edge("research_company", "gather_notes_extract_schema")
builder.add_edge("gather_notes_extract_schema", "check_and_update_conversation")
builder.add_edge("check_and_update_conversation", "reflection")
builder.add_conditional_edges("reflection", route_from_reflection)
builder.add_conditional_edges("summarize_conversation", route_after_summarization)

# Compile
graph = builder.compile()
