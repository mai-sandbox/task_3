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
from agent.utils import deduplicate_sources, format_sources, format_all_notes
from agent.conversation import ConversationManager
from agent.prompts import (
    EXTRACTION_PROMPT,
    REFLECTION_PROMPT,
    INFO_PROMPT,
    QUERY_WRITER_PROMPT,
)
from langchain_core.messages import HumanMessage, AIMessage

# LLMs

rate_limiter = InMemoryRateLimiter(
    requests_per_second=4,
    check_every_n_seconds=0.1,
    max_bucket_size=10,  # Controls the maximum burst size.
)
claude_3_5_sonnet = ChatAnthropic(
    model="claude-3-5-sonnet-latest", temperature=0, rate_limiter=rate_limiter
)

# Initialize conversation manager
conversation_manager = ConversationManager(
    max_tokens=20000,
    summary_trigger_ratio=0.8,
    preserve_recent_messages=5
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


async def generate_queries(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Generate search queries based on the user input and extraction schema."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_search_queries = configurable.max_search_queries

    # Check and manage conversation history
    context = {
        "company": state.company,
        "extraction_schema": state.extraction_schema,
        "info": state.info or {},
    }
    managed_history = await conversation_manager.manage_conversation_history(
        state.conversation_history, context
    )

    # Generate search queries
    structured_llm = claude_3_5_sonnet.with_structured_output(Queries)

    # Format system instructions
    query_instructions = QUERY_WRITER_PROMPT.format(
        company=state.company,
        info=json.dumps(state.extraction_schema, indent=2),
        user_notes=state.user_notes,
        max_search_queries=max_search_queries,
    )

    # Add user message to history
    user_msg = HumanMessage(content="Generate search queries for company research")
    
    # Generate queries
    results = cast(
        Queries,
        await structured_llm.ainvoke(
            [
                {"role": "system", "content": query_instructions},
                {
                    "role": "user",
                    "content": "Please generate a list of search queries related to the schema that you want to populate.",
                },
            ]
        ),
    )

    # Add AI response to history
    ai_msg = AIMessage(content=f"Generated {len(results.queries)} search queries: {', '.join(results.queries)}")
    
    # Update conversation history
    new_history = managed_history + [user_msg, ai_msg]
    
    # Queries
    query_list = [query for query in results.queries]
    return {
        "search_queries": query_list,
        "conversation_history": new_history,
        "conversation_summarized": len(managed_history) < len(state.conversation_history)
    }


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

    # Check and manage conversation history
    context = {
        "company": state.company,
        "extraction_schema": state.extraction_schema,
        "info": state.info or {},
    }
    managed_history = await conversation_manager.manage_conversation_history(
        state.conversation_history, context
    )
    
    # Generate structured notes relevant to the extraction schema
    p = INFO_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2),
        content=source_str,
        company=state.company,
        user_notes=state.user_notes,
    )
    
    # Add research action to history
    user_msg = HumanMessage(content=f"Researching {len(state.search_queries)} queries about {state.company}")
    
    result = await claude_3_5_sonnet.ainvoke(p)
    
    # Add AI response to history
    ai_msg = AIMessage(content=f"Completed research and found relevant information from {len(deduplicated_search_docs)} sources")
    
    # Update conversation history
    new_history = managed_history + [user_msg, ai_msg]
    
    state_update = {
        "completed_notes": [str(result.content)],
        "conversation_history": new_history,
        "conversation_summarized": len(managed_history) < len(state.conversation_history)
    }
    if configurable.include_search_results:
        state_update["search_results"] = deduplicated_search_docs

    return state_update


async def gather_notes_extract_schema(state: OverallState) -> dict[str, Any]:
    """Gather notes from the web search and extract the schema fields."""
    
    # Check and manage conversation history
    context = {
        "company": state.company,
        "extraction_schema": state.extraction_schema,
        "info": state.info or {},
    }
    managed_history = await conversation_manager.manage_conversation_history(
        state.conversation_history, context
    )

    # Format all notes
    notes = format_all_notes(state.completed_notes)
    
    # Add extraction action to history
    user_msg = HumanMessage(content="Extracting structured information from research notes")

    # Extract schema fields
    system_prompt = EXTRACTION_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2), notes=notes
    )
    structured_llm = claude_3_5_sonnet.with_structured_output(state.extraction_schema)
    result = await structured_llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Produce a structured output from these notes.",
            },
        ]
    )
    
    # Add AI response to history
    ai_msg = AIMessage(content=f"Extracted structured information: {json.dumps(result, indent=2)}")
    
    # Update conversation history
    new_history = managed_history + [user_msg, ai_msg]
    
    return {
        "info": result,
        "conversation_history": new_history,
        "conversation_summarized": len(managed_history) < len(state.conversation_history)
    }


async def reflection(state: OverallState) -> dict[str, Any]:
    """Reflect on the extracted information and generate search queries to find missing information."""
    
    # Check and manage conversation history
    context = {
        "company": state.company,
        "extraction_schema": state.extraction_schema,
        "info": state.info or {},
    }
    managed_history = await conversation_manager.manage_conversation_history(
        state.conversation_history, context
    )
    
    structured_llm = claude_3_5_sonnet.with_structured_output(ReflectionOutput)
    
    # Add reflection action to history
    user_msg = HumanMessage(content="Reflecting on extracted information completeness")

    # Format reflection prompt
    system_prompt = REFLECTION_PROMPT.format(
        schema=json.dumps(state.extraction_schema, indent=2),
        info=state.info,
    )

    # Invoke
    result = cast(
        ReflectionOutput,
        await structured_llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Produce a structured reflection output."},
            ]
        ),
    )
    
    # Add AI response to history
    if result.is_satisfactory:
        ai_msg = AIMessage(content="All required information has been successfully extracted")
    else:
        ai_msg = AIMessage(content=f"Missing information identified. Generated {len(result.search_queries)} new search queries")
    
    # Update conversation history
    new_history = managed_history + [user_msg, ai_msg]

    if result.is_satisfactory:
        return {
            "is_satisfactory": result.is_satisfactory,
            "conversation_history": new_history,
            "conversation_summarized": len(managed_history) < len(state.conversation_history)
        }
    else:
        return {
            "is_satisfactory": result.is_satisfactory,
            "search_queries": result.search_queries,
            "reflection_steps_taken": state.reflection_steps_taken + 1,
            "conversation_history": new_history,
            "conversation_summarized": len(managed_history) < len(state.conversation_history)
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


async def monitor_conversation(state: OverallState) -> dict[str, Any]:
    """Monitor conversation statistics and log if summarization occurred."""
    stats = conversation_manager.get_conversation_stats(state.conversation_history)
    
    if state.conversation_summarized:
        print(f"📊 Conversation summarized to manage token usage")
        print(f"   - Messages: {stats['total_messages']}")
        print(f"   - Token usage: {stats['token_usage_ratio']:.1%}")
    
    # Return empty dict as this is a monitoring node
    return {}


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
builder.add_node("monitor_conversation", monitor_conversation)

builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", "monitor_conversation")
builder.add_edge("monitor_conversation", "research_company")
builder.add_edge("research_company", "gather_notes_extract_schema")
builder.add_edge("gather_notes_extract_schema", "reflection")
builder.add_conditional_edges("reflection", route_from_reflection)

# Compile
graph = builder.compile()
