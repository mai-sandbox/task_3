import asyncio
import json
from typing import Any, Literal, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient

from agent.configuration import Configuration
from agent.prompts import (
    EXTRACTION_PROMPT,
    INFO_PROMPT,
    QUERY_WRITER_PROMPT,
    REFLECTION_PROMPT,
    SUMMARIZATION_PROMPT,
)
from agent.state import InputState, OutputState, OverallState
from agent.utils import (
    count_conversation_tokens,
    deduplicate_sources,
    format_all_notes,
    format_sources,
    should_summarize,
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

    # Add conversation tracking - user request for query generation
    user_message = HumanMessage(
        content=f"Generate search queries for researching {state.company}. "
                f"Schema: {json.dumps(state.extraction_schema, indent=2)}. "
                f"User notes: {state.user_notes}"
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
    
    # Add conversation tracking - AI response with generated queries
    ai_message = AIMessage(
        content=f"Generated {len(query_list)} search queries for {state.company}: {', '.join(query_list)}"
    )

    return {
        "search_queries": query_list,
        "messages": [user_message, ai_message]
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

    # Add conversation tracking - user request for research
    user_message = HumanMessage(
        content=f"Execute web research for {state.company} using these search queries: {', '.join(state.search_queries)}"
    )

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
    
    # Add conversation tracking - AI response with research findings
    ai_message = AIMessage(
        content=f"Completed web research for {state.company}. Found {len(deduplicated_search_docs)} unique sources and generated structured research notes covering the requested schema fields."
    )
    
    state_update = {
        "completed_notes": [str(result.content)],
        "messages": [user_message, ai_message]
    }
    if configurable.include_search_results:
        state_update["search_results"] = deduplicated_search_docs

    return state_update


def gather_notes_extract_schema(state: OverallState) -> dict[str, Any]:
    """Gather notes from the web search and extract the schema fields."""
    # Add conversation tracking - user request for schema extraction
    user_message = HumanMessage(
        content=f"Extract structured information for {state.company} from the research notes according to the defined schema."
    )

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
    
    # Add conversation tracking - AI response with extracted information
    ai_message = AIMessage(
        content=f"Successfully extracted structured information for {state.company} from research notes. Populated schema fields with available data and identified any missing information."
    )
    
    return {
        "info": result,
        "messages": [user_message, ai_message]
    }


def reflection(state: OverallState) -> dict[str, Any]:
    """Reflect on the extracted information and generate search queries to find missing information."""
    # Add conversation tracking - user request for reflection
    user_message = HumanMessage(
        content=f"Analyze the completeness of extracted information for {state.company} and determine if additional research is needed."
    )
    
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

    # Add conversation tracking - AI response with reflection results
    if result.is_satisfactory:
        ai_message = AIMessage(
            content=f"Research analysis complete for {state.company}. All required information has been successfully gathered and extracted according to the schema requirements."
        )
        return {
            "is_satisfactory": result.is_satisfactory,
            "messages": [user_message, ai_message]
        }
    else:
        ai_message = AIMessage(
            content=f"Research analysis for {state.company} identified missing information. Generated {len(result.search_queries)} additional search queries to fill gaps: {', '.join(result.search_queries)}"
        )
        return {
            "is_satisfactory": result.is_satisfactory,
            "search_queries": result.search_queries,
            "reflection_steps_taken": state.reflection_steps_taken + 1,
            "messages": [user_message, ai_message]
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


def route_to_summarization(
    state: OverallState, config: RunnableConfig
) -> Literal["summarize_conversation", "generate_queries"]:
    """Route the graph to summarization if token limits are exceeded, otherwise proceed to query generation."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    
    # Check if summarization is needed based on token limits
    if should_summarize(state, configurable):
        return "summarize_conversation"
    else:
        return "generate_queries"


def summarize_conversation(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Summarize conversation history to manage token limits while preserving key research findings.
    
    Args:
        state: Current state containing conversation history and research data
        config: Configuration containing summarization settings
        
    Returns:
        dict: Updated state with new summary and trimmed messages
    """
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    
    # If no messages to summarize, return unchanged state
    if not hasattr(state, 'messages') or not state.messages:
        return {}
    
    # Prepare messages for summarization
    messages_to_summarize = state.messages[:-configurable.messages_to_keep_after_summary] if len(state.messages) > configurable.messages_to_keep_after_summary else state.messages
    
    if not messages_to_summarize:
        return {}
    
    # Format conversation messages for the prompt
    conversation_messages_text = chr(10).join([f"{msg.__class__.__name__}: {msg.content}" for msg in messages_to_summarize])
    
    # Prepare the summarization prompt with appropriate context
    if state.summary:
        # Extend existing summary
        existing_summary_text = f"Current summary to extend:\n{state.summary}\n\nNew conversation messages to incorporate:"
    else:
        # Create new summary
        existing_summary_text = "Conversation messages to summarize:"
    
    # Use the SUMMARIZATION_PROMPT template
    summary_instruction = SUMMARIZATION_PROMPT.format(
        existing_summary=existing_summary_text,
        conversation_messages=conversation_messages_text
    )
    
    # Generate summary using Claude 3.5 Sonnet
    try:
        summary_messages = [
            {"role": "system", "content": summary_instruction},
            {"role": "user", "content": "Please create the summary as requested."}
        ]
        
        response = claude_3_5_sonnet.invoke(summary_messages)
        new_summary = response.content
        
        # Keep only the most recent messages after summarization
        recent_messages = state.messages[-configurable.messages_to_keep_after_summary:] if len(state.messages) > configurable.messages_to_keep_after_summary else []
        
        # Update token count for the new state
        new_token_count = count_conversation_tokens(recent_messages)
        
        return {
            "summary": new_summary,
            "messages": recent_messages,
            "total_tokens": new_token_count
        }
        
    except Exception as e:
        # If summarization fails, fall back to simple message trimming
        print(f"Warning: Summarization failed: {e}")
        recent_messages = state.messages[-configurable.messages_to_keep_after_summary:] if len(state.messages) > configurable.messages_to_keep_after_summary else state.messages
        new_token_count = count_conversation_tokens(recent_messages)
        
        return {
            "messages": recent_messages,
            "total_tokens": new_token_count
        }


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

# Add conditional routing from START to check for summarization needs
builder.add_conditional_edges(START, route_to_summarization)
builder.add_edge("summarize_conversation", "generate_queries")
builder.add_edge("generate_queries", "research_company")
builder.add_edge("research_company", "gather_notes_extract_schema")
builder.add_edge("gather_notes_extract_schema", "reflection")
builder.add_conditional_edges("reflection", route_from_reflection)

# Compile
graph = builder.compile()












