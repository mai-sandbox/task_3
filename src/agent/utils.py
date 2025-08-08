def deduplicate_sources(search_response: dict | list[dict]) -> list[dict]:
    """
    Takes either a single search response or list of responses from Tavily API and de-duplicates them based on the URL.

    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results

    Returns:
        str: Formatted string with deduplicated sources
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response["results"]
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and "results" in response:
                sources_list.extend(response["results"])
            else:
                sources_list.extend(response)
    else:
        raise ValueError(
            "Input must be either a dict with 'results' or a list of search results"
        )

    # Deduplicate by URL
    unique_urls = set()
    unique_sources_list = []
    for source in sources_list:
        if source["url"] not in unique_urls:
            unique_urls.add(source["url"])
            unique_sources_list.append(source)

    return unique_sources_list


def format_sources(
    sources_list: list[dict],
    include_raw_content: bool = True,
    max_tokens_per_source: int = 1000,
) -> str:
    """
    Takes a list of unique results from Tavily API and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.

    Args:
        sources_list: list of unique results from Tavily API
        max_tokens_per_source: int, maximum number of tokens per each search result to include in the formatted string
        include_raw_content: bool, whether to include the raw_content from Tavily in the formatted string

    Returns:
        str: Formatted string with deduplicated sources
    """
    # Format output
    formatted_text = "Sources:\n\n"
    for source in sources_list:
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += (
            f"Most relevant content from source: {source['content']}\n===\n"
        )
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get("raw_content", "")
            if raw_content is None:
                raw_content = ""
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"

    return formatted_text.strip()


def format_all_notes(completed_notes: list[str]) -> str:
    """Format a list of notes into a string"""
    formatted_str = ""
    for idx, company_notes in enumerate(completed_notes, 1):
        formatted_str += f"""
{'='*60}
Note: {idx}:
{'='*60}
Notes from research:
{company_notes}"""
    return formatted_str


# Conversation management utilities

import tiktoken
from typing import TYPE_CHECKING
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

if TYPE_CHECKING:
    from agent.state import OverallState
    from agent.configuration import Configuration


def count_conversation_tokens(messages: list[BaseMessage]) -> int:
    """
    Count the total number of tokens in a list of messages using tiktoken.
    
    Args:
        messages: List of BaseMessage objects to count tokens for
        
    Returns:
        int: Total token count for all messages
    """
    if not messages:
        return 0
    
    # Use cl100k_base encoding (used by GPT-4 and Claude models)
    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    
    for message in messages:
        # Count tokens for message content
        if hasattr(message, 'content') and message.content:
            total_tokens += len(encoding.encode(str(message.content)))
        
        # Add tokens for message metadata (role, etc.)
        # Approximate 4 tokens per message for metadata
        total_tokens += 4
    
    return total_tokens


def should_summarize(state: "OverallState", config: "Configuration") -> bool:
    """
    Check if conversation summarization should be triggered based on token limits.
    
    Args:
        state: OverallState object containing conversation history
        config: Configuration object with token limits
        
    Returns:
        bool: True if summarization should be triggered, False otherwise
    """
    if not hasattr(state, 'messages') or not state.messages:
        return False
    
    # Count current tokens in conversation
    current_tokens = count_conversation_tokens(state.messages)
    
    # Check if we've exceeded the summarization trigger threshold
    return current_tokens >= config.summarization_trigger_tokens


def create_conversation_messages(state: "OverallState") -> list[BaseMessage]:
    """
    Convert state data into message format for LLM interactions.
    
    Args:
        state: OverallState object containing conversation data
        
    Returns:
        list[BaseMessage]: List of messages formatted for LLM consumption
    """
    messages = []
    
    # Add existing conversation history
    if hasattr(state, 'messages') and state.messages:
        messages.extend(state.messages)
    
    # Add summary as a system-like message if it exists
    if hasattr(state, 'summary') and state.summary:
        summary_message = HumanMessage(
            content=f"Previous conversation summary: {state.summary}"
        )
        # Insert summary at the beginning after any existing messages
        messages.insert(0, summary_message)
    
    return messages




