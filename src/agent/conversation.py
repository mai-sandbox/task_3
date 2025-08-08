"""Conversation history management with dynamic summarization."""

from typing import Any, List, Dict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
import tiktoken


class ConversationManager:
    """Manages conversation history with automatic summarization."""
    
    def __init__(
        self,
        max_tokens: int = 20000,
        summary_trigger_ratio: float = 0.8,
        preserve_recent_messages: int = 5,
    ):
        """Initialize the conversation manager.
        
        Args:
            max_tokens: Maximum number of tokens before triggering summarization
            summary_trigger_ratio: Trigger summarization when reaching this ratio of max_tokens
            preserve_recent_messages: Number of recent messages to preserve unsummarized
        """
        self.max_tokens = max_tokens
        self.trigger_tokens = int(max_tokens * summary_trigger_ratio)
        self.preserve_recent_messages = preserve_recent_messages
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.summarizer = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self.encoding.encode(text))
    
    def count_message_tokens(self, message: BaseMessage) -> int:
        """Count tokens in a message."""
        content = str(message.content)
        # Add overhead for message structure
        return self.count_tokens(content) + 10
    
    def count_conversation_tokens(self, messages: List[BaseMessage]) -> int:
        """Count total tokens in conversation history."""
        return sum(self.count_message_tokens(msg) for msg in messages)
    
    def should_summarize(self, messages: List[BaseMessage]) -> bool:
        """Check if conversation should be summarized."""
        if len(messages) <= self.preserve_recent_messages:
            return False
        
        total_tokens = self.count_conversation_tokens(messages)
        return total_tokens >= self.trigger_tokens
    
    async def summarize_messages(
        self, 
        messages: List[BaseMessage],
        context: Dict[str, Any]
    ) -> BaseMessage:
        """Summarize a list of messages preserving key information.
        
        Args:
            messages: Messages to summarize
            context: Additional context (company info, research findings, etc.)
        """
        # Build the summarization prompt
        conversation_text = "\n".join([
            f"{msg.__class__.__name__}: {msg.content}"
            for msg in messages
        ])
        
        summarization_prompt = f"""Summarize the following conversation history from a company research session.
        
Company being researched: {context.get('company', 'Unknown')}

Key Requirements for Summary:
1. Preserve ALL company research findings and insights
2. Keep all discovered facts about the company (founding year, founders, products, funding, etc.)
3. Maintain important context about what information was searched for
4. Note any gaps in information that still need to be filled
5. Keep the summary concise but comprehensive

Conversation to summarize:
{conversation_text}

Current extraction schema being filled:
{context.get('extraction_schema', {})}

Already gathered information:
{context.get('info', {})}

Provide a comprehensive summary that preserves all critical research findings:"""

        result = await self.summarizer.ainvoke(summarization_prompt)
        
        # Return as a system message to distinguish from regular messages
        return SystemMessage(
            content=f"[SUMMARIZED HISTORY]: {result.content}"
        )
    
    async def manage_conversation_history(
        self,
        messages: List[BaseMessage],
        context: Dict[str, Any]
    ) -> List[BaseMessage]:
        """Manage conversation history with automatic summarization.
        
        Args:
            messages: Current conversation messages
            context: Additional context for summarization
            
        Returns:
            Managed conversation history with summarization if needed
        """
        if not self.should_summarize(messages):
            return messages
        
        # Split messages to summarize vs preserve
        if len(messages) <= self.preserve_recent_messages:
            return messages
            
        messages_to_summarize = messages[:-self.preserve_recent_messages]
        recent_messages = messages[-self.preserve_recent_messages:]
        
        # Generate summary
        summary = await self.summarize_messages(messages_to_summarize, context)
        
        # Return summary + recent messages
        return [summary] + recent_messages
    
    def get_conversation_stats(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Get statistics about the conversation."""
        total_tokens = self.count_conversation_tokens(messages)
        return {
            "total_messages": len(messages),
            "total_tokens": total_tokens,
            "token_usage_ratio": total_tokens / self.max_tokens,
            "should_summarize": self.should_summarize(messages),
            "tokens_until_summarization": max(0, self.trigger_tokens - total_tokens)
        }