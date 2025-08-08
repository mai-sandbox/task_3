"""Test edge case where summarization should trigger."""

from src.agent.conversation import ConversationManager
from langchain_core.messages import HumanMessage, AIMessage

def test_summarization_trigger():
    """Test that summarization triggers at the right threshold."""
    
    # Initialize with very low limits for testing
    manager = ConversationManager(
        max_tokens=500,  # Very low for testing
        summary_trigger_ratio=0.8,  # Trigger at 400 tokens
        preserve_recent_messages=2
    )
    
    # Create messages that will exceed the limit
    messages = []
    for i in range(20):
        messages.append(HumanMessage(
            content=f"Query {i}: Find comprehensive information about the company including all details about products, services, team, and funding."
        ))
        messages.append(AIMessage(
            content=f"Response {i}: Found extensive information about the company's operations, products, services, leadership team, and funding rounds."
        ))
    
    # Check stats
    stats = manager.get_conversation_stats(messages)
    print("Conversation Statistics:")
    print(f"  Total messages: {stats['total_messages']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Max tokens allowed: {manager.max_tokens}")
    print(f"  Trigger threshold: {manager.trigger_tokens}")
    print(f"  Token usage ratio: {stats['token_usage_ratio']:.1%}")
    print(f"  Should summarize: {stats['should_summarize']}")
    
    if stats['should_summarize']:
        print("\n✅ Summarization would be triggered!")
        print(f"   - Would preserve {manager.preserve_recent_messages} recent messages")
        print(f"   - Would summarize {len(messages) - manager.preserve_recent_messages} older messages")
    else:
        print("\n❌ Summarization not triggered")

if __name__ == "__main__":
    test_summarization_trigger()