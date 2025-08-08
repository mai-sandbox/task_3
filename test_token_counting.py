"""Test script for token counting functionality without API calls."""

from src.agent.conversation import ConversationManager
from langchain_core.messages import HumanMessage, AIMessage

def test_token_counting():
    """Test the token counting functionality."""
    
    # Initialize conversation manager
    manager = ConversationManager(
        max_tokens=2000,
        summary_trigger_ratio=0.8,
        preserve_recent_messages=3
    )
    
    # Create sample messages
    messages = [
        HumanMessage(content="Research company XYZ"),
        AIMessage(content="Found information about XYZ Corporation"),
        HumanMessage(content="What is their funding history?"),
        AIMessage(content="XYZ raised $10M in Series A funding in 2023"),
    ]
    
    # Test token counting
    print("Testing token counting functionality:\n")
    
    for i, msg in enumerate(messages):
        tokens = manager.count_message_tokens(msg)
        print(f"Message {i+1} ({msg.__class__.__name__}): {tokens} tokens")
        print(f"  Content: {msg.content[:50]}...")
    
    print(f"\nTotal conversation tokens: {manager.count_conversation_tokens(messages)}")
    
    # Test with longer messages
    long_messages = []
    for i in range(15):
        long_messages.append(HumanMessage(
            content=f"Query {i}: Please find detailed information about the company's products, services, founding team, funding history, market position, competitors, and growth strategy."
        ))
        long_messages.append(AIMessage(
            content=f"Response {i}: The company offers multiple product lines including enterprise software, cloud services, and consulting. Founded by experienced entrepreneurs with backgrounds in tech and finance. They've raised significant funding through multiple rounds and are positioned as a market leader in their segment."
        ))
    
    print(f"\nLong conversation with {len(long_messages)} messages:")
    total_tokens = manager.count_conversation_tokens(long_messages)
    print(f"  Total tokens: {total_tokens}")
    print(f"  Max tokens: {manager.max_tokens}")
    print(f"  Trigger threshold: {manager.trigger_tokens} tokens")
    print(f"  Should summarize: {manager.should_summarize(long_messages)}")
    
    # Test conversation stats
    stats = manager.get_conversation_stats(long_messages)
    print(f"\nConversation statistics:")
    print(f"  Messages: {stats['total_messages']}")
    print(f"  Token usage: {stats['token_usage_ratio']:.1%}")
    print(f"  Tokens until summarization: {stats['tokens_until_summarization']}")

if __name__ == "__main__":
    test_token_counting()