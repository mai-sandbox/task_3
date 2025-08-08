"""Test script for conversation summarization functionality."""

import asyncio
from src.agent.conversation import ConversationManager
from langchain_core.messages import HumanMessage, AIMessage

async def test_summarization():
    """Test the conversation summarization functionality."""
    
    # Initialize conversation manager with lower limits for testing
    manager = ConversationManager(
        max_tokens=1000,  # Lower limit for testing
        summary_trigger_ratio=0.8,
        preserve_recent_messages=2
    )
    
    # Create sample conversation history
    messages = []
    
    # Add multiple messages to trigger summarization
    for i in range(10):
        messages.append(HumanMessage(content=f"Research query {i}: Find information about company XYZ's product line, funding history, and founding team members."))
        messages.append(AIMessage(content=f"Response {i}: Found the following information about company XYZ: Product line includes software solutions for data analytics. Founded in 2020 by John Doe and Jane Smith. Raised $10M in Series A funding from Venture Capital Firm ABC."))
    
    # Check initial stats
    print("Initial conversation stats:")
    stats = manager.get_conversation_stats(messages)
    print(f"  Total messages: {stats['total_messages']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Should summarize: {stats['should_summarize']}")
    print()
    
    # Test summarization
    context = {
        "company": "XYZ Corporation",
        "extraction_schema": {
            "company_name": "XYZ Corporation",
            "founding_year": 2020,
            "founders": ["John Doe", "Jane Smith"],
            "product_description": "Software solutions for data analytics",
            "funding_summary": "$10M Series A"
        },
        "info": {}
    }
    
    print("Testing conversation management...")
    managed_messages = await manager.manage_conversation_history(messages, context)
    
    print(f"\nAfter management:")
    print(f"  Original message count: {len(messages)}")
    print(f"  Managed message count: {len(managed_messages)}")
    
    if len(managed_messages) < len(messages):
        print("\n✅ Summarization triggered successfully!")
        print("\nFirst message in managed history:")
        print(f"  Type: {managed_messages[0].__class__.__name__}")
        print(f"  Content preview: {managed_messages[0].content[:200]}...")
    else:
        print("\n❌ Summarization not triggered (may need to adjust token limits)")
    
    # Check final stats
    print("\nFinal conversation stats:")
    final_stats = manager.get_conversation_stats(managed_messages)
    print(f"  Total messages: {final_stats['total_messages']}")
    print(f"  Total tokens: {final_stats['total_tokens']}")
    print(f"  Token usage ratio: {final_stats['token_usage_ratio']:.1%}")

if __name__ == "__main__":
    asyncio.run(test_summarization())