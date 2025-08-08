Add dynamic summarization to the company researcher agent. When the conversation history goes over 20k tokens, automatically summarize and remove the oldest messages to keep things manageable.
The summarization should:

Preserve key company research findings and insights
Keep the most recent context intact
Maintain conversation flow and important details
Trigger automatically when approaching token limits

Implement this within the existing LangGraph state management - probably best to add a check in the state update logic that monitors message count/tokens and calls a summarization node when needed.