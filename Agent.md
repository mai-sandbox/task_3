# LangGraph Agent Development Guide

## Core Requirements

**Always use LangGraph** for building agents. All agent implementations must:

- Export the compiled graph as `app` from `./agent.py`
- Include a `langgraph.json` configuration file
- Be deployment-ready

## File Structure
```
./agent.py          # Main agent file, exports: app
./langgraph.json    # LangGraph configuration
```

## Export Pattern
```python
# agent.py
from langgraph import StateGraph

# ... your agent implementation ...

app = graph.compile()  # Export compiled graph as 'app'
```

## Documentation References

- **LangGraph Python**: https://langchain-ai.github.io/langgraph/llms.txt
- **LangGraph JS**: https://langchain-ai.github.io/langgraphjs/llms.txt  
- **LangChain Python**: https://python.langchain.com/llms.txt
- **LangChain JS**: https://js.langchain.com/llms.txt

Refer to documentation for implementation details, best practices, and deployment configurations.