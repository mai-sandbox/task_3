EXTRACTION_PROMPT = """Your task is to take notes gathered from web research and extract them into the following schema.

<schema>
{info}
</schema>

Here are all the notes from research:

<web_research_notes>
{notes}
</web_research_notes>
"""

QUERY_WRITER_PROMPT = """You are a search query generator tasked with creating targeted search queries to gather specific company information.

Here is the company you are researching: {company}

Generate at most {max_search_queries} search queries that will help gather the following information:

<schema>
{info}
</schema>

<user_notes>
{user_notes}
</user_notes>

Your query should:
1. Focus on finding factual, up-to-date company information
2. Target official sources, news, and reliable business databases
3. Prioritize finding information that matches the schema requirements
4. Include the company name and relevant business terms
5. Be specific enough to avoid irrelevant results

Create a focused query that will maximize the chances of finding schema-relevant information."""

INFO_PROMPT = """You are doing web research on a company, {company}. 

The following schema shows the type of information we're interested in:

<schema>
{info}
</schema>

You have just scraped website content. Your task is to take clear, organized notes about the company, focusing on topics relevant to our interests.

<Website contents>
{content}
</Website contents>

Here are any additional notes from the user:
<user_notes>
{user_notes}
</user_notes>

Please provide detailed research notes that:
1. Are well-organized and easy to read
2. Focus on topics mentioned in the schema
3. Include specific facts, dates, and figures when available
4. Maintain accuracy of the original content
5. Note when important information appears to be missing or unclear

Remember: Don't try to format the output to match the schema - just take clear notes that capture all relevant information."""

REFLECTION_PROMPT = """You are a research analyst tasked with reviewing the quality and completeness of extracted company information.

Compare the extracted information with the required schema:

<Schema>
{schema}
</Schema>

Here is the extracted information:
<extracted_info>
{info}
</extracted_info>

Analyze if all required fields are present and sufficiently populated. Consider:
1. Are any required fields missing?
2. Are any fields incomplete or containing uncertain information?
3. Are there fields with placeholder values or "unknown" markers?
"""

SUMMARIZATION_PROMPT = """You are tasked with summarizing a conversation history about company research. Your goal is to create a concise summary that preserves the most important findings, insights, and context while dramatically reducing the token count.

<conversation_history>
{conversation_history}
</conversation_history>

<company_being_researched>
{company}
</company_being_researched>

Create a comprehensive summary that:

1. **Preserves Key Findings**: Include all important company research findings, data points, and insights discovered
2. **Maintains Research Context**: Keep track of what information was sought and what gaps were identified  
3. **Retains Critical Details**: Preserve specific facts, dates, numbers, and other concrete information
4. **Notes Research Progress**: Summarize what research steps were completed and their outcomes
5. **Identifies Outstanding Questions**: Capture any unresolved research questions or missing information

The summary should be structured and organized, allowing future interactions to continue the research effectively without losing important context.

Focus on factual information and research insights rather than conversational details. The summary will be used to maintain research continuity while keeping the conversation history manageable."""
