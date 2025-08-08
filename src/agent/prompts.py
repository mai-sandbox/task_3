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

SUMMARIZATION_PROMPT = """You are tasked with creating or extending a conversation summary for a company research session.

{existing_summary}

{conversation_messages}

Create a comprehensive summary that:
1. **Preserves all key company research findings and insights** - Never lose important business information, financial data, strategic details, or research discoveries
2. **Maintains conversation flow and context** - Keep track of the research progression, decision-making process, and how the investigation evolved
3. **Keeps important details** - Retain specific facts, dates, figures, sources, methodological notes, and any quantitative data discovered
4. **Focuses on company-specific information** - Prioritize findings about the target company, its business model, market position, and competitive landscape
5. **Organizes information clearly** - Structure the summary in a logical, easy-to-follow format with clear sections for different types of findings
6. **Tracks research progress** - Note what has been discovered, what information gaps remain, what areas need further investigation, and what research strategies were employed
7. **Consolidates without losing critical details** - Combine related information while preserving nuance, specificity, and the context in which information was discovered
8. **Maintains research quality indicators** - Note the reliability of sources, confidence levels in findings, and any uncertainties or conflicting information

The summary should be comprehensive enough that someone could understand the full scope of research conducted and findings discovered without needing to review the original conversation.

Provide only the updated summary, no additional commentary or meta-discussion."""


