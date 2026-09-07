import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from tools import scrape_url, web_search

# Load environment variables (GROQ_API_KEY) from a local .env file
load_dotenv()

# Shared LLM instance used by every agent/chain below.
# Created once at import time, so it's reused rather than rebuilt per call.
llm = ChatGroq(
    model_name="openai/gpt-oss-120b", temperature=0.2, api_key=os.getenv("GROQ_API_KEY")
)


# First Agent
# Bound only to the web_search tool — its job is to find recent, relevant
# sources for a topic, not to read or write anything.
def build_search_agent():
    return create_agent(model=llm, tools=[web_search])


# seacond Agent
# Bound only to the scrape_url tool — its job is to pick a URL from the
# search results and pull its full text content for deeper context.
def build_reader_agent():
    return create_agent(model=llm, tools=[scrape_url])


# write chain
# A plain prompt -> llm -> parser chain (no tools). Takes the topic plus
# combined research text and turns it into a structured Markdown report.
writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert research writer. Write clear, structured and insightful reports.",
        ),
        (
            "human",
            """Write a detailed research report on the topic below
    
    Topic : {topic}
    Research Gathered:
    {research}
    
    Structure the report as :
    -Introduction
    -Key Findings (minimun 3 well-explain points)
    -conclusion
    -sources (list all URLs found in the research)
    
    Be detailed, factual and professional.""",
        ),
    ]
)

# StrOutputParser converts the LLM's raw response object into a plain string
writer_chain = writer_prompt | llm | StrOutputParser()

# critic chain
# Reviews the report written above and returns a score, strengths, and
# areas to improve, in a fixed text format the app can display as-is.
critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a sharp and constructive research critic. Be honest and specific.",
        ),
        (
            "human",
            """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...
one line verdict:
...""",
        ),
    ]
)

critic_chain = critic_prompt | llm | StrOutputParser()
