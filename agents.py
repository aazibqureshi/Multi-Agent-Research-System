import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from tools import scrape_url, web_search

load_dotenv()

llm = ChatGroq(
    model_name="openai/gpt-oss-120b", temperature=0.7, api_key=os.getenv("GROQ_API_KEY")
)


# First Agent
def build_search_agent():
    return create_agent(model=llm, tools=[web_search])


# seacond Agent


def build_reader_agent():
    return create_agent(model=llm, tools=[scrape_url])


# write chain

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

writer_chain = writer_prompt | llm | StrOutputParser()

# critic chain

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