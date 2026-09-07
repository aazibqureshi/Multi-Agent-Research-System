from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from rich import print

# Load environment variables (TAVILY_API_KEY) from a local .env file
load_dotenv()

# Tavily client used for web search — needs TAVILY_API_KEY set in .env
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# First Tool using tavily for fetching current news from websites
@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Return Titles, URLs and snippets."""
    # Ask Tavily for up to 5 relevant results for the query
    results = tavily.search(query=query, max_results=5)  

    # Format each result into a readable "Title / URL / Snippet" block
    out = []
    for r in results['results']:
        out.append(
            # Snippet is truncated to the first 300 characters to keep the
            # agent's context small
            f"Title: {r["title"]}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n"
        )

    # Join all formatted results into one string, separated by dashed lines,
    # so the LLM sees a clear boundary between different search hits
    return "\n----\n".join(out)


# print(web_search.invoke(""))  # manual test call, left here for quick debugging


# Second Tool
@tool
def scrape_url(url: str) -> str:
    """Scrape and retrun clean text content from a given URL for deeper reading."""
    try:
        # Fetch the page; a browser-like User-Agent avoids some basic bot blocks
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozila/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")

        # Strip out non-content tags (scripts, styles, nav, footer) before
        # extracting text, so the output isn't full of JS/CSS/menu noise.
        # `return` is OUTSIDE the loop so every matching tag gets removed
        # first, and only then is the cleaned text returned.
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator="", strip=True)[:300]
    except Exception as e:
        # Any network/parsing error is returned as text instead of raising,
        # so a single bad URL doesn't crash the calling agent
        return f"Could not scrape :URL {str(e)}"


# print(scrape_url.invoke(''))  # manual test call, left here for quick debugging