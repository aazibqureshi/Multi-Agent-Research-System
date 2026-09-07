from agents import build_search_agent, build_reader_agent, critic_chain, writer_chain


def run_research_pipeline(topic: str) -> dict:
    """
    Run the full 4-step research pipeline for a single topic:
    search -> read/scrape -> write report -> critique report.

    Returns a dict with keys: search_result, scraped_content, report, feedback.
    """
    state = {}

    # ---- Step 1: Search agent --------------------------------------------------
    # Gathers recent, relevant web information about the topic using the
    # search agent (LLM + web_search tool).
    print("\n" + " =" * 50)
    print("step 1 - search agent is working ...")
    print("=" * 50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Find, recent, reliable and detailed information about: {topic}",
                )
            ]
        }
    )
    # Agent responses are a list of messages; the last one is the final answer
    state["search_result"] = search_result["messages"][-1].content

    print("\n search result", state["search_result"])

    # ---- Step 2: Reader agent ---------------------------------------------------
    # Picks the most relevant URL from the search results and scrapes it for
    # deeper content, using the reader agent (LLM + scrape_url tool).
    print("\n" + " =" * 50)
    print("step 2 - Reader agent is scraping top resources...")
    print("=" * 50)

    reader_agent = build_reader_agent()
    # reader_agent()  # not callable directly — must use .invoke() below
    reader_result = reader_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    (
                        f"Based on the following seaech results about '{topic}',"
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        # Only the first 800 chars are passed in, to keep the
                        # prompt short — full search_result is still saved above
                        f"Search Results:\n{state['search_result'][:800]} "
                    ),
                )
            ]
        }
    )
    state["scraped_content"] = reader_result["messages"][-1].content
    print("\nscraped content: \n", state["scraped_content"])

    # ---- Step 3: Writer chain ---------------------------------------------------
    # Combines the search result and scraped content into one research blob,
    # then asks the writer_chain (prompt | llm | parser, no tools) to draft
    # a structured report from it.
    print("\n" + " =" * 50)
    print("step 3 - Writer is drafting the report ....")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULT : \n {state['search_result']}\n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )
    state["report"] = writer_chain.invoke(
        {"topic": topic, "research": research_combined}
    )
    print("\n Final Report\n", state["report"])

    # ---- Step 4: Critic chain ---------------------------------------------------
    # Sends the finished report to the critic_chain, which scores it and
    # lists strengths / areas to improve.
    print("\n" + " =" * 50)
    print("step 4 - Critic is reviewing the report ....")
    print("=" * 50)

    critic_result = critic_chain.invoke({"report": state["report"]})
    state["feedback"] = critic_result  
    print("\n critic report \n", state["feedback"])

    return state


if __name__ == "__main__":
    # Simple CLI entry point: ask for a topic, run the full pipeline, and
    # let the print statements above show progress in the terminal.
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)