import streamlit as st

from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain

st.set_page_config(page_title="Multi-Agent Research System", page_icon="🔎", layout="wide")

st.title("🔎 Multi-Agent Research System")
st.caption("Search → Read → Write → Critique — powered by LangChain agents + Groq")

if "state" not in st.session_state:
    st.session_state.state = {}
if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""

topic = st.text_input(
    "Enter a research topic",
    placeholder="e.g. impact of war on economy",
)
start = st.button("🚀 Start Research", type="primary", disabled=not topic.strip())

if start and topic.strip():
    state = {}
    try:
        # Step 1 - Search agent
        with st.status("Step 1 — Searching for information...", expanded=True) as status:
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
            state["search_result"] = search_result["messages"][-1].content
            status.update(label="Step 1 — Search complete ✅", state="complete")

        with st.expander("🔍 Search Result", expanded=False):
            st.markdown(state["search_result"])

        # Step 2 - Reader agent
        with st.status("Step 2 — Reading & scraping top resource...", expanded=True) as status:
            reader_agent = build_reader_agent()
            reader_result = reader_agent.invoke(
                {
                    "messages": [
                        (
                            "user",
                            (
                                f"Based on the following search results about '{topic}', "
                                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                                f"Search Results:\n{state['search_result'][:800]} "
                            ),
                        )
                    ]
                }
            )
            state["scraped_content"] = reader_result["messages"][-1].content
            status.update(label="Step 2 — Scraping complete ✅", state="complete")

        with st.expander("📄 Scraped Content", expanded=False):
            st.markdown(state["scraped_content"])

        # Step 3 - Writer chain
        with st.status("Step 3 — Drafting the report...", expanded=True) as status:
            research_combined = (
                f"SEARCH RESULT : \n {state['search_result']}\n\n"
                f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
            )
            state["report"] = writer_chain.invoke(
                {"topic": topic, "research": research_combined}
            )
            status.update(label="Step 3 — Report drafted ✅", state="complete")

        # Step 4 - Critic chain
        with st.status("Step 4 — Critic reviewing the report...", expanded=True) as status:
            state["feedback"] = critic_chain.invoke({"report": state["report"]})
            status.update(label="Step 4 — Review complete ✅", state="complete")

        st.session_state.state = state
        st.session_state.last_topic = topic
        st.success("Research pipeline finished!")

    except Exception as e:
        st.error(f"⚠️ Pipeline failed: {e}")

# Persisted results (survive reruns / widget interactions)
if st.session_state.state.get("report"):
    st.divider()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📝 Final Report")
    with col2:
        st.download_button(
            "⬇️ Download Report",
            data=st.session_state.state["report"],
            file_name=f"{st.session_state.last_topic.replace(' ', '_')}_report.md",
            mime="text/markdown",
        )
    st.markdown(st.session_state.state["report"])

    st.subheader("🧐 Critic Feedback")
    st.markdown(st.session_state.state["feedback"])