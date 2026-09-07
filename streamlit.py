import logging
import time

import streamlit as st

from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain

# Basic logger: full error details go here (visible in your terminal locally,
# or in the app logs on Streamlit Cloud) — never shown to the end user.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# How many times to silently retry a single agent call before giving up.
# Groq's gpt-oss models occasionally hallucinate a tool call that doesn't
# exist (a known, intermittent issue) — a quick retry usually succeeds.
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1.5


def invoke_with_retry(runnable, payload, step_name: str):
    """
    Call runnable.invoke(payload), retrying a couple of times on failure.

    This exists because some Groq models intermittently fail tool calls
    (e.g. hallucinating a tool name that wasn't provided). Most of the time
    a fresh attempt succeeds, so we retry quietly before surfacing an error.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return runnable.invoke(payload)
        except Exception as e:
            last_error = e
            logger.warning(
                "%s failed on attempt %d/%d: %s", step_name, attempt, MAX_RETRIES, e
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
    # All retries exhausted — re-raise so the outer try/except can handle it
    raise last_error


# Configure the browser tab title, favicon, and use the full page width
st.set_page_config(page_title="Multi-Agent Research System", page_icon="🔎", layout="wide")

# ----------------------------------------------------------------------------
# Theme / color layer (dark background + orange accents).
# This only affects visual styling — it does not change any app logic below.
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Page background and default text color */
    .stApp {
        background-color: #0d0d0f;
        color: #e6e6e9;
    }

    /* Main page title */
    h1 {
        color: #ffffff;
    }

    /* Text input box (research topic field) */
    div[data-testid="stTextInput"] input {
        background-color: #17171a;
        border: 1px solid #2a2a2e;
        border-radius: 8px;
        color: #e6e6e9;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #ff8a3d;
        box-shadow: 0 0 0 1px #ff8a3d;
    }

    /* Primary action button ("Start Research") */
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(90deg, #ff9a3d, #ff5f1f);
        border: none;
        color: #0a0a0a;
        font-weight: 700;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        filter: brightness(1.08);
    }

    /* Expander cards (Search Result / Scraped Content) */
    div[data-testid="stExpander"] {
        background-color: #141417;
        border: 1px solid #232326;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🔎 Multi-Agent Research System")
st.caption("Search → Read → Write → Critique — powered by LangChain agents + Groq")

# ----------------------------------------------------------------------------
# Session state
# Streamlit re-runs this whole script on every interaction (e.g. button click),
# so anything that needs to survive a re-run must live in st.session_state.
# ----------------------------------------------------------------------------
if "state" not in st.session_state:
    # Holds the pipeline's output: search_result, scraped_content, report, feedback
    st.session_state.state = {}
if "last_topic" not in st.session_state:
    # Remembers the topic used for the last completed run (used in the download filename)
    st.session_state.last_topic = ""

# ----------------------------------------------------------------------------
# User input
# ----------------------------------------------------------------------------
topic = st.text_input(
    "Enter a research topic",
    placeholder="e.g. impact of war on economy",
)
# Button is disabled until the user actually types a topic, to avoid empty runs
start = st.button("🚀 Start Research", type="primary", disabled=not topic.strip())

# ----------------------------------------------------------------------------
# Pipeline execution — runs only when the button is clicked with a valid topic
# ----------------------------------------------------------------------------
if start and topic.strip():
    state = {}  # local dict built up step by step, saved to session_state at the end
    try:
        # ---- Step 1: Search agent -------------------------------------------------
        # Uses the search_agent (LangChain agent bound to a web-search tool) to
        # gather recent, relevant information about the topic.
        with st.status("Step 1 — Searching for information...", expanded=True) as status:
            search_agent = build_search_agent()
            search_result = invoke_with_retry(
                search_agent,
                {
                    "messages": [
                        (
                            "user",
                            f"Find, recent, reliable and detailed information about: {topic}",
                        )
                    ]
                },
                step_name="Search agent",
            )
            # The agent returns a list of messages; the last one is its final answer
            state["search_result"] = search_result["messages"][-1].content
            status.update(label="Step 1 — Search complete ✅", state="complete")

        # Show the raw search output in a collapsible section (hidden by default)
        with st.expander("🔍 Search Result", expanded=False):
            st.markdown(state["search_result"])

        # ---- Step 2: Reader agent ---------------------------------------------------
        # Uses the reader_agent (LangChain agent bound to a scraping tool) to pick
        # the most relevant URL from the search results and pull its full content.
        with st.status("Step 2 — Reading & scraping top resource...", expanded=True) as status:
            reader_agent = build_reader_agent()
            reader_result = invoke_with_retry(
                reader_agent,
                {
                    "messages": [
                        (
                            "user",
                            (
                                f"Based on the following search results about '{topic}', "
                                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                                # Truncated to 800 chars to keep the prompt small
                                f"Search Results:\n{state['search_result'][:800]} "
                            ),
                        )
                    ]
                },
                step_name="Reader agent",
            )
            state["scraped_content"] = reader_result["messages"][-1].content
            status.update(label="Step 2 — Scraping complete ✅", state="complete")

        # Show the scraped page content in a collapsible section
        with st.expander("📄 Scraped Content", expanded=False):
            st.markdown(state["scraped_content"])

        # ---- Step 3: Writer chain ---------------------------------------------------
        # A plain LangChain prompt | llm | parser chain (no tools) that turns the
        # combined research into a structured report.
        with st.status("Step 3 — Drafting the report...", expanded=True) as status:
            research_combined = (
                f"SEARCH RESULT : \n {state['search_result']}\n\n"
                f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
            )
            state["report"] = invoke_with_retry(
                writer_chain,
                {"topic": topic, "research": research_combined},
                step_name="Writer chain",
            )
            status.update(label="Step 3 — Report drafted ✅", state="complete")

        # ---- Step 4: Critic chain ---------------------------------------------------
        # Reviews the drafted report and returns a score + strengths/weaknesses.
        with st.status("Step 4 — Critic reviewing the report...", expanded=True) as status:
            state["feedback"] = invoke_with_retry(
                critic_chain, {"report": state["report"]}, step_name="Critic chain"
            )
            status.update(label="Step 4 — Review complete ✅", state="complete")

        # Persist the finished pipeline output so it survives future re-runs
        # (e.g. when the user just expands/collapses a section afterwards)
        st.session_state.state = state
        st.session_state.last_topic = topic
        st.success("Research pipeline finished!")

    except Exception as e:
        # Log the full technical error (with traceback) for you/the developer —
        # this goes to the terminal / Streamlit Cloud logs, never to the user.
        logger.exception("Research pipeline failed for topic: %s", topic)

        # The end user only ever sees this friendly, non-technical message —
        # no stack traces, API error codes, or internal details are exposed.
        st.error(
            "😕 Something went wrong while generating your report. "
            "This can happen occasionally — please try again, or try a "
            "slightly different topic."
        )

# ----------------------------------------------------------------------------
# Results section
# Reads from session_state (not the local `state` above) so the report stays
# visible even after a re-run that isn't a fresh "Start Research" click.
# ----------------------------------------------------------------------------
if st.session_state.state.get("report"):
    st.divider()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📝 Final Report")
    with col2:
        # Lets the user save the report as a Markdown file, named after the topic
        st.download_button(
            "⬇️ Download Report",
            data=st.session_state.state["report"],
            file_name=f"{st.session_state.last_topic.replace(' ', '_')}_report.md",
            mime="text/markdown",
        )
    st.markdown(st.session_state.state["report"])

    st.subheader("🧐 Critic Feedback")
    st.markdown(st.session_state.state["feedback"])