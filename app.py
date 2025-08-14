import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any

st.set_page_config(
    page_title="ShipNegotiate AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "kind": "text",
                "content": (
                    "Hi Christopher! I'm Chris, your AI negotiation assistant. I have access to all your "
                    "contract data, market intelligence, and customer insights. I can help you analyze margins, "
                    "assess customer risks, benchmark rates, and develop negotiation strategies. What would you like to explore today?"
                ),
            }
        ]
    if "pending_user_prompt" not in st.session_state:
        st.session_state.pending_user_prompt = None


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### ShipNegotiate AI")

        st.markdown("**Portfolio Overview**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Active Contracts", value="247")
            st.metric("Renewals Due", value="23")
        with col_b:
            st.metric("Avg Margin", value="16.8%")
            st.metric("High Risk Accounts", value="5")

        st.divider()
        st.markdown("**Data Sources**")
        data_sources = [
            ("Contract Management System", "#22c55e"),
            ("Financial Database", "#22c55e"),
            ("Market Intelligence Feed", "#22c55e"),
            ("Credit Rating Service", "#22c55e"),
            ("Commodity Exchange Data", "#ef4444"),
        ]
        for name, color in data_sources:
            st.markdown(
                f"<span style='color:{color};font-size:18px;'>●</span> {name}",
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("**Recent Alerts**")
        st.info("Credit Rating Downgrade — GlobalTrade Corp downgraded to BBB" )

        # LLM config quick view (uses env vars or Streamlit secrets)
        with st.expander("LLM configuration"):
            provider = os.getenv("LLM_PROVIDER", "openai")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
            st.write({"provider": provider, "base_url": base_url, "model": model_name})
            st.caption("Set OPENAI_API_KEY (and OPENAI_BASE_URL for Databricks) in env vars or .streamlit/secrets.toml")


def render_header() -> None:
    st.markdown("### Chris - Your Negotiation Assistant")
    st.caption("Online • Ready to help with contract insights")


def create_margin_dataframe() -> pd.DataFrame:
    data = [
        {"Customer": "GlobalTrade Corp", "Revenue (M)": 8.5, "COGS (M)": 6.6, "Last Renewal": "2024-11-01"},
        {"Customer": "Pacific Logistics", "Revenue (M)": 6.2, "COGS (M)": 5.1, "Last Renewal": "2024-07-15"},
        {"Customer": "Apex Retail Group", "Revenue (M)": 4.9, "COGS (M)": 3.9, "Last Renewal": "2025-01-30"},
        {"Customer": "EuroChem Ltd", "Revenue (M)": 7.3, "COGS (M)": 6.3, "Last Renewal": "2024-12-10"},
        {"Customer": "NorthStar Foods", "Revenue (M)": 5.4, "COGS (M)": 4.4, "Last Renewal": "2025-02-12"},
    ]
    df = pd.DataFrame(data)
    df["Margin %"] = ((df["Revenue (M)"] - df["COGS (M)"]) / df["Revenue (M)"]) * 100
    return df


def create_risk_dataframe() -> pd.DataFrame:
    data = [
        {"Customer": "GlobalTrade Corp", "Credit Rating": "BBB", "DSO": 62, "On-time %": 88, "Risk": "High"},
        {"Customer": "Pacific Logistics", "Credit Rating": "A-", "DSO": 45, "On-time %": 95, "Risk": "Medium"},
        {"Customer": "Apex Retail Group", "Credit Rating": "A", "DSO": 38, "On-time %": 97, "Risk": "Low"},
        {"Customer": "EuroChem Ltd", "Credit Rating": "BBB+", "DSO": 50, "On-time %": 92, "Risk": "Medium"},
        {"Customer": "NorthStar Foods", "Credit Rating": "BBB", "DSO": 58, "On-time %": 90, "Risk": "Medium"},
    ]
    return pd.DataFrame(data)


def create_rate_trend_dataframe(periods: int = 12) -> pd.DataFrame:
    start_date = datetime.today().date().replace(day=1)
    dates = [start_date - pd.offsets.MonthBegin(n) for n in range(periods - 1, -1, -1)]
    market_rate = [1.85, 1.9, 1.95, 1.98, 2.05, 2.1, 2.08, 2.12, 2.2, 2.25, 2.22, 2.28]
    your_rate = [1.95, 1.97, 2.02, 2.04, 2.12, 2.18, 2.13, 2.17, 2.26, 2.3, 2.27, 2.35]
    df = pd.DataFrame({"Month": dates, "Market Benchmark Rate": market_rate, "Your Avg Rate": your_rate})
    return df


def append_assistant_message(message: dict) -> None:
    st.session_state.messages.append(message)


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("kind") == "margin_analysis":
            st.write(message["content"])
            df = pd.DataFrame(message["data"]["rows"])
            st.dataframe(df, use_container_width=True)
            chart_df = df.set_index("Customer")["Margin %"]
            st.bar_chart(chart_df)
        elif message.get("kind") == "risk_assessment":
            st.write(message["content"])
            df = pd.DataFrame(message["data"]["rows"])
            st.dataframe(df, use_container_width=True)
        elif message.get("kind") == "rate_trends":
            st.write(message["content"])
            df = pd.DataFrame(message["data"]["rows"])
            fig = px.line(df, x="Month", y=["Market Benchmark Rate", "Your Avg Rate"], markers=True)
            fig.update_layout(height=360, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(message["content"])


# ---------- LLM integration ----------

def _get_secret_env(name: str, default: str | None = None) -> str | None:
    if name in st.secrets:
        try:
            return st.secrets[name]
        except Exception:
            pass
    return os.getenv(name, default)


def _build_system_prompt() -> str:
    return (
        "You are Chris, an AI contract negotiation assistant. Be concise, data-driven, and action-oriented. "
        "When appropriate, suggest concrete next steps, thresholds, and negotiation levers (rate, term, indexation, volume, surcharges)."
    )


def _convert_history_to_openai_messages(history: List[Dict[str, Any]], user_input: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": _build_system_prompt()}]
    for msg in history[-8:]:  # last few for context
        if msg.get("kind") == "text" and msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})
    return messages


def generate_llm_response(user_input: str) -> str:
    api_key = _get_secret_env("OPENAI_API_KEY")
    base_url = _get_secret_env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = _get_secret_env("LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        return (
            "LLM is not configured. Please set OPENAI_API_KEY (and OPENAI_BASE_URL for Databricks). "
            "See README for details."
        )

    try:
        from openai import OpenAI
    except Exception:
        return "The 'openai' package is not installed. Run: pip install openai"

    client = OpenAI(api_key=api_key, base_url=base_url)
    chat_messages = _convert_history_to_openai_messages(st.session_state.messages, user_input)

    with st.spinner("Thinking…"):
        response = client.chat.completions.create(
            model=model_name,
            messages=chat_messages,
            temperature=0.2,
            max_tokens=700,
        )
    return response.choices[0].message.content or ""


def handle_prompt(prompt_text: str) -> None:
    append_assistant_message({"role": "user", "kind": "text", "content": prompt_text})

    lower = prompt_text.lower()

    if "margin" in lower:
        df = create_margin_dataframe()
        append_assistant_message(
            {
                "role": "assistant",
                "kind": "margin_analysis",
                "content": "Here is a quick margin analysis for top accounts:",
                "data": {"rows": df.to_dict(orient="records")},
            }
        )
    elif "risk" in lower:
        df = create_risk_dataframe()
        append_assistant_message(
            {
                "role": "assistant",
                "kind": "risk_assessment",
                "content": "Customer risk assessment based on credit rating, DSO, and payment behavior:",
                "data": {"rows": df.to_dict(orient="records")},
            }
        )
    elif "rate" in lower or "market" in lower or "trend" in lower:
        df = create_rate_trend_dataframe()
        append_assistant_message(
            {
                "role": "assistant",
                "kind": "rate_trends",
                "content": "Market benchmark rates vs your average realized rates:",
                "data": {"rows": df.to_dict(orient="records")},
            }
        )
    else:
        llm_text = generate_llm_response(prompt_text)
        append_assistant_message(
            {
                "role": "assistant",
                "kind": "text",
                "content": llm_text,
            }
        )


initialize_session_state()
render_sidebar()
render_header()

for message in st.session_state.messages:
    render_message(message)

cta_cols = st.columns([1, 1, 1])
with cta_cols[0]:
    if st.button("Show me margin analysis"):
        st.session_state.pending_user_prompt = "Show me margin analysis"
with cta_cols[1]:
    if st.button("Customer risk assessment"):
        st.session_state.pending_user_prompt = "Customer risk assessment"
with cta_cols[2]:
    if st.button("Market rate trends"):
        st.session_state.pending_user_prompt = "Market rate trends"

user_text = st.chat_input("Ask about customers, margins, rates, negotiations…")
prompt_to_use = user_text or st.session_state.pending_user_prompt
if prompt_to_use:
    st.session_state.pending_user_prompt = None
    handle_prompt(prompt_to_use)
    st.rerun()  # ensure message renders in history