import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

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
        append_assistant_message(
            {
                "role": "assistant",
                "kind": "text",
                "content": (
                    "I can analyze margins, assess customer risk, show market rate trends, and help craft negotiation strategies. "
                    "Try: 'Show me margin analysis' or 'Customer risk assessment'."
                ),
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