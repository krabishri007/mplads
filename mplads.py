"""
=============================================================================
 MPLADS Transparency Dashboard + AI Risk Intelligence Layer — Streamlit
=============================================================================
This app is laid out like the public "Empowered Indian" MPLADS dashboard
(empoweredindian.in/mplads) — a national overview, a browsable list of
states, and per-MP profile pages — pulling the SAME live public data that
site uses (api.empoweredindian.in, no key required).

On top of that browsing experience, this adds the piece that site doesn't
have: an 🤖 AI RISK INTELLIGENCE LAYER — a priority queue of works and MPs
worth a closer look, with plain-English reasons and a verification
checklist. That layer is clearly marked throughout as our addition; its
scores come from the companion Colab notebook (Part 1), not from the live
site data.

Two independent data flows, don't confuse them:
  1. Site-like browsing (Home / States / MP Profiles) -> live public API,
     falls back to the local export if the API is unreachable.
  2. AI Risk Intelligence (Priority Queue / MP Fund Risk) -> always comes
     from ./data/risk_scores_*.json, produced by the Colab notebook.

Run locally with:
    streamlit run app.py

Folder layout expected:
    streamlit_app/
    ├── app.py
    └── data/
        ├── national_overview.json
        ├── risk_scores_works.json
        ├── risk_scores_mps.json
        ├── aggregate_by_state.json
        ├── aggregate_by_worktype.json
        └── meta_summary.json
=============================================================================
"""

# -----------------------------------------------------------------------
# STEP 1 — Imports & page config
# -----------------------------------------------------------------------
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(
    page_title="MPLADS Transparency + AI Risk Intelligence",
    page_icon="🏛️",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"
RISK_COLORS = {"High": "#d62728", "Medium": "#ff9f1c", "Low": "#2ca02c"}

# -----------------------------------------------------------------------
# STEP 2 — Light "government transparency site" styling
# -----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #0b3d91 0%, #1a56db 100%);
        padding: 2rem 2rem; border-radius: 12px; color: white; margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p { margin: .4rem 0 0 0; opacity: .9; }
    .state-card {
        border: 1px solid #e3e7ee; border-radius: 10px; padding: .9rem 1rem;
        background: #fafbfe; margin-bottom: .6rem;
    }
    .ai-banner {
        background: #fff4e5; border: 1px solid #ffb84d; border-radius: 10px;
        padding: 1rem 1.2rem; margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------
# STEP 3 — 🔌 INTEGRATION POINT #1: live site-like data (works today)
# -----------------------------------------------------------------------
# This is the same public, unauthenticated API the real site is built on —
# verified working. It gives us raw MPLADS data (not AI-scored) for the
# Home / States / MP Profiles pages.
PUBLIC_DATA_API_BASE_URL = "https://api.empoweredindian.in"

COLUMN_MAP_LIVE = {
    "mpName": "MP Name", "state": "State", "constituency": "Constituency", "house": "House",
    "allocatedAmount": "Allocated Amount (₹)", "totalExpenditure": "Total Expenditure (₹)",
    "utilizationPercentage": "Utilization %", "completionRate": "Completion Rate %",
    "completedWorksCount": "Completed Works", "recommendedWorksCount": "Recommended Works",
    "pendingWorks": "Pending Works", "unpaidBalance": "Balance Not Yet Paid to Vendors (₹)",
}


@st.cache_data(ttl=600)
def fetch_all_mp_summaries_live() -> pd.DataFrame:
    resp = requests.get(
        f"{PUBLIC_DATA_API_BASE_URL}/api/summary/mps",
        params={"page": 1, "limit": 800},
        timeout=20,
    )
    resp.raise_for_status()
    raw = pd.DataFrame(resp.json()["data"])
    return raw.rename(columns=COLUMN_MAP_LIVE)[list(COLUMN_MAP_LIVE.values())]


def _local_json(filename: str):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_browse_data() -> tuple[pd.DataFrame, str]:
    """Returns (dataframe, source_label) for the site-like browsing pages.
    Tries the live public API first (when toggled on); falls back to the
    local AI-scored MP export, which has the same core columns."""
    if st.session_state.get("use_live_data"):
        try:
            return fetch_all_mp_summaries_live(), "live"
        except Exception as e:
            st.sidebar.error(f"Live API unreachable, showing last known data.\n\n{e}")
    local = _local_json("risk_scores_mps.json")
    return (pd.DataFrame(local) if local else pd.DataFrame()), "local"


# -----------------------------------------------------------------------
# STEP 4 — 🔌 INTEGRATION POINT #2: your own risk-scoring API (fill in later)
# -----------------------------------------------------------------------
# Distinct from STEP 3 above. This is for when you wrap the trained model
# (Colab notebook) in your own backend serving pre-scored risk data. Until
# then, works_df / mps_df below load straight from the Colab JSON export.
RISK_API_BASE_URL = ""   # <- e.g. "https://your-risk-api.example.com/api/v1"
RISK_API_KEY = "AQ.Ab8RN6ISwEQ9wh8aui9ULghadwJJHyM_ew_n9S8TPI4xpbQ49Q"


def _risk_headers():
    return {"Authorization": f"Bearer {RISK_API_KEY}"} if RISK_API_KEY else {}


@st.cache_data(ttl=300)
def load_work_risk_scores() -> pd.DataFrame:
    if RISK_API_BASE_URL:
        r = requests.get(f"{RISK_API_BASE_URL}/risk/works", headers=_risk_headers(), timeout=30)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    data = _local_json("risk_scores_works.json")
    return pd.DataFrame(data) if data else pd.DataFrame()


@st.cache_data(ttl=300)
def load_mp_risk_scores() -> pd.DataFrame:
    if RISK_API_BASE_URL:
        r = requests.get(f"{RISK_API_BASE_URL}/risk/mps", headers=_risk_headers(), timeout=30)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    data = _local_json("risk_scores_mps.json")
    return pd.DataFrame(data) if data else pd.DataFrame()


@st.cache_data(ttl=300)
def load_chart_aggregates():
    return (
        pd.DataFrame(_local_json("aggregate_by_state.json") or []),
        pd.DataFrame(_local_json("aggregate_by_worktype.json") or []),
    )


@st.cache_data(ttl=300)
def load_meta() -> dict:
    return _local_json("meta_summary.json") or {}


def load_national_overview_static() -> dict:
    data = _local_json("national_overview.json")
    return data["data"] if data and "data" in data else (data or {})


# -----------------------------------------------------------------------
# STEP 5 — Load everything up front
# -----------------------------------------------------------------------
works_df = load_work_risk_scores()
mps_scored_df = load_mp_risk_scores()  # has AI risk fields
state_agg, worktype_agg = load_chart_aggregates()
meta = load_meta()

if "nav" not in st.session_state:
    st.session_state.nav = "🏠 Home"
if "selected_state" not in st.session_state:
    st.session_state.selected_state = None
if "selected_mp" not in st.session_state:
    st.session_state.selected_mp = None

# -----------------------------------------------------------------------
# STEP 6 — Sidebar: data source + nav
# -----------------------------------------------------------------------
st.sidebar.header("Data source")
st.sidebar.toggle(
    "🔄 Use live public API for browsing pages",
    key="use_live_data",
    help=f"Home / States / MP Profiles pull from {PUBLIC_DATA_API_BASE_URL} when on. "
         "The AI Risk Intelligence tab is unaffected either way — it always reads "
         "the Colab notebook's scored output.",
)
browse_df, browse_source = get_browse_data()

st.sidebar.header("Navigate")
nav_options = ["🏠 Home", "🗺️ Browse by State", "🏛️ MP Profiles", "🤖 AI Risk Intelligence — New", "📊 Analytics"]
st.session_state.nav = st.sidebar.radio("Go to", nav_options, index=nav_options.index(st.session_state.nav))

# -----------------------------------------------------------------------
# STEP 7 — Hero header
# -----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏛️ MPLADS Transparency Dashboard</h1>
        <p>Explore how MP Local Area Development funds are allocated, spent, and completed —
        now with an added AI layer that tells officials which works deserve a closer look, and why.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(f"Browsing data source: **{'Live public API' if browse_source == 'live' else 'Local snapshot'}**")

# =========================================================================
# PAGE: HOME
# =========================================================================
if st.session_state.nav == "🏠 Home":
    overview = fetch_all_mp_summaries_live() if browse_source == "live" else None
    if overview is not None and not overview.empty:
        total_allocated = overview["Allocated Amount (₹)"].sum()
        total_expenditure = overview["Total Expenditure (₹)"].sum()
        completion_rate = (
            overview["Completed Works"].sum()
            / max(overview["Completed Works"].sum() + overview["Pending Works"].sum(), 1) * 100
        )
        total_mps = len(overview)
        utilization = overview["Utilization %"].mean()
    else:
        static = load_national_overview_static()
        total_allocated = static.get("totalAllocated", 0)
        total_expenditure = static.get("totalExpenditure", 0)
        completion_rate = static.get("completionRate", 0)
        total_mps = static.get("totalMPs", 0)
        utilization = static.get("utilizationPercentage", 0)

    st.subheader("National Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Allocated", f"₹{total_allocated:,.0f}")
    c2.metric("Total Expenditure", f"₹{total_expenditure:,.0f}")
    c3.metric("Utilisation %", f"{utilization:.1f}%")
    c4.metric("Completion Rate", f"{completion_rate:.1f}%")
    c5.metric("MPs Tracked", f"{total_mps:,}")

    st.markdown(
        """
        <div class="ai-banner">
        🤖 <strong>New in this dashboard:</strong> an AI Risk Intelligence layer flags works and
        MPs whose spending patterns are statistically unusual and worth verifying first —
        something the standard transparency view doesn't do. See the
        <strong>"AI Risk Intelligence — New"</strong> tab in the sidebar.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Browse by State")
    if not browse_df.empty and "State" in browse_df.columns:
        state_summary = (
            browse_df.groupby("State")
            .agg(mps=("MP Name", "count"), allocated=("Allocated Amount (₹)", "sum"))
            .reset_index()
            .sort_values("allocated", ascending=False)
            .head(12)
        )
        cols = st.columns(4)
        for i, row in enumerate(state_summary.itertuples()):
            with cols[i % 4]:
                st.markdown(
                    f'<div class="state-card"><strong>{row.State}</strong><br>'
                    f'{row.mps} MP(s) · ₹{row.allocated:,.0f} allocated</div>',
                    unsafe_allow_html=True,
                )
                if st.button("View →", key=f"state_btn_{row.State}"):
                    st.session_state.selected_state = row.State
                    st.session_state.nav = "🗺️ Browse by State"
                    st.rerun()
    else:
        st.info("Browse data isn't available right now.")

# =========================================================================
# PAGE: BROWSE BY STATE
# =========================================================================
elif st.session_state.nav == "🗺️ Browse by State":
    st.subheader("Browse by State")

    if browse_df.empty or "State" not in browse_df.columns:
        st.warning("No browsing data available.")
    else:
        states = sorted(browse_df["State"].dropna().unique())
        default_idx = states.index(st.session_state.selected_state) if st.session_state.selected_state in states else 0
        picked_state = st.selectbox("Select a state", states, index=default_idx)
        st.session_state.selected_state = picked_state

        state_df = browse_df[browse_df["State"] == picked_state]
        c1, c2, c3 = st.columns(3)
        c1.metric("MPs in this state", f"{len(state_df):,}")
        c2.metric("Total Allocated", f"₹{state_df['Allocated Amount (₹)'].sum():,.0f}")
        c3.metric("Avg. Completion Rate", f"{state_df['Completion Rate %'].mean():.1f}%")

        display_cols = ["MP Name", "Constituency", "House", "Allocated Amount (₹)",
                         "Total Expenditure (₹)", "Utilization %", "Completion Rate %"]
        display_cols = [c for c in display_cols if c in state_df.columns]
        st.dataframe(state_df[display_cols].sort_values("Allocated Amount (₹)", ascending=False),
                     width="stretch", height=350)

        mp_names = state_df["MP Name"].tolist()
        if mp_names:
            st.markdown("##### Jump to an MP's profile")
            picked_mp = st.selectbox("Select an MP", mp_names, key="state_page_mp_pick")
            if st.button("View full profile →"):
                st.session_state.selected_mp = picked_mp
                st.session_state.nav = "🏛️ MP Profiles"
                st.rerun()

# =========================================================================
# PAGE: MP PROFILES
# =========================================================================
elif st.session_state.nav == "🏛️ MP Profiles":
    st.subheader("MP Profiles")

    if browse_df.empty or "MP Name" not in browse_df.columns:
        st.warning("No MP data available.")
    else:
        all_names = sorted(browse_df["MP Name"].dropna().unique())
        query = st.text_input("Search for an MP by name")
        matches = [n for n in all_names if query.lower() in n.lower()] if query else all_names

        default_idx = matches.index(st.session_state.selected_mp) if st.session_state.selected_mp in matches else 0
        if matches:
            picked = st.selectbox("Select an MP", matches, index=default_idx)
            st.session_state.selected_mp = picked
            row = browse_df[browse_df["MP Name"] == picked].iloc[0]

            st.markdown(f"### {row['MP Name']}")
            st.caption(f"{row.get('Constituency', '')} · {row.get('State', '')} · {row.get('House', '')}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Allocated", f"₹{row['Allocated Amount (₹)']:,.0f}")
            c2.metric("Expenditure", f"₹{row['Total Expenditure (₹)']:,.0f}")
            c3.metric("Utilisation", f"{row['Utilization %']:.1f}%")
            c4.metric("Completion", f"{row['Completion Rate %']:.1f}%")

            # 🤖 Our addition: inline AI risk assessment on the MP profile itself
            scored_match = mps_scored_df[mps_scored_df["MP Name"] == picked] if not mps_scored_df.empty else pd.DataFrame()
            st.markdown('<div class="ai-banner">', unsafe_allow_html=True)
            if not scored_match.empty:
                srow = scored_match.iloc[0]
                st.markdown(f"🤖 **AI Risk Assessment** — score **{srow['mp_final_risk_score']:.0f}/100** "
                            f"({srow['mp_risk_band']} risk)")
                for reason in srow.get("reasons", []):
                    st.markdown(f"- {reason}")
            else:
                st.markdown("🤖 **AI Risk Assessment** — this MP hasn't been scored in the current "
                            "Colab run yet. Re-run the notebook to include them.")
            st.markdown("</div>", unsafe_allow_html=True)

            mp_works = works_df[works_df["MP Name"] == picked] if not works_df.empty else pd.DataFrame()
            if not mp_works.empty:
                st.markdown("##### Flagged works for this MP")
                st.dataframe(
                    mp_works[["Work Description", "amount", "status", "final_risk_score", "risk_band"]]
                    .sort_values("final_risk_score", ascending=False).head(20),
                    width="stretch",
                )
        else:
            st.info("No MPs match that search.")

# =========================================================================
# PAGE: 🤖 AI RISK INTELLIGENCE — our added layer
# =========================================================================
elif st.session_state.nav == "🤖 AI Risk Intelligence — New":
    st.markdown(
        """
        <div class="ai-banner">
        🤖 <strong>This section is our addition</strong> — it does not exist on the standard
        MPLADS transparency dashboard. It answers three questions officials can't get answered
        elsewhere: which works to look at first, why they look unusual, and what to verify.
        A flagged item is <strong>not</strong> a finding of fraud — it's a statistical anomaly
        worth a human look.
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Works scored", f"{meta.get('total_works_scored', len(works_df)):,}")
    m2.metric("🔴 High-risk works", f"{meta.get('high_risk_works', 0):,}")
    m3.metric("🟠 Medium-risk works", f"{meta.get('medium_risk_works', 0):,}")
    m4.metric("🔴 High-risk MPs", f"{meta.get('high_risk_mps', 0):,}")

    st.sidebar.header("AI Risk Filters")
    states = sorted(works_df["State"].dropna().unique()) if not works_df.empty else []
    worktypes = sorted(works_df["work_type"].dropna().unique()) if not works_df.empty else []
    selected_states = st.sidebar.multiselect("State", states, default=[])
    selected_worktypes = st.sidebar.multiselect("Work type", worktypes, default=[])
    selected_bands = st.sidebar.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium"])
    min_amount = st.sidebar.number_input("Minimum work amount (₹)", value=0, step=50_000)

    def apply_filters(df, band_col):
        out = df.copy()
        if selected_states:
            out = out[out["State"].isin(selected_states)]
        if selected_worktypes and "work_type" in out.columns:
            out = out[out["work_type"].isin(selected_worktypes)]
        if selected_bands:
            out = out[out[band_col].isin(selected_bands)]
        if min_amount and "amount" in out.columns:
            out = out[out["amount"] >= min_amount]
        return out

    sub_queue, sub_mp = st.tabs(["🎯 Priority Queue", "🏛️ MP / Fund Risk"])

    with sub_queue:
        filtered_works = apply_filters(works_df, "risk_band")
        st.caption(f"Showing {len(filtered_works):,} of {len(works_df):,} scored works "
                   "(top 3000 riskiest nationwide; narrow with the sidebar filters)")
        if filtered_works.empty:
            st.warning("No works match the current filters.")
        else:
            top = filtered_works.sort_values("final_risk_score", ascending=False).head(200)
            for _, row in top.iterrows():
                band = row.get("risk_band", "Low")
                dot = "red" if band == "High" else "orange" if band == "Medium" else "green"
                with st.expander(
                    f":{dot}[●] **{row['final_risk_score']:.0f}/100** — "
                    f"{row['Work Description'][:90]}  ·  {row['MP Name']}  ·  {row['State']}"
                ):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown("**Why flagged:**")
                        for reason in row.get("reasons", []):
                            st.markdown(f"- {reason}")
                        st.markdown("**What to verify first:**")
                        for item in row.get("verify_checklist", []):
                            st.markdown(f"- [ ] {item}")
                    with c2:
                        st.markdown(f"**Amount:** ₹{row['amount']:,.0f}")
                        st.markdown(f"**Status:** {row['status']}")
                        st.markdown(f"**Work type:** {row['work_type'].replace('_', ' ').title()}")
                        st.markdown(f"**Constituency:** {row['Constituency']}")
                        st.markdown(f"**Work ID:** {row['Work ID']}")

    with sub_mp:
        if mps_scored_df.empty:
            st.warning("No MP-level risk data found.")
        else:
            filtered_mps = apply_filters(mps_scored_df, "mp_risk_band")
            mp_view = (filtered_mps if not filtered_mps.empty else mps_scored_df).sort_values(
                "mp_final_risk_score", ascending=False
            )
            display_cols = [c for c in [
                "MP Name", "State", "Constituency", "mp_final_risk_score", "mp_risk_band",
                "Utilization %", "Completion Rate %", "top_vendor", "top_vendor_share",
            ] if c in mp_view.columns]
            st.dataframe(
                mp_view[display_cols].head(100).style.format({
                    "mp_final_risk_score": "{:.1f}", "Utilization %": "{:.1f}%",
                    "Completion Rate %": "{:.1f}%", "top_vendor_share": "{:.0%}",
                }),
                width="stretch", height=420,
            )
            mp_names = mp_view["MP Name"].head(100).tolist()
            if mp_names:
                picked = st.selectbox("Select an MP to see details", mp_names, key="ai_mp_pick")
                mp_row = mp_view[mp_view["MP Name"] == picked].iloc[0]
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown("**Why flagged:**")
                    for reason in mp_row.get("reasons", []):
                        st.markdown(f"- {reason}")
                    st.markdown("**What to verify first:**")
                    for item in mp_row.get("verify_checklist", []):
                        st.markdown(f"- [ ] {item}")
                with c2:
                    st.metric("Risk score", f"{mp_row['mp_final_risk_score']:.0f}/100")
                    st.metric("Risk band", mp_row["mp_risk_band"])

# =========================================================================
# PAGE: ANALYTICS
# =========================================================================
elif st.session_state.nav == "📊 Analytics":
    st.subheader("Where the AI-flagged risk is concentrated")

    c1, c2 = st.columns(2)
    with c1:
        if not state_agg.empty:
            fig = px.bar(state_agg.head(15), x="State", y="high_risk",
                         title="High-risk works by state (top 15)",
                         color_discrete_sequence=[RISK_COLORS["High"]])
            st.plotly_chart(fig, width="stretch")
    with c2:
        if not worktype_agg.empty:
            fig2 = px.bar(worktype_agg, x="work_type", y="high_risk",
                          title="High-risk works by category",
                          color_discrete_sequence=[RISK_COLORS["High"]])
            st.plotly_chart(fig2, width="stretch")

    if not works_df.empty:
        fig3 = px.histogram(works_df, x="final_risk_score", nbins=40, color="risk_band",
                             color_discrete_map=RISK_COLORS,
                             title="Distribution of AI risk scores across all scored works")
        st.plotly_chart(fig3, width="stretch")

    if not mps_scored_df.empty:
        st.caption("Points far above the diagonal = money reported spent, little to show for it on the ground.")
        fig4 = px.scatter(mps_scored_df, x="Completion Rate %", y="Utilization %",
                          color="mp_risk_band", color_discrete_map=RISK_COLORS,
                          hover_data=["MP Name", "State"],
                          title="Utilisation % vs Completion Rate % by MP")
        fig4.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(dash="dash", color="gray"))
        st.plotly_chart(fig4, width="stretch")

# -----------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------
st.divider()
st.caption(
    "Browsing pages modeled on the public Empowered Indian MPLADS dashboard "
    "(empoweredindian.in/mplads) and its public API. The AI Risk Intelligence "
    "tab is our own addition, scored by a hybrid rule-based + Isolation Forest "
    f"model (see the companion Colab notebook). Last AI scoring run: {meta.get('generated_at', 'unknown')}."
)
