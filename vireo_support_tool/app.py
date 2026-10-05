
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from analysis import build_outputs

st.set_page_config(page_title="Vireo Support Capacity Analyzer", layout="wide")
st.title("Vireo Support Capacity Analyzer")
st.caption("AI-assisted retrospective tag correction + workload analysis")

data_dir = Path("data")
output_dir = Path("outputs")

required = ["tickets.csv", "agents.csv", "orders.csv", "customers.csv", "products.csv"]
missing = [s for s in required if not list(data_dir.glob(f"*{s}"))]

if missing:
    st.error(
        "Put the supplied Vireo CSVs in ./data/ before running. Missing suffixes: "
        + ", ".join(missing)
    )
    st.stop()

if st.button("Rebuild analysis") or not (output_dir / "business_metrics.json").exists():
    with st.spinner("Running analysis..."):
        build_outputs(data_dir, output_dir)

business = json.loads((output_dir / "business_metrics.json").read_text())
model = json.loads((output_dir / "model_metrics.json").read_text())
quality = json.loads((output_dir / "data_quality.json").read_text())
team = pd.read_csv(output_dir / "team_capacity_summary.csv")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Billing misroute rate", f"{business['billing_mismatch_rate']*100:.1f}%")
c2.metric("Billing -> Logistics", f"{business['billing_to_logistics_rate']*100:.1f}%")
c3.metric("Target misroute rate", f"{business['target_billing_mismatch_rate']*100:.0f}%")
c4.metric("Est. annual transfer saving", f"Rs {business['estimated_annual_savings_inr']/100000:.2f}L")

st.info(
    "Recommendation: do not allocate two hires to Billing based on first-route volume alone. "
    "Correct the Billing/Delivery tagging problem first. If capacity must be added now, "
    "Logistics has the stronger Tier-1 pressure signal in the recent data."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Monthly categories", "Teams", "Validation", "Data quality", "Method & Iterations"]
)

with tab1:
    st.subheader("Monthly volume by corrected category")
    st.image(str(output_dir / "monthly_by_category.png"))
    st.caption(
        "The existing category is retained except for high-confidence AI corrections "
        "to Billing & Payments."
    )

with tab2:
    st.subheader("Monthly completed tickets by resolving team")
    st.image(str(output_dir / "monthly_by_team.png"))
    st.subheader("Recent workload")
    st.dataframe(team, use_container_width=True, hide_index=True)
    st.image(str(output_dir / "tickets_per_agent.png"))
    st.caption(
        "Escalations & Warranty is Tier 2 and should not be compared with Tier 1 on raw volume."
    )

with tab3:
    st.subheader("Automated proxy validation")
    c1, c2, c3 = st.columns(3)
    c1.metric("Holdout tickets", f"{model['holdout_n']:,}")
    c2.metric("Proxy accuracy", f"{model['hybrid_accuracy']*100:.1f}%")
    c3.metric("Proxy error rate", f"{model['hybrid_error_rate']*100:.1f}%")

    st.write(
        f"Billing-tagged holdout: {model['billing_holdout_correct']} / "
        f"{model['billing_holdout_n']} correct "
        f"({model['billing_holdout_accuracy']*100:.1f}%)."
    )

    st.warning(
        "This automated result uses operational proxy labels, not human gold labels."
    )

    st.divider()

    st.subheader("Manual human validation")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tickets reviewed", "50")
    m2.metric("Correct", "44")
    m3.metric("Manual accuracy", "88%")
    m4.metric("Manual error rate", "12%")

    st.caption(
        "50 tickets were manually reviewed by reading the customer opening message "
        "and the agent closing note. The main errors were closely related or "
        "multi-intent categories, including Billing vs Delivery/Returns and "
        "overlapping software, connectivity, battery and audio issues."
    )

with tab4:
    st.subheader("Data-quality decisions")
    st.json(quality)
    st.write(
        "Legacy resolution timestamps are shifted +05:30 for duration analysis only, "
        "based on the policy's UTC-vs-IST note. Transfer economics use only current-helpdesk rows."
    )


with tab5:
    st.subheader("Method & Iterations")

    st.markdown("### AI used")
    st.write(
        "ChatGPT was used for interpreting the brief, data inspection, validation design, "
        "business reasoning and checking assumptions. Claude Code was used for coding, "
        "debugging and iterating on the local application. "
        "The working classifier itself is a local TF-IDF + logistic regression model, "
        "so the final tool makes no paid model/API calls."
    )

    st.markdown("### What changed")
    st.write(
        "The first approach tried replacing every ticket category with a new text classifier. "
        "That performed worse than keeping the existing tags, so the final approach was narrowed "
        "to high-confidence corrections of Billing & Payments only."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Existing tags", "90.6%")
    c2.metric("Full replacement model", "81.0%")
    c3.metric("Final hybrid approach", "94.6%")

    st.markdown("### What I discarded")
    st.write(
        "I discarded full category replacement, LLM/RAG/agent architecture, cloud deployment "
        "and real-time routing because they added complexity without improving the staffing decision "
        "within the five-hour scope."
    )

    st.markdown("### Final recommendation")
    st.info(
        "Fix the Billing routing problem first. Do not allocate both hires using first-route volume alone. "
        "Use resolving-team workload and headcount-normalised volume before finalising the staffing decision."
    )
