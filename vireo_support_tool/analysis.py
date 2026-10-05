
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2026-07-01")
GO_LIVE = pd.Timestamp("2025-09-14")
CONFIDENCE_THRESHOLD = 0.80
TRANSFER_COST = 305
WEEKLY_VOLUME = 650

FRONT_CATS = {
    "Account & Login", "App & Firmware", "Audio Quality",
    "Charging & Battery", "Connectivity", "Other", "Product Enquiry"
}
SPECIAL_MAP = {
    "Billing": "Billing & Payments",
    "Logistics": "Delivery & Shipping",
    "Returns Desk": "Returns & Refunds",
    "Escalations & Warranty": "Warranty & Repair",
}

def find_one(data_dir: Path, suffix: str) -> Path:
    matches = sorted(data_dir.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"Could not find *{suffix} in {data_dir}")
    return matches[0]

def load(data_dir: Path):
    tickets = pd.read_csv(find_one(data_dir, "tickets.csv"))
    agents = pd.read_csv(find_one(data_dir, "agents.csv"))
    orders = pd.read_csv(find_one(data_dir, "orders.csv"))
    customers = pd.read_csv(find_one(data_dir, "customers.csv"))
    products = pd.read_csv(find_one(data_dir, "products.csv"))
    return tickets, agents, orders, customers, products

def make_model():
    features = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=25000, sublinear_tf=True
    )
    clf = LogisticRegression(max_iter=500, class_weight="balanced", C=4, solver="lbfgs")
    return features, clf

def prepare(tickets: pd.DataFrame, agents: pd.DataFrame):
    t = tickets.copy()
    for c in ["created_at", "first_response_at", "resolved_at"]:
        t[c] = pd.to_datetime(t[c], errors="coerce")

    raw_rows = len(t)
    t = t[(t["created_at"] >= START) & (t["created_at"] < END)].copy()
    excluded_outside_window = raw_rows - len(t)

    # Preserve exported timestamp; use corrected value only for duration analysis.
    t["resolved_at_adjusted"] = t["resolved_at"]
    legacy = t["source_system"].eq("legacy_fd") & t["resolved_at"].notna()
    t.loc[legacy, "resolved_at_adjusted"] = (
        t.loc[legacy, "resolved_at"] + pd.Timedelta(hours=5, minutes=30)
    )

    team_map = agents.drop_duplicates("agent_id").set_index("agent_id")["team"]
    t["resolver_team"] = t["agent_id"].map(team_map)
    t["month"] = t["created_at"].dt.to_period("M").astype(str)
    t["resolution_hours"] = (
        t["resolved_at_adjusted"] - t["created_at"]
    ).dt.total_seconds() / 3600.0

    # Operational proxy labels:
    # final specialist resolver determines specialist category;
    # frontline category keeps existing tag.
    t["proxy_category"] = t["resolver_team"].map(SPECIAL_MAP)
    frontline = t["proxy_category"].isna() & t["category"].isin(FRONT_CATS)
    t.loc[frontline, "proxy_category"] = t.loc[frontline, "category"]

    return t, excluded_outside_window

def validate_and_predict(t: pd.DataFrame):
    labeled = t[t["proxy_category"].notna()].sort_values("created_at").copy()
    split = int(len(labeled) * 0.80)
    train = labeled.iloc[:split].copy()
    test = labeled.iloc[split:].copy()

    features, clf = make_model()
    x_train = features.fit_transform(
        train["customer_message"].fillna("") + " [NOTE] " + train["agent_notes"].fillna("")
    )
    x_test = features.transform(
        test["customer_message"].fillna("") + " [NOTE] " + test["agent_notes"].fillna("")
    )
    clf.fit(x_train, train["proxy_category"])
    test["model_pred"] = clf.predict(x_test)
    test["model_conf"] = clf.predict_proba(x_test).max(axis=1)

    # Hybrid: keep existing tags except high-confidence Billing corrections.
    test["hybrid_category"] = test["category"]
    change = (
        test["category"].eq("Billing & Payments")
        & test["model_conf"].ge(CONFIDENCE_THRESHOLD)
    )
    test.loc[change, "hybrid_category"] = test.loc[change, "model_pred"]

    overall_accuracy = accuracy_score(test["proxy_category"], test["hybrid_category"])
    overall_macro_f1 = f1_score(
        test["proxy_category"], test["hybrid_category"], average="macro"
    )

    billing_test = test[test["category"].eq("Billing & Payments")].copy()
    billing_accuracy = accuracy_score(
        billing_test["proxy_category"], billing_test["hybrid_category"]
    )
    billing_errors = int(
        (billing_test["proxy_category"] != billing_test["hybrid_category"]).sum()
    )

    # Retrain on all proxy-labelled history for final retrospective report.
    full_features, full_clf = make_model()
    x_full = full_features.fit_transform(
        labeled["customer_message"].fillna("") + " [NOTE] " + labeled["agent_notes"].fillna("")
    )
    full_clf.fit(x_full, labeled["proxy_category"])

    x_all = full_features.transform(
        t["customer_message"].fillna("") + " [NOTE] " + t["agent_notes"].fillna("")
    )
    t["model_pred"] = full_clf.predict(x_all)
    t["model_conf"] = full_clf.predict_proba(x_all).max(axis=1)
    t["corrected_category"] = t["category"]

    change_all = (
        t["category"].eq("Billing & Payments")
        & t["model_conf"].ge(CONFIDENCE_THRESHOLD)
    )
    t.loc[change_all, "corrected_category"] = t.loc[change_all, "model_pred"]

    metrics = {
        "holdout_n": int(len(test)),
        "holdout_start": str(test["created_at"].min()),
        "holdout_end": str(test["created_at"].max()),
        "hybrid_accuracy": float(overall_accuracy),
        "hybrid_error_rate": float(1 - overall_accuracy),
        "hybrid_macro_f1": float(overall_macro_f1),
        "billing_holdout_n": int(len(billing_test)),
        "billing_holdout_correct": int(len(billing_test) - billing_errors),
        "billing_holdout_errors": billing_errors,
        "billing_holdout_accuracy": float(billing_accuracy),
        "billing_holdout_error_rate": float(1 - billing_accuracy),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "model_note": (
            "Proxy labels: final specialist resolver mapped to specialist category; "
            "frontline categories retain existing tag. This is not human gold truth."
        )
    }
    return t, test, metrics

def build_outputs(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    tickets, agents, orders, customers, products = load(data_dir)
    t, excluded = prepare(tickets, agents)
    t, validation, model_metrics = validate_and_predict(t)

    current = t[t["source_system"].eq("helpdesk")].copy()
    current["route_mismatch"] = current["assigned_team"] != current["resolver_team"]

    # Data quality diagnostics.
    legacy_raw_negative = int((
        (t["source_system"].eq("legacy_fd"))
        & t["resolved_at"].notna()
        & (t["resolved_at"] < t["created_at"])
    ).sum())
    legacy_adjusted_negative = int((
        t["resolved_at_adjusted"].notna()
        & (t["resolved_at_adjusted"] < t["created_at"])
    ).sum())

    # Team workload: recent H1 2026, completed attendance only.
    attendance = current[current["status"].isin(["resolved", "closed"])].copy()
    h1 = attendance[
        (attendance["created_at"] >= pd.Timestamp("2026-01-01"))
        & (attendance["created_at"] < pd.Timestamp("2026-07-01"))
    ].copy()

    headcount = agents.groupby("team")["agent_id"].nunique()
    team_summary = h1.groupby("resolver_team").agg(
        tickets=("ticket_id", "size"),
        median_resolution_hours=("resolution_hours", "median"),
        avg_csat=("csat_score", "mean"),
    )
    team_summary["agents"] = team_summary.index.map(headcount)
    team_summary["tickets_per_agent_per_month"] = (
        team_summary["tickets"] / team_summary["agents"] / 6.0
    )
    team_summary = team_summary.sort_values("tickets_per_agent_per_month", ascending=False)

    # Routing / transfer economics.
    billing = current[current["assigned_team"].eq("Billing")].copy()
    billing["mismatch"] = billing["resolver_team"] != "Billing"
    billing_mismatch_rate = float(billing["mismatch"].mean())
    billing_to_logistics_rate = float((billing["resolver_team"] == "Logistics").mean())
    mismatch_transfers = float(billing.loc[billing["mismatch"], "transfers"].sum())
    mismatch_n = int(billing["mismatch"].sum())
    avg_transfers_per_mismatch = mismatch_transfers / mismatch_n if mismatch_n else 0.0

    annual_tickets = WEEKLY_VOLUME * 52
    billing_share = len(billing) / len(current)
    annual_billing_tickets = annual_tickets * billing_share
    target_mismatch_rate = 0.15

    current_mismatch_transfer_cost = (
        annual_billing_tickets
        * billing_mismatch_rate
        * avg_transfers_per_mismatch
        * TRANSFER_COST
    )
    target_mismatch_transfer_cost = (
        annual_billing_tickets
        * target_mismatch_rate
        * avg_transfers_per_mismatch
        * TRANSFER_COST
    )
    estimated_annual_savings = (
        current_mismatch_transfer_cost - target_mismatch_transfer_cost
    )

    business = {
        "weekly_volume_assumption": WEEKLY_VOLUME,
        "monthly_volume_assumption": WEEKLY_VOLUME * 52 / 12,
        "current_helpdesk_n": int(len(current)),
        "billing_share_current": float(billing_share),
        "billing_mismatch_rate": billing_mismatch_rate,
        "billing_to_logistics_rate": billing_to_logistics_rate,
        "billing_mismatch_transfers_observed": mismatch_transfers,
        "avg_transfers_per_billing_mismatch": float(avg_transfers_per_mismatch),
        "transfer_cost_inr": TRANSFER_COST,
        "target_billing_mismatch_rate": target_mismatch_rate,
        "annual_billing_misroute_transfer_cost_inr": float(current_mismatch_transfer_cost),
        "target_annual_transfer_cost_inr": float(target_mismatch_transfer_cost),
        "estimated_annual_savings_inr": float(estimated_annual_savings),
    }

    # Aggregates.
    monthly_category = pd.crosstab(t["month"], t["corrected_category"])
    all_attendance = t[t["status"].isin(["resolved", "closed"])].copy()
    monthly_team = pd.crosstab(
        all_attendance["month"], all_attendance["resolver_team"]
    )

    # Public-safe row-level output: no customer text or direct PII.
    safe = t[[
        "ticket_id", "month", "channel", "category", "corrected_category",
        "model_conf", "assigned_team", "resolver_team", "status", "transfers"
    ]].copy()

    monthly_category.to_csv(output_dir / "monthly_by_category.csv")
    monthly_team.to_csv(output_dir / "monthly_by_team.csv")
    team_summary.to_csv(output_dir / "team_capacity_summary.csv")
    safe.to_csv(output_dir / "classified_ticket_summary.csv", index=False)

    with open(output_dir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(model_metrics, f, indent=2)
    with open(output_dir / "business_metrics.json", "w", encoding="utf-8") as f:
        json.dump(business, f, indent=2)

    quality = {
        "raw_rows": int(len(tickets)),
        "rows_in_stated_window": int(len(t)),
        "excluded_outside_stated_window": int(excluded),
        "legacy_resolved_before_created_raw": legacy_raw_negative,
        "resolved_before_created_after_legacy_adjustment": legacy_adjusted_negative,
        "legacy_transfer_blank_is_not_zero": True,
    }
    with open(output_dir / "data_quality.json", "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)

    # Charts - each chart is a separate figure.
    ax = monthly_category.plot(figsize=(12, 6))
    ax.set_title("Monthly support volume by corrected category")
    ax.set_xlabel("Month")
    ax.set_ylabel("Tickets")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_by_category.png", dpi=160)
    plt.close()

    ax = monthly_team.plot(figsize=(12, 6))
    ax.set_title("Monthly completed ticket volume by resolving team")
    ax.set_xlabel("Month")
    ax.set_ylabel("Completed tickets")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_by_team.png", dpi=160)
    plt.close()

    ts = team_summary.reset_index()
    ax = ts.plot(
        x="resolver_team", y="tickets_per_agent_per_month",
        kind="bar", legend=False, figsize=(10, 5)
    )
    ax.set_title("H1 2026 tickets per agent per month")
    ax.set_xlabel("Resolving team")
    ax.set_ylabel("Tickets / agent / month")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "tickets_per_agent.png", dpi=160)
    plt.close()

    return {
        "quality": quality,
        "model": model_metrics,
        "business": business,
        "team_summary": team_summary.reset_index().to_dict(orient="records"),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="outputs")
    args = p.parse_args()
    result = build_outputs(Path(args.data_dir), Path(args.output_dir))
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
