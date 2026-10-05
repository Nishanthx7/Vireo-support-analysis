# Vireo Support Capacity Analyzer

A small, local AI-assisted tool for the Vireo Audio support-ticket task.

## What it does

- Loads the supplied Vireo CSV exports.
- Restricts the analysis to the stated Jan 2025 - Jun 2026 window.
- Corrects the known legacy resolution timestamp offset for duration analysis.
- Uses a lightweight text classifier (TF-IDF + logistic regression) as a **correction layer** for noisy ticket tags.
- Deliberately narrows the AI intervention to high-confidence corrections of `Billing & Payments`, because that is where the data shows the routing problem is concentrated.
- Produces monthly ticket volume by corrected category.
- Produces monthly workload by resolving team.
- Compares first-routed vs resolving team.
- Shows team workload per active agent.
- Reports validation metrics on a chronological holdout set.
- Estimates the annual cost of Billing misroutes using Vireo's Rs 305/transfer planning cost.

## Why the scope is narrow

The existing tags are not uniformly bad. In the current helpdesk period, specialist tags other than Billing mostly land with the expected specialist team, while Billing has a materially higher mismatch rate. Replacing every tag with a new model added risk and did not improve the decision enough to justify the complexity in a five-hour task.

## Requirements

Python 3.10+ recommended.

Install:

```bash
pip install -r requirements.txt
```

## Put the provided Vireo files in `data/`

The tool finds files by suffix, so the original UUID-style filenames are fine.

Required:

- `*tickets.csv`
- `*agents.csv`
- `*orders.csv`
- `*customers.csv`
- `*products.csv`

The policy PDF and email thread are not required at runtime; their business rules are documented in this README and reflected in the analysis.

Example:

```text
vireo_support_tool/
  app.py
  analysis.py
  requirements.txt
  data/
    99ca...-tickets.csv
    69f9...-agents.csv
    ...
```

## Run the analysis

```bash
python analysis.py --data-dir data --output-dir outputs
```

## Run the dashboard

```bash
streamlit run app.py
```

The dashboard regenerates the analysis from `data/` if needed.

## Important assumptions / decisions

1. **Date window**
   - The README says Jan 2025 to Jun 2026.
   - Rows outside that window are excluded.

2. **Legacy timestamps**
   - Vireo policy says migrated resolution timestamps came from a UTC legacy event log while helpdesk reports display IST.
   - For duration analysis only, `legacy_fd` resolution timestamps are shifted by +05:30.
   - Original exported values are never overwritten.

3. **Transfers**
   - The email thread and policy say the transfer field did not exist in the legacy system.
   - Transfer economics therefore use only `helpdesk` rows. Legacy blanks are not converted to zero.

4. **Team workload**
   - `assigned_team` is the first-routed team.
   - The resolving team is derived from `agent_id -> agents.team`.
   - For workload/headcount evidence, the tool uses the resolving team and completed attendance (`resolved` or `closed`).
   - Escalations & Warranty is Tier 2 and is not compared with Tier 1 on volume, per policy.

5. **AI categorisation**
   - The tool is retrospective: it uses both `customer_message` and `agent_notes`.
   - It is **not** presented as a production real-time router because closing notes do not exist at intake.
   - Existing tags are retained except for high-confidence AI corrections to `Billing & Payments`.
   - This is intentional: a smaller intervention tested better and directly addresses the business problem.

6. **Validation labels**
   - Specialist-category proxy labels are derived from the team that ultimately resolved the ticket.
   - Frontline categories retain the existing tag.
   - This is operationally useful but not perfect human ground truth; the limitation is called out in the memo and dashboard.

## Model

- Word TF-IDF (1-2 grams)
- Logistic Regression, class-balanced
- Chronological 80/20 validation split
- High-confidence correction threshold: 0.80

No paid API calls are required.

## Cost

Paid model/API cost per run: **Rs 0**.

At roughly 650 tickets/week:

```text
650 * 52 / 12 = 2,816.7 tickets/month
2,816.7 * Rs 0 = Rs 0/month paid inference cost
```

Local CPU/electricity is not separately priced.

## Known limitations

- The validation target is an operational proxy, not a manually adjudicated gold set.
- The model is retrospective because it uses closing notes.
- Multi-intent tickets still receive one primary category.
- `Other` and warranty-adjacent cases remain harder than clear billing/delivery cases.
- Ticket count is a workload proxy; true staffing should use active handle-time / occupancy data, which is not in the pack.
- No production monitoring, drift detection, authentication, API integration or cloud deployment is included.
