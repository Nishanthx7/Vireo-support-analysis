# AI / model iteration log

## Version 0 - existing tags only
I first measured the supplied tags against the operational proxy labels on the chronological holdout.

- Overall accuracy: **90.6%**
- Billing-tagged accuracy: **72.0%**

Conclusion: the tags are not "rubbish" everywhere. The largest material weakness is Billing.

## Version 1 - replace every category with a text model
I tested a full replacement classifier using customer opening message + agent closing note.

- Overall holdout accuracy: **81.0%**

This was worse than simply keeping the existing tags, so I discarded it.

## Version 2 - narrow correction layer
Final approach:

- Keep the existing category by default.
- Only allow the text model to override `Billing & Payments`.
- Only override when model confidence is at least 0.80.

Results:

- Overall holdout accuracy: **94.6%**
- Billing-tagged holdout accuracy: **93.6%**

Why this version won:
It is smaller, cheaper, less destructive, and directly fixes the category that distorts the staffing decision.

## Tools used
- ChatGPT: brief interpretation, data inspection, validation design, business reasoning, assumption stress-testing and drafting.
- Claude Code: coding, debugging and iterating on the local application.
- Local model in the deliverable: word TF-IDF with class-balanced logistic regression.
- Paid inference/API calls in the working tool: **none**.

## What I deliberately threw away
- Full category replacement model: measurably worse.
- LLM/RAG/agent architecture: more moving parts and cost without evidence it would improve the decision within the five-hour scope.
- Real-time routing claim: the retrospective model uses closing notes, which are unavailable at intake.
