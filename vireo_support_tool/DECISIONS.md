# Decisions / scope log

## 1. I did not replace every existing tag
The existing tags are useful as a prior. The strongest defect is concentrated in `Billing & Payments`, so the AI layer only overrides Billing when model confidence is at least 0.80.

**Why:** a full replacement model created unnecessary errors in categories that were already reasonably stable.

## 2. I use resolving team for workload, not first-routed team
`assigned_team` is explicitly the first route. The client is making a staffing decision, so resolving-team volume is a better workload signal.

**Why:** 31% of current-helpdesk tickets first routed to Billing are ultimately resolved elsewhere; 28% resolve in Logistics.

## 3. I do not compare Tier 2 with Tier 1 on raw ticket volume
The policy explicitly says Tier 2 work is multi-touch and must not be compared with Tier 1 using volume metrics.

## 4. I corrected one timestamp issue but did not silently rewrite source data
For legacy duration analysis only, I add +05:30 to `resolved_at`, because the policy says migrated resolution timestamps came from a UTC event log while standard reports display IST.

## 5. I use only current-helpdesk rows for transfer economics
The transfer field did not exist in the legacy system. Blank legacy values are unknown, not zero.

## 6. I did not build real-time routing
The retrospective classifier uses the closing note, which is useful for Priya's historical monthly analysis but is not available at ticket intake.

## 7. I did not fine-tune an LLM / build RAG / build agents
The business question is classification + workload analysis. A local linear text model is faster, cheaper, explainable enough, and has no paid inference cost.
