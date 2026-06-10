# Documentation Index

Long-form documentation for Quant Alpha Foundation. For module-level READMEs see each module's directory; for the project overview see the [top-level README](../README.md).

## Documents

| Document | Audience | Description |
|---|---|---|
| [gat_experiment_log.md](gat_experiment_log.md) | Researchers / paper | **Canonical GAT experiment record** — setup, leakage controls, real-data four-gate results, limitations, next experiments. Primary source for the capstone paper's results section |
| [gnn_capstone_design.md](gnn_capstone_design.md) | All | GNN relational-factor capstone architecture design (thesis, two-track graphs, milestones M1–M5) |
| [CAPSTONE_STATUS.md](CAPSTONE_STATUS.md) | Agents / resume | Progress snapshot and resume point for the GAT capstone — module map, test counts, environment quirks, prioritised next steps |
| [adr/](adr/) | All | Architecture decision records 0001–0005 (propagate seam, factor provider, training objective + split hygiene, scope, equity graph) |
| [alpha_research.md](alpha_research.md) | Researchers | Alpha factor methodology, full 18-factor catalog (10 equity + 8 energy), four-gate validation framework (Robustness / Uniqueness / Value-added / Consistency), worked examples |
| [architecture.md](architecture.md) | Engineers | System layers (ingestion → warehouse → factor engine → backtest → dashboard), production boundary, data flow diagram |
| [cloud_kubernetes.md](cloud_kubernetes.md) | DevOps | GCP deployment guide — Terraform IaC (Workload Identity, Secret Manager, BigQuery 3-layer), GKE cluster setup, Helm chart deployment, dev/prod overlays |
| [zoomcamp_coverage.md](zoomcamp_coverage.md) | Reviewers | Module-by-module DataTalksClub Zoomcamp coverage matrix — M1 through M7 + Workshops + Cloud, with concrete file references and test counts |
| [semester_plan.md](semester_plan.md) | All | Project thesis, semester deliverables, milestone schedule |
| [decisions.md](decisions.md) | All | Open architectural decisions and trade-offs awaiting resolution |
| [gpt55_audit_response.md](gpt55_audit_response.md) | Maintainers | Historical record of changes made in response to the first audit pass |

## Related Audit Reports

These live at the project root, not under `docs/`:

| File | Description |
|---|---|
| [../audit_report_v1.md](../audit_report_v1.md) | First code-review report (raw findings) |
| [../audit_fixes.md](../audit_fixes.md) | Fix log for v1 — 17 fixes, organized by file with rationale and effect |
| [../audit_report_v2.md](../audit_report_v2.md) | Second code-review report (raw findings) |
| [../audit_fixes_v2.md](../audit_fixes_v2.md) | Fix log for v2 — 13 fixes, with skip-list explanations |

## Where to Look First

| If you want to… | Read |
|---|---|
| Write or review the capstone paper's results | [gat_experiment_log.md](gat_experiment_log.md) + [docs/results/](results/) |
| Resume work on the GAT capstone | [CAPSTONE_STATUS.md](CAPSTONE_STATUS.md) |
| Understand the alpha factor universe and gating logic | [alpha_research.md](alpha_research.md) |
| Trace data flow from raw API to dashboard | [architecture.md](architecture.md) |
| Deploy to GCP (Terraform → GKE → Helm) | [cloud_kubernetes.md](cloud_kubernetes.md) |
| Verify Zoomcamp module coverage for grading | [zoomcamp_coverage.md](zoomcamp_coverage.md) |
| See what was hardened in the security/correctness audits | [audit_fixes.md](../audit_fixes.md), [audit_fixes_v2.md](../audit_fixes_v2.md) |
