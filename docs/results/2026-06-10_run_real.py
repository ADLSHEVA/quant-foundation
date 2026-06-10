"""Real-data GAT equity run: yfinance fetch, IC loss, four gates."""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\AI_Models\quant-alpha-foundation\src")
from quant_alpha.run_gat_equity import run_gat_equity

out = run_gat_equity(
    Path("configs/project.yaml"),
    Path(r"D:\AI_Models\quant-alpha-foundation"),
    offline=False,
    epochs=50,
    loss="ic",
)

print("\n=== GATE REPORT (real data, IC loss) ===")
print(json.dumps(out["gate_report"], indent=2, default=str))

diag = out["diagnostics"].set_index("alpha_name")
cols = [c for c in ("oos_ic_mean", "oos_ic_ir", "oos_sharpe", "consistency_score", "robustness_score") if c in diag.columns]
print("\n=== DIAGNOSTICS (OOS) ===")
print(diag[cols].round(4).to_string())

out["diagnostics"].to_csv(".scratch/real_run_diagnostics.csv", index=False)
print(f"\nweights: {out['weights_path']}")
print(f"panel rows: {len(out['panel'])}, dates: {out['panel']['date'].nunique()}, symbols: {out['panel']['symbol'].nunique()}")
