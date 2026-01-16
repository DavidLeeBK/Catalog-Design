# 📈 Delegated Lending Simulator (Streamlit)

This project is an interactive **delegated lending simulation** where a **principal** offers a menu of **zero-coupon bond (ZCB) contracts** to multiple firms under uncertainty. Each firm has a **true (hidden) type** (safe / mixed / risky), while the principal only has a **belief** about that type.

The simulator shows how firms respond to the contract menus, how menus adapt across rounds, and whether the market “stabilizes” (locks) into a consistent matching outcome.

---

## Requirements
streamlit>=1.30
pandas>=1.5

## What this simulator does

For each firm, you provide:

- The principal’s **belief** (how likely the firm is “safe” in %)
- The firm’s **funding need**
- The principal’s expected **project returns** (Year 1 & Year 2)
- The firm’s **true type** (safe / mixed / risky) — treated as “hidden truth”

Then the simulation:

1. Allocates budgets from two global lending pools (**Club A** and **Club B**) across firms.
2. Builds three catalogs (**Safe / Mixed / Risky**) with:
   - ZCB prices for short vs long maturity
   - Borrowing caps derived from the allocated budgets
3. Lets each firm choose the best available catalog item (or opt out).
4. Updates catalog prices (and expands caps when firms opt out).
5. Repeats for multiple rounds until choices stabilize (firms become “locked”).

You can view and compare saved runs in a results dashboard.

---

## Key ideas

- **Principal vs Firm information gap:**  
  The principal does not know the firm’s real type, only a belief distribution derived from the “Believes SAFE (%)” input.

- **Menus / catalogs:**  
  The principal offers 3 catalogs: safe, mixed, risky. Each catalog contains **3 menu items** with slightly different price/cap tradeoffs.

- **Two lending “clubs”:**
  - **Club A** = short-term side (Year 1)
  - **Club B** = long-term side (Year 2)

- **Opt-out:**  
  If the best available adjusted payoff is not positive, the firm opts out.

- **Learning / adaptation:**  
  Even if a firm does not pick a catalog, prices are updated using “hypothetical usage”, and if the firm opts out, caps are expanded toward a ceiling.

---

## Features

### ✅ Simulation UI (Streamlit)
- Configure number of firms (1–10)
- Set global funds available for Club A and Club B
- Set per-firm belief, funding need, expected returns, and true type
- Confirmation screen before running
- Automatically saves run results to a local database  
  (see `save_simulation(...)` usage)

> Main UI app: `Simulator.py` :contentReference[oaicite:3]{index=3}

### ✅ Results Dashboard
- Select past simulation runs
- Market summary: match rate, deviation rate, opt-out rate, locked firms, average payoff
- Per-firm drill-down:
  - belief breakdown (π_safe, π_mixed, π_risky)
  - budget + offered amounts
  - initial catalog caps
  - iteration-by-iteration plan menus & payoff computation tables
  - final decision (match vs not match)

> Results page: `Result.py` :contentReference[oaicite:4]{index=4}

---


