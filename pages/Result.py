import streamlit as st
import json
import pandas as pd
from data.db import load_simulations, delete_simulation, clear_database

# =========================================================
# Page Config
# =========================================================
st.set_page_config(page_title="Simulation Results", layout="wide")
st.title("📊 Delegated Lending Simulation Results")

# =========================================================
# Load stored simulations
# =========================================================
df = load_simulations()

if df is None or df.empty:
    st.warning("No simulations found. Run a simulation first.")
    st.stop()

# =========================================================
# Simulation chooser
# =========================================================
st.markdown("### 📁 Select a simulation run")

selected_row = st.selectbox(
    "Simulation history",
    df.index,
    format_func=lambda idx: f"{df.loc[idx, 'sim_name']} — {df.loc[idx, 'run_time']}"
)

sim = df.loc[selected_row]

if "results" not in sim or sim["results"] is None:
    st.error("This simulation has no stored results. Run a new simulation.")
    st.stop()

results = sim["results"]
if isinstance(results, str):
    results = json.loads(results)

st.divider()

# =========================================================
# Overview Metrics
# =========================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Simulation Name", sim["sim_name"])
col2.metric("Number of Firms", sim["num_firms"])
col3.metric("Principal Welfare (Avg)", f"{sim['welfare']:.4f}")
col4.metric("Iterations", sim["iterations"])

st.divider()

# =========================================================
# Market Summary
# =========================================================
st.markdown("## 📊 Market Summary")

summary_rows = []
for firm in results:
    choice = firm["choice"]
    summary_rows.append({
        "Firm ID": firm["firm_id"],
        "True Type": choice["true_type"],
        "Intended": choice["principal_intended_plan"],
        "Chosen": choice["chosen_plan"],
        "Opted Out": choice["non_participation"],
        "Locked": firm["locked"],
        "Payoff": choice["chosen_payoff"],
    })

summary_df = pd.DataFrame(summary_rows)

num_firms = len(summary_df)
num_opt_out = summary_df["Opted Out"].sum()
num_locked = summary_df["Locked"].sum()
num_matched = (summary_df["True Type"] == summary_df["Chosen"]).sum()
num_deviation = num_firms - num_matched - num_opt_out

colA, colB, colC, colD, colE = st.columns(5)
colA.metric("Match Rate", f"{num_matched/num_firms*100:.1f}%")
colB.metric("Deviation Rate", f"{num_deviation/num_firms*100:.1f}%")
colC.metric("Opt-out Rate", f"{num_opt_out/num_firms*100:.1f}%")
colD.metric("Locked Firms", f"{num_locked}/{num_firms}")
colE.metric("Avg Payoff", f"{summary_df['Payoff'].mean():.3f}")

st.dataframe(summary_df, use_container_width=True)

st.divider()

# =========================================================
# Per-Firm Iteration History
# =========================================================
st.markdown("## 🧾 Per-Firm Iteration History")

for firm in results:
    firm_id = firm["firm_id"]
    principal = firm["principal"]
    history = firm["history"]
    final_choice = firm["choice"]

    st.markdown(f"## Firm {firm_id} {'🔒' if firm['locked'] else '🔄'}")

    # ---------------- Principal Beliefs ----------------
    with st.expander("📊 Firm Overview"):
        # ===============================
        # Principal Beliefs
        # ===============================
        st.markdown("### 🧠 Principal Beliefs")

        belief_df = pd.DataFrame({
            "Metric": ["Belief SAFE (%)", "π_safe", "π_mixed", "π_risky"],
            "Value": [
                f"{principal['belief_safe_score']:.1f}%",
                f"{principal['pi_safe']:.3f}",
                f"{principal['pi_mixed']:.3f}",
                f"{principal['pi_risky']:.3f}",
            ],
        })
        st.table(belief_df)

        # ===============================
        # Firm Capacity
        # ===============================
        cap = firm.get("capacity")


        st.markdown("### 📦 Firm Capacity")

        # ---- Firm-level budget (Stage 1) ----
        st.markdown("**Firm Budget (Stage 1 Allocation)**")
        st.table(pd.DataFrame([
            {"Club": "A", "Budget": cap["budget"]["A"]},
            {"Club": "B", "Budget": cap["budget"]["B"]},
        ]))

        # ---- Offered budget (after reserve) ----
        st.markdown("**Offered to Firm (after reserve)**")
        st.table(pd.DataFrame([
            {"Club": "A", "Offered": cap["offered"]["A"]},
            {"Club": "B", "Offered": cap["offered"]["B"]},
        ]))

        # ---- Catalog caps ----
        st.markdown("**Catalog Initial Caps**")
        cap_rows = []
        for cat, (a, b) in cap["catalog_caps"].items():
            cap_rows.append({
                "Catalog": cat.capitalize(),
                "Cap A": a,
                "Cap B": b,
            })

        st.table(pd.DataFrame(cap_rows))


    # ---------------- Iterations ----------------
    with st.expander("Iteration Details"):

        # ===============================
        # Iteration navigation (per firm)
        # ===============================
        iter_key = f"iter_idx_firm_{firm_id}"
        if iter_key not in st.session_state:
            st.session_state[iter_key] = 0

        max_iter = len(history) - 1
        idx = min(st.session_state[iter_key], max_iter)
        st.session_state[iter_key] = idx
        step = history[idx]

        col_prev, col_mid, col_next = st.columns([1, 2, 1])

        with col_prev:
            if st.button("⬅ Previous", key=f"prev_{firm_id}", disabled=(idx == 0)):
                st.session_state[iter_key] -= 1
                st.rerun()

        with col_next:
            if st.button("Next ➡", key=f"next_{firm_id}", disabled=(idx == max_iter)):
                st.session_state[iter_key] += 1
                st.rerun()

        st.markdown(
            f"### Iteration {step['iteration']} — Chosen: **{step['choice']['chosen_plan']}**"
        )

        # ===============================
        # Menu display (TABLES, not JSON)
        # ===============================
        cols = st.columns(3)

        for i, plan_name in enumerate(["safe", "mixed", "risky"]):
            with cols[i]:
                st.markdown(f"**{plan_name.capitalize()} Menu**")

                menu_df = pd.DataFrame(step["plans"][plan_name])
                menu_df = menu_df.rename(columns={
                    "short_price": "Short Price",
                    "long_price": "Long Price",
                    "short_cap": "Cap A",
                    "long_cap": "Cap B",
                })

                menu_df["Short Price"] = menu_df["Short Price"].map(lambda x: f"{x:.2f}")
                menu_df["Long Price"] = menu_df["Long Price"].map(lambda x: f"{x:.2f}")

                # Highlight chosen item
                chosen_idx = step["choice"].get("chosen_item_index")
                if step["choice"]["chosen_plan"] == plan_name:
                    menu_df["Selected"] = [
                        "⬅" if j == chosen_idx else "" for j in range(len(menu_df))
                    ]

                st.table(menu_df)

        # ===============================
        # Payoff computation tables
        # ===============================
        st.markdown("#### 📦 Payoff Computation")

        for plan_name, detail in step["choice"]["plan_details"].items():

            if detail is None:
                st.warning(f"{plan_name.capitalize()} Plan: ❌ Infeasible")
                continue

            capA, capB = detail["caps"]
            priceA, priceB = detail["zcb_prices"]

            payoff_df = pd.DataFrame({
                "Metric": [
                    "Borrow A",
                    "Borrow B",
                    "Cap A",
                    "Cap B",
                    "Short Price",
                    "Long Price",
                    "Expected Return",
                    "Funds Borrowed",
                    "Raw Payoff",
                ],
                "Value": [
                    detail["A"],
                    detail["B"],
                    capA,
                    capB,
                    f"{priceA:.2f}",
                    f"{priceB:.2f}",
                    f"{detail['expected_return']:.2f}",
                    f"{detail['funds_borrowed']:.2f}",
                    f"{detail['raw_payoff']:.2f}",
                ],
            })

            st.markdown(f"**{plan_name.capitalize()} Plan**")
            st.table(payoff_df)


    # ---------------- Final Decision ----------------
    with st.expander("Final Decision"):
        true_type = final_choice["true_type"]
        chosen = final_choice["chosen_plan"]

        st.write(f"**True Type:** {true_type}")
        st.write(f"**Chosen Plan:** {chosen}")
        st.write(f"**Intended Plan:** {final_choice['principal_intended_plan']}")
        st.write(f"**Final Payoff:** {final_choice['chosen_payoff']:.3f}")

        if chosen == true_type:
            st.success("✅ MATCH: True type matches chosen plan")
        else:
            st.error("❌ NOT MATCH: True type does not match chosen plan")


    st.divider()

# =========================================================
# Delete Controls
# =========================================================
with st.expander("🗑 Delete Simulations"):
    colA, colB = st.columns(2)
    with colA:
        if st.button("Delete Selected Simulation"):
            delete_simulation(int(sim["id"]))
            st.success("Simulation deleted.")
            st.rerun()
    with colB:
        if st.button("Clear ALL Simulations"):
            clear_database()
            st.success("All simulations deleted.")
            st.rerun()
