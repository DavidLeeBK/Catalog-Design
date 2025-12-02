import streamlit as st
import json
import pandas as pd
from data.db import load_simulations, delete_simulation, clear_database

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

# Load results JSON
results = sim["results"]
if isinstance(results, str):
    results = json.loads(results)

st.divider()


# =========================================================
# NEW Overview Header
# =========================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Simulation Name", sim["sim_name"])
col2.metric("Number of Firms", sim["num_firms"])
col3.metric("Principal Welfare (Avg)", f"{sim['welfare']:.4f}")
col4.metric("Iterations (Until Lock)", sim["iterations"])

st.divider()

# =========================================================
# MARKET SUMMARY (UPDATED FOR MULTI-ITERATION)
# =========================================================
st.markdown("## 📊 Market Summary Dashboard")

summary_df = []
for firm in results:
    final_choice = firm["choice"]
    summary_df.append({
        "firm_id": firm["firm_id"],
        "true_type": final_choice["true_type"],
        "intended": final_choice["principal_intended_plan"],
        "chosen": final_choice["chosen_plan"],
        "non_participation": final_choice["non_participation"],
        "locked": firm["locked"],
        "payoff": final_choice["chosen_payoff"],
    })

summary_df = pd.DataFrame(summary_df)

num_firms = len(summary_df)
num_opt_out = summary_df["non_participation"].sum()
num_locked = summary_df["locked"].sum()
num_matched = (summary_df["chosen"] == summary_df["true_type"]).sum()
num_deviation = num_firms - num_matched - num_opt_out

match_rate = num_matched / num_firms
deviation_rate = num_deviation / num_firms
opt_out_rate = num_opt_out / num_firms
avg_payoff = summary_df["payoff"].mean()

colA, colB, colC, colD, colE = st.columns(5)
colA.metric("Match Rate", f"{match_rate*100:.1f}%")
colB.metric("Deviation Rate", f"{deviation_rate*100:.1f}%")
colC.metric("Opt-out Rate", f"{opt_out_rate*100:.1f}%")
colD.metric("Locked Firms", f"{num_locked}/{num_firms}")
colE.metric("Avg Realized Payoff", f"{avg_payoff:.3f}")

# Insights
if opt_out_rate > 0.3:
    st.warning("🔴 Many firms opted out — catalog unattractive.")

elif deviation_rate > 0.4:
    st.warning("🟡 Many firms deviated — plans likely misaligned.")

else:
    st.success("🟢 Catalog performing well overall.")

st.divider()


# =========================================================
# VISUALIZATIONS
# =========================================================
st.markdown("## 📈 Visualizations")

# ---------- CHART 1: Final Chosen Plan Distribution ----------
st.markdown("### 📌 Final Chosen Plans Distribution")

chosen_counts = summary_df["chosen"].value_counts().reset_index()
chosen_counts.columns = ["Plan", "Count"]
st.bar_chart(chosen_counts.set_index("Plan"))

colA , colB = st.columns(2)

with colA:
    st.markdown("### 🥧 Participation vs Non-Participation")
    participation_df = pd.DataFrame({
        "category": ["Participating", "Opt-Out"],
        "count": [num_firms - num_opt_out, num_opt_out]
    })
    st.bar_chart(participation_df.set_index("category"))
    
with colB:
    st.markdown("### 🔄 Deviations Breakdown")
    dev_df = pd.DataFrame({
        "category": ["Matched", "Deviated"],
        "count": [num_matched, num_deviation]
    })
    st.bar_chart(dev_df.set_index("category"))

st.divider()


# =========================================================
#  PER-FIRM ITERATION HISTORY (MOST IMPORTANT NEW SECTION)
# =========================================================
st.markdown("## 🏢 Firm-by-Firm Iteration History")

for firm in results:
    firm_id = firm["firm_id"]
    principal = firm["principal"]
    history = firm["history"]
    final_choice = firm["choice"]

    st.markdown(f"## Firm {firm_id} {'🔒' if firm['locked'] else '🔄'}")

    # -------------------- Principal Beliefs --------------------
    with st.expander("🎯 Principal Beliefs"):
        belief_df = pd.DataFrame({
            "Metric": ["Belief SAFE (%)", "Safe", "Mixed", "Risky"],
            "Value": [
                f"{principal['belief_safe_score']:.1f}%",
                f"{principal['pi_safe']:.3f}",
                f"{principal['pi_mixed']:.3f}",
                f"{principal['pi_risky']:.3f}",
            ]
        })
        st.table(belief_df)

    # -------------------- Iteration History --------------------
    with st.expander("📜 Full Iteration History"):
        for step in history:
            iter_num = step["iteration"]
            chosen = step["choice"]["chosen_plan"]

            st.markdown(f"### 🔁 Iteration {iter_num} — Chosen: **{chosen}**")

            # Show plan parameters
            plans_snapshot = step["plans"]
            with st.container():
                colA, colB, colC = st.columns(3)
                for idx, plan_name in enumerate(["safe", "mixed", "risky"]):
                    plan_data = plans_snapshot[plan_name]
                    col = [colA, colB, colC][idx]
                    with col:
                        st.markdown(f"**{plan_name.capitalize()} Plan**")
                        st.json(plan_data)

            # Show plan payoffs (borrow_A, borrow_B, caps, cost)
            payoff_details = step["choice"]["plan_details"]
            with st.container():
                st.markdown("#### 📦 Borrowing & Payoff Details")
                for plan_name, detail in payoff_details.items():
                    st.markdown(f"**{plan_name.capitalize()} Plan**")
                    df_detail = pd.DataFrame({
                        "Metric": [
                            "Borrow A", "Borrow B", 
                            "Cap A", "Cap B",
                            "Raw Value", "Multiplier",
                            "Value After Multiplier", "Debt Cost", "Payoff"
                        ],
                        "Value": [
                            detail["borrow_A"], detail["borrow_B"],
                            detail["cap_A"], detail["cap_B"],
                            f"{detail['raw_value']:.3f}",
                            f"{detail['fit_multiplier']:.3f}",
                            f"{detail['final_value']:.3f}",
                            f"{detail['cost']:.3f}" if detail["cost"] is not None else "N/A",
                            f"{detail['payoff']:.3f}",
                        ]
                    })
                    st.table(df_detail)
            st.markdown("---")

    # -------------------- Final Decision --------------------
    with st.expander("🤖 Final Firm Decision"):
        true_t = final_choice['true_type']
        chosen = final_choice['chosen_plan']

        is_match = (true_t == chosen)
        is_opt_out = final_choice['non_participation']

        if is_opt_out:
            status = "🔴 Opt-Out (Club C)"
        elif is_match:
            status = "🟢 Matched"
        else:
            status = "🟡 Deviated"

        st.write(f"**True Type:** `{true_t}`")
        st.write(f"**Chosen Plan:** `{chosen}`")
        st.write(f"**Intended Plan:** `{final_choice['principal_intended_plan']}`")
        st.write(f"**Final Payoff:** {final_choice['chosen_payoff']:.3f}")
        st.markdown(f"### Status: {status}")


        payoff_table = pd.DataFrame({
            "Plan": ["Safe", "Mixed", "Risky"],
            "Payoff": [
                final_choice["payoffs"]["safe"],
                final_choice["payoffs"]["mixed"],
                final_choice["payoffs"]["risky"],
            ]
        })
        st.table(payoff_table)

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
            st.experimental_rerun()
    with colB:
        if st.button("Clear ALL Simulations"):
            clear_database()
            st.success("All simulations deleted.")
            st.experimental_rerun()
