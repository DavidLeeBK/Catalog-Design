import streamlit as st
import time
from algorithm.algorithm import run_catalog_simulation
from data.db import init_db, save_simulation

st.markdown(
    """
    <style>
    /* Main app font */
    html, body, [class*="css"]  {
        font-size: 18px !important;
    }

    /* Page title */
    h1 {
        font-size: 40px !important;
    }

    /* Section headers */
    h2 {
        font-size: 32px !important;
    }

    h3 {
        font-size: 26px !important;
    }

    h4 {
        font-size: 22px !important;
    }

    /* Labels above inputs */
    label {
        font-size: 18px !important;
    }

    /* Input text, dropdowns, buttons */
    input, textarea, select, button {
        font-size: 18px !important;
    }

    /* Tables */
    .stDataFrame, .stTable {
        font-size: 17px !important;
    }

    /* Metrics */
    div[data-testid="metric-container"] {
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
#  Page setup
# =========================================================
st.set_page_config(page_title="Delegated Lending Simulator", layout="wide")
st.title("📈 Delegated Lending Simulator")

init_db()

# =========================================================
#  Simulation name
# =========================================================
sim_name = st.text_input("🧠 Simulation Name", value="Untitled Simulation", help="Enter a name for this simulation")

st.divider()

# =========================================================
#  GLOBAL CONFIGURATION — Only number of firms needed
# =========================================================
st.markdown("### 🔧 Global Configuration")

num_firms = st.number_input("Number of firms", 1, 10, 3)
st.caption("Number of firms participating in the delegated lending mechanism.")

c1,c2 = st.columns(2)
with c1:
    st.number_input("Club A funds available", 0.0, None, 500.0, 10.0, key="club_a_funds")
with c2:
    st.number_input("Club B funds available", 0.0, None, 500.0, 10.0, key="club_b_funds")
st.caption("Total funds available in each lending club.")

st.divider()

# =========================================================
#  FIRMS CONFIGURATION (new simplified version)
# =========================================================
st.markdown("### 🏢 Firms Setup — Principal’s Expectations & Actual Types")

firms = []

with st.container(border=True):
    for i in range(num_firms):
        st.markdown(f"## Firm {i+1}")

        # ----------------- Principal’s Expectations -----------------
        st.markdown("### Principal’s Expected Information")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            belief_safe_score = st.number_input(
                f"Believes firm is SAFE (%) (Firm {i+1})",
                0, 100, 50, 1,
                key=f"belief_safe_{i}"
            )

        with c2:
            funding_need = st.number_input(
                f"Funding need (Firm {i+1})",
                0.0, None, 120.0, 10.0,
                key=f"funding_{i}"
            )

        with c3:
            V1 = st.number_input(
                f"Expected return Year 1 (Firm {i+1})",
                value=0.9, step=0.1, format="%.2f",
                key=f"V1_{i}"
            )

        with c4:
            V2 = st.number_input(
                f"Expected return Year 2 (Firm {i+1})",
                value=1.2, step=0.1, format="%.2f",
                key=f"V2_{i}"
            )

        # ----------------- Actual Firm Type -----------------
        st.markdown("### Actual Type (Hidden Truth from the Principal)")
        actual_type = st.selectbox(
            f"Actual type (Firm {i+1})",
            ["Safe", "Mixed", "Risky"],
            index=1,
            key=f"actual_{i}"
        )

        # Build firm dict
        firms.append({
            "id": i + 1,
            "principal": {
                "funding_need": funding_need,
                "project_returns": [V1, V2],
                "belief_safe_score": belief_safe_score,
            },
            "actual": {"type": actual_type.lower()},
        })

        st.divider()

# =========================================================
#  RUN SIMULATION (CONFIRMATION MODEL)
# =========================================================
st.markdown("### ▶️ Run Simulation")

if "confirm_mode" not in st.session_state:
    st.session_state.confirm_mode = False

# STEP 1 — user clicks button → go to confirmation mode
if not st.session_state.confirm_mode:
    if st.button("Run Simulation", use_container_width=True, type="primary"):

        # Store scenario
        st.session_state.scenario = {
            "firms": firms,
            "club_a_funds": float(st.session_state.club_a_funds),
            "club_b_funds": float(st.session_state.club_b_funds),
        }


        st.session_state.confirm_mode = True
        st.rerun()

# STEP 2 — confirmation modal (pretty popup)
else:
    st.markdown(
        """
        <div style="
            padding: 25px;
            border-radius: 12px;
            margin-top: 20px;
        ">
        """,
        unsafe_allow_html=True
    )

    st.markdown("## ⚠️ Please review your inputs before running the simulation")

    # ----- Firms Overview -----
    st.markdown("### 🏢 Firms Overview")
    firm_rows = []
    for firm in st.session_state.scenario["firms"]:
        firm_rows.append({
            "Firm ID": firm["id"],
            "Belief SAFE (%)": firm["principal"]["belief_safe_score"],
            "Funding Need": firm["principal"]["funding_need"],
            "Y1 Return": firm["principal"]["project_returns"][0],
            "Y2 Return": firm["principal"]["project_returns"][1],
            "Actual Type": firm["actual"]["type"].capitalize(),
        })

    st.dataframe(firm_rows, use_container_width=True, hide_index=True)

    # ----- Action Buttons -----
    col_confirm, col_cancel = st.columns(2)

    with col_confirm:
        if st.button("✅ Confirm & Run", use_container_width=True):
            output = run_catalog_simulation(st.session_state.scenario)
            st.success("✅ Simulation complete!")

            save_simulation(
                sim_name,
                num_firms=len(st.session_state.scenario["firms"]),
                num_plans=3,  # Safe, Mixed, Risky
                welfare=output.get("welfare", 0),
                iterations=output.get("iterations", 1),
                results=output.get("results", {})
            )

            st.session_state.confirm_mode = False
            st.info("Redirecting to Report page...")
            time.sleep(1)
            st.switch_page("pages/result.py")

    with col_cancel:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.confirm_mode = False
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
