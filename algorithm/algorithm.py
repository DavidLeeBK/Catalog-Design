from dataclasses import dataclass, asdict, field
from typing import List, Dict, Literal
import math
import copy


# ============================================================
#  TYPE DEFINITIONS
# ============================================================

FirmType = Literal["safe", "mixed", "risky"]
Maturity = Literal["short", "long"]


# ============================================================
#  DATA CLASSES
# ============================================================

@dataclass
class PrincipalInfo:
    funding_need: float
    project_returns: List[float]   # [V1, V2]
    belief_safe_score: float
    pi_safe: float
    pi_mixed: float
    pi_risky: float


@dataclass
class PlanSide:
    zcb_price: float
    max_cap: float


@dataclass
class Plan:
    name: str
    club_A: PlanSide    # short-term side
    club_B: PlanSide    # long-term side


@dataclass
class FirmSimState:
    firm_id: int
    principal: PrincipalInfo
    true_type: str
    plans: Dict[str, List[Dict]]
    cap_limits: Dict[str, tuple[int, int]]
    firm_budget: Dict[str, float]
    locked: bool = False
    stable_rounds: int = 0
    last_chosen_plan: str | None = None
    history: List[Dict] = field(default_factory=list)


# ============================================================
#  BELIEF SYSTEM
# ============================================================

def _compute_beliefs_from_safe_score(safe_score: float) -> Dict[str, float]:
    """
    Map S ∈ [0,100] into probabilities over safe/mixed/risky.
    This is the same structure we used earlier.
    """
    S = max(0.0, min(100.0, safe_score))
    s = S / 100.0
    d = abs(S - 50.0)

    M_max = 0.5
    M_min = 0.01

    pi_mixed = M_min + (M_max - M_min) * (1.0 - d / 50.0)
    remaining = 1.0 - pi_mixed

    pi_safe = remaining * s
    pi_risky = remaining * (1.0 - s)

    return {
        "pi_safe": pi_safe,
        "pi_mixed": pi_mixed,
        "pi_risky": pi_risky,
    }

# ============================================================
#  CAPACITY SYSTEM
# ============================================================

def _allocate_firm_budgets(
    firms_meta: List[Dict],
    club_a_funds: float,
    club_b_funds: float,
) -> Dict[int, Dict[str, float]]:
    """
    Stage 1 (Across firms):
    Split the two global pools (Club A funds, Club B funds) across firms.

    We use the principal's belief about each firm to weight the split:
      weight_i = eps + pi_safe_i
    so that firms believed to be safer receive a larger portion of both pools.
    """
    
    BASE_SHARE = 0.6   # 60% equally shared
    BELIEF_SHARE = 0.4 # 40% belief-weighted
    eps = 0.1  # small floor so every firm can get *some* budget
    

    weights: List[float] = []
    for fm in firms_meta:
        w = eps + float(fm["principal"].pi_safe)
        weights.append(w)

    total_w = sum(weights) if sum(weights) > 0 else float(len(weights) or 1)

    out: Dict[int, Dict[str, float]] = {}
    N = len(firms_meta) or 1
    for fm, w in zip(firms_meta, weights):
        equal_frac = BASE_SHARE / N
        belief_frac = BELIEF_SHARE * (w / total_w if total_w > 0 else 1.0 / N)

        frac = equal_frac + belief_frac

        out[int(fm["firm_id"])] = {
            "A": float(club_a_funds) * frac,
            "B": float(club_b_funds) * frac,
        }
    return out

def _compute_catalog_cap_limits(
    budgetA: float,
    budgetB: float,
) -> Dict[str, tuple[int, int]]:
    """
    Stage 2 (Within a firm):
    Convert a firm's allocated budgets (budgetA, budgetB) into catalog-level cap limits.

    Design:
      - Each firm's budget is fixed from Stage 1.
      - Only a fraction of the budget is offered (OFFER_FRAC).
      - Catalog caps are deterministic slices of the offered budget using BASE_SHARE_A/B.
    """

    # Fraction of firm budget actually offered in catalogs
    OFFER_FRAC = 0.90

    # Fixed catalog shapes (literal shares, no renormalization)
    BASE_SHARE_A = {"safe": 0.80, "mixed": 0.60, "risky": 0.40}
    BASE_SHARE_B = {"safe": 0.40, "mixed": 0.60, "risky": 0.80}

    # Apply offer fraction
    offeredA = float(budgetA) * OFFER_FRAC
    offeredB = float(budgetB) * OFFER_FRAC

    limits: Dict[str, tuple[int, int]] = {}

    for cat in ["safe", "mixed", "risky"]:
        capA = int(round(offeredA * BASE_SHARE_A[cat]))
        capB = int(round(offeredB * BASE_SHARE_B[cat]))

        # Ensure non-negative integer caps
        limits[cat] = (max(0, capA), max(0, capB))

    return limits




# ============================================================
#  ORIGINAL ALPHA / PROJECT VALUE LOGIC (YOUR VERSION)
# ============================================================

def _base_alpha_for_type(type_: FirmType) -> float:
    """
    Intrinsic preference for short-term (Year 1) vs long-term (Year 2).
    Higher alpha = more weight on Year 1. Lower alpha = more weight on Year 2.
    """
    if type_ == "safe":
        return 0.7
    elif type_ == "mixed":
        return 0.5
    else:  # "risky"
        return 0.3


def _simple_project_value(
    type_: FirmType,
    D: float,
    V1: float,
    V2: float,
    maturity: Maturity,
    principal: PrincipalInfo,
) -> float:
    """
    Belief-aware project value function with:
      - base alpha by true type,
      - belief-weighted alpha,
      - maturity tilt,
      - discounting of Y1/Y2,
      - extra bonus for (safe, short) and (risky, long).
    """

    # 1) Base alpha by true type
    base_alpha = _base_alpha_for_type(type_)

    # 2) Belief-weighted alpha over all types
    alpha_safe = _base_alpha_for_type("safe")
    alpha_mixed = _base_alpha_for_type("mixed")
    alpha_risky = _base_alpha_for_type("risky")

    belief_alpha = (
        principal.pi_safe * alpha_safe +
        principal.pi_mixed * alpha_mixed +
        principal.pi_risky * alpha_risky
    )

    # 3) Blend true-type and belief view
    blend = 0.5
    alpha = (1.0 - blend) * base_alpha + blend * belief_alpha

    # 4) Maturity tilt
    tilt = 0.2
    if maturity == "short":
        alpha = min(1.0, alpha + tilt)
    else:  # "long"
        alpha = max(0.0, alpha - tilt)

    # 5) Discounting / boosting of V1 and V2
    discount_v1 = 1.1   # boost early returns
    discount_v2 = 0.85  # discount later returns

    v1_adj = discount_v1 * V1
    v2_adj = discount_v2 * V2

    expected_multiplier = alpha * v1_adj + (1.0 - alpha) * v2_adj

    # 6) Type–maturity bonus
    bonus_factor = 1.05
    bonus = 1.0

    if type_ == "safe" and maturity == "short":
        bonus = bonus_factor
    elif type_ == "risky" and maturity == "long":
        bonus = bonus_factor

    # 7) Final project value
    return D * expected_multiplier * bonus


def _expected_welfare_for_type(
    type_: FirmType,
    D: float,
    V1: float,
    V2: float,
    maturity: Maturity,
    principal: PrincipalInfo,
) -> float:
    """Welfare = project value - funding cost (D)."""
    value = _simple_project_value(type_, D, V1, V2, maturity, principal)
    return value - D


def _orientation_scores(principal: PrincipalInfo) -> Dict:
    """
    For each type and maturity, compute welfare, plus expected welfare
    under the principal's belief distribution.
    """
    D = principal.funding_need
    V1, V2 = principal.project_returns

    Ws_short = _expected_welfare_for_type("safe", D, V1, V2, "short", principal)
    Ws_long = _expected_welfare_for_type("safe", D, V1, V2, "long", principal)

    Wm_short = _expected_welfare_for_type("mixed", D, V1, V2, "short", principal)
    Wm_long = _expected_welfare_for_type("mixed", D, V1, V2, "long", principal)

    Wr_short = _expected_welfare_for_type("risky", D, V1, V2, "short", principal)
    Wr_long = _expected_welfare_for_type("risky", D, V1, V2, "long", principal)

    E_short = (
        principal.pi_safe * Ws_short +
        principal.pi_mixed * Wm_short +
        principal.pi_risky * Wr_short
    )
    E_long = (
        principal.pi_safe * Ws_long +
        principal.pi_mixed * Wm_long +
        principal.pi_risky * Wr_long
    )

    return {
        "safe": {"short": Ws_short, "long": Ws_long},
        "mixed": {"short": Wm_short, "long": Wm_long},
        "risky": {"short": Wr_short, "long": Wr_long},
        "expected_short": E_short,
        "expected_long": E_long,
    }


# ============================================================
#  TYPE–PLAN FIT MULTIPLIERS (AFTER PLAN CHOICE)
# ============================================================

TYPE_PLAN_MULTIPLIER: Dict[FirmType, Dict[str, float]] = {
    "safe": {
        "safe": 1.00,
        "mixed": 0.90,
        "risky": 0.80,
    },
    "mixed": {
        "safe": 0.90,
        "mixed": 1.00,
        "risky": 0.90,
    },
    "risky": {
        "safe": 0.80,
        "mixed": 0.90,
        "risky": 1.00,
    },
}

# ============================================================
#  INITIAL CATALOG CONSTRUCTION
# ============================================================

def _build_plans_from_orientation(principal: PrincipalInfo,
                                  scores: Dict,
                                  cap_limits: Dict[str, tuple[int, int]]) -> Dict[str, List[Dict]]:
    """
    Uses:
      - Base IC-inspired structure (your base prices + caps)
      - Old belief-driven favoring/unfavoring system
      - Short vs long welfare tilt (diff_short_long)
      - 3 symmetric menu items per catalog (Option A)
    """

    # 1) Welfare differences
    diff_safe = scores["safe"]["short"] - scores["safe"]["long"]
    diff_mixed = scores["mixed"]["short"] - scores["mixed"]["long"]
    diff_risky = scores["risky"]["short"] - scores["risky"]["long"]

    # 2) Base prices by catalog
    BASE_PRICES = {
        "safe":  (0.90, 0.85),
        "mixed": (0.88, 0.88),
        "risky": (0.85, 0.90),
    }

    # 3) Classify tilt
    def classify_tilt(diff: float) -> str:
        BIG = 20.0
        SMALL = 7.0
        if diff > BIG:
            return "strong_short"
        elif diff > SMALL:
            return "mild_short"
        elif diff < -BIG:
            return "strong_long"
        elif diff < -SMALL:
            return "mild_long"
        else:
            return "neutral"

    PRICE_TILT = {
        "strong_short": (+0.02, -0.02),
        "mild_short":   (+0.01, -0.01),
        "strong_long":  (-0.02, +0.02),
        "mild_long":    (-0.01, +0.01),
        "neutral":      (0.00,  0.00),
    }


    # 4) Favored System
    belief_list = [
        ("safe", principal.pi_safe),
        ("mixed", principal.pi_mixed),
        ("risky", principal.pi_risky),
    ]
    belief_list.sort(key=lambda x: x[1], reverse=True)

    favor_levels = {
        belief_list[0][0]: "favored",
        belief_list[1][0]: "neutral",
        belief_list[2][0]: "unfavored",
    }

    FAVOR_PRICE = {
        "favored": +0.01,
        "neutral": 0.0,
        "unfavored": -0.01,
    }

    # 5) Build a catalog (3 items)
    def build_single_catalog(type_name: str, diff: float) -> List[Dict]:
        base_short, base_long = BASE_PRICES[type_name]
        base_capS, base_capL = cap_limits[type_name]

        # FAVOR price adjustments
        favor_tag = favor_levels[type_name]
        favor_price_adj = FAVOR_PRICE[favor_tag]

        # Apply favor price tilt
        S_favored = base_short + favor_price_adj
        L_favored = base_long + favor_price_adj

        # Apply welfare tilt
        tilt = classify_tilt(diff)
        price_tilt_short, price_tilt_long = PRICE_TILT[tilt]

        S_center = S_favored + price_tilt_short
        L_center = L_favored + price_tilt_long

        S_center = max(0.50, min(0.99, S_center))
        L_center = max(0.50, min(0.99, L_center))


        # Round center ONCE
        S0 = round(S_center, 2)
        L0 = round(L_center, 2)

        capA, capB = cap_limits[type_name]

        items = []
        for k in range(3):
            S_item = round(S0 - 0.01 * k, 2)
            L_item = round(L0 + 0.01 * k, 2)

            short_cap_k = int(round(capA * (1.0 - 0.05 * k)))
            long_cap_k  = int(round(capB * (1.0 + 0.05 * k)))

            items.append({
                "short_price": float(f"{max(0.50, min(0.99, S_item)):.2f}"),
                "long_price":  float(f"{max(0.50, min(0.99, L_item)):.2f}"),
                "short_cap": max(0, short_cap_k),
                "long_cap":  max(0, long_cap_k),
            })




        return items

    # 7) Build catalogs
    return {
        "safe":  build_single_catalog("safe", diff_safe),
        "mixed": build_single_catalog("mixed", diff_mixed),
        "risky": build_single_catalog("risky", diff_risky),
    }

# ============================================================
#  PAYOFF ENGINE — GRID SEARCH, EXACT D FUNDING
# ============================================================

def _compute_plan_payoff(
    true_type: FirmType,
    D: int,
    V1: float,
    V2: float,
    plan: Plan,
) -> Dict:
    """
    Compute RAW payoff for a single catalog item (Plan).

    Optimization logic (lexicographic, deterministic):

    1) Minimize total number of bonds issued (A + B)
    2) Among ties, minimize funds borrowed Ps*A + Pl*B
    3) Among ties, break indifference using firm-type preference:
         SAFE  -> prefer more short bonds
         MIXED -> prefer balanced borrowing
         RISKY -> prefer more long bonds

    Payoff definition (unchanged):
        payoff = R - (A + B)

    Returns a dictionary with borrowing decision and payoff details.
    """

    # --------------------------------------------
    # Extract plan parameters
    # --------------------------------------------
    Ps = plan.club_A.zcb_price      # short-term price
    Pl = plan.club_B.zcb_price      # long-term price
    capS = plan.club_A.max_cap      # short-term cap
    capL = plan.club_B.max_cap      # long-term cap

    # --------------------------------------------
    # Expected return (project side)
    # --------------------------------------------
    R = D * 1.5   # keep your original rule

    # --------------------------------------------
    # Helper: type-aligned deterministic tie-break
    # --------------------------------------------
    def tie_break_score(A: int, B: int) -> float:
        """
        Lower score is preferred.
        Used ONLY when (A+B) and funds borrowed are identical.
        """
        if true_type == "safe":
            return -(A - B)          # prefer more short
        elif true_type == "mixed":
            return abs(A - B)        # prefer balance
        elif true_type == "risky":
            return -(B - A)          # prefer more long
        else:
            return 0.0

    # --------------------------------------------
    # Search for optimal (A, B)
    # --------------------------------------------
    best_total_caps = None
    best_funds = None
    best_tie = None
    best_A = 0
    best_B = 0

    for A in range(0, capS + 1):
        # Remaining funds needed after issuing A short bonds
        remaining = D - Ps * A

        if remaining <= 0:
            B_min = 0
        else:
            B_min = math.ceil(remaining / Pl)

        # Check feasibility
        if B_min < 0 or B_min > capL:
            continue

        total_caps = A + B_min
        funds = A * Ps + B_min * Pl   # DO NOT round here
        tie = tie_break_score(A, B_min)

        # First feasible solution
        if best_total_caps is None:
            best_total_caps = total_caps
            best_funds = funds
            best_tie = tie
            best_A = A
            best_B = B_min
            continue

        # 1️⃣ Minimize total number of bonds
        if total_caps < best_total_caps:
            best_total_caps = total_caps
            best_funds = funds
            best_tie = tie
            best_A = A
            best_B = B_min

        # 2️⃣ Same number of bonds → minimize funds borrowed
        elif total_caps == best_total_caps:
            if funds < best_funds:
                best_funds = funds
                best_tie = tie
                best_A = A
                best_B = B_min

            # 3️⃣ Same bonds & same funds → deterministic tie-break
            elif abs(funds - best_funds) < 1e-9:
                if tie < best_tie:
                    best_tie = tie
                    best_A = A
                    best_B = B_min

    # --------------------------------------------
    # Infeasible case
    # --------------------------------------------
    if best_total_caps is None:
        return {
            "raw_payoff": float("-inf"),
            "A": 0,
            "B": 0,
            "zcb_prices": (Ps, Pl),
            "caps": (capS, capL),
            "funds_borrowed": 0.0,
            "expected_return": R,
        }

    # --------------------------------------------
    # Final payoff computation
    # --------------------------------------------
    funds_borrowed = round(best_A * Ps + best_B * Pl, 2)
    payoff = R - best_total_caps

    return {
        "raw_payoff": payoff,
        "A": best_A,
        "B": best_B,
        "zcb_prices": (Ps, Pl),
        "caps": (capS, capL),
        "funds_borrowed": funds_borrowed,
        "expected_return": R,
    }

# ============================================================
#  FIRM CHOICE GIVEN CURRENT CATALOG
# ============================================================

def _principal_intended_plan_name(principal: PrincipalInfo) -> str:
    weights = {
        "safe": principal.pi_safe,
        "mixed": principal.pi_mixed,
        "risky": principal.pi_risky,
    }
    return max(weights, key=weights.get)


def _simulate_firm_choice(true_type: FirmType,
                          principal: PrincipalInfo,
                          catalogs: Dict[str, List[Dict]]) -> Dict:
    """
    OPTION B IMPLEMENTATION:
      - Evaluate each catalog's menu items using RAW payoff (no type multiplier)
      - Pick best item within each catalog
      - AFTER that, apply type plan multiplier to compare catalogs
      - Choose catalog with highest adjusted payoff
      - Apply opt-out if best adjusted payoff <= 0
    """

    D = int(principal.funding_need)
    V1, V2 = principal.project_returns

    # FOR RAW PAYOFFS
    raw_payoffs = {}          # raw best payoff per catalog (no multiplier)
    best_item_index = {}      # which menu item index was best
    plan_details = {}         # raw payoff details
    effective_plans = {}      # Plan object for the best item of the catalog

    # --------------------------------------------
    # 1. Evaluate RAW payoffs within each catalog
    # --------------------------------------------
    for plan_name, items in catalogs.items():
        best_raw = float("-inf")
        best_details = None
        best_plan = None
        best_idx = None

        for idx, item in enumerate(items):
            # Convert menu item to a Plan object
            plan_obj = Plan(
                name=plan_name,
                club_A=PlanSide(
                    zcb_price=item["short_price"],
                    max_cap=item["short_cap"],
                ),
                club_B=PlanSide(
                    zcb_price=item["long_price"],
                    max_cap=item["long_cap"],
                )
            )

            # Compute RAW payoff (we remove multiplier inside this function)
            details = _compute_plan_payoff(
                true_type, D, V1, V2, plan_obj
            )
            raw = details["raw_payoff"]

            if raw > best_raw:
                best_raw = raw
                best_details = details
                best_plan = plan_obj
                best_idx = idx
                
        if best_details is None:
            raw_payoffs[plan_name] = float("-inf")
            plan_details[plan_name] = None
            effective_plans[plan_name] = None
            best_item_index[plan_name] = None
            continue

        raw_payoffs[plan_name] = best_raw

        best_details = dict(best_details)
        best_details["item_index"] = best_idx
        best_details["item_prices_caps"] = items[best_idx]

        plan_details[plan_name] = best_details
        effective_plans[plan_name] = best_plan
        best_item_index[plan_name] = best_idx

    # --------------------------------------------
    # 2. Apply TYPE–PLAN MULTIPLIER at catalog level
    # --------------------------------------------
    adjusted_payoffs = {}
    for plan_name, raw_val in raw_payoffs.items():
        mult = TYPE_PLAN_MULTIPLIER[true_type][plan_name]
        adjusted_payoffs[plan_name] = raw_val * mult

    # --------------------------------------------
    # 3. Choose catalog with highest adjusted payoff
    # --------------------------------------------
    best_catalog = max(adjusted_payoffs, key=lambda k: adjusted_payoffs[k])
    best_adjusted_payoff = adjusted_payoffs[best_catalog]

    # Opt-out rule
    if best_adjusted_payoff <= 0:
        chosen_plan = "opt_out"
        chosen_payoff = 0.0
        chosen_index = None
    else:
        chosen_plan = best_catalog
        chosen_payoff = best_adjusted_payoff
        chosen_index = best_item_index[best_catalog]

    intended = _principal_intended_plan_name(principal)

    return {
        "true_type": true_type,
        "principal_intended_plan": intended,
        "chosen_plan": chosen_plan,
        "chosen_item_index": chosen_index,
        "deviation": (chosen_plan not in [intended, "opt_out"]),
        "non_participation": (chosen_plan == "opt_out"),
        "raw_payoffs": raw_payoffs,
        "adjusted_payoffs": adjusted_payoffs,
        "plan_details": plan_details,
        "chosen_payoff": chosen_payoff,
    }


# ============================================================
#  CATALOG UPDATE RULES
# ============================================================

def _fmt_price(x: float) -> float:
    return float(f"{max(0.50, min(0.99, x)):.2f}")

def _update_plans_for_firm(state: FirmSimState, choice_info: Dict) -> None:
    """
    Each catalog updates its prices based on its OWN hypothetical usage,
    regardless of whether it was chosen.

      - If usage is HIGH (near cap)  -> price FALLS by 0.01
      - If usage is LOW (far cap)    -> price RISES by 0.01
      - Otherwise                   -> no change
    """

    STEP = 0.01
    CAP_GROWTH = 1.05
    
    plan_details = choice_info.get("plan_details", {})
    chosen_plan = choice_info.get("chosen_plan")
    adjusted_payoffs = choice_info.get("adjusted_payoffs", {})
    
    firm_capA = int(round(state.firm_budget["A"]))
    firm_capB = int(round(state.firm_budget["B"]))
    
    if chosen_plan == "opt_out" and adjusted_payoffs:

        # Identify the "closest" catalog (highest adjusted payoff)
        target_catalog = max(adjusted_payoffs, key=lambda k: adjusted_payoffs[k])

        items = state.plans[target_catalog]

        # Expand caps
        for item in items:
            item["short_cap"] = int(round(item["short_cap"] * CAP_GROWTH))
            item["long_cap"]  = int(round(item["long_cap"]  * CAP_GROWTH))

        # ---- Renormalize if firm hard caps exceeded ----
        total_A = sum(items[0]["short_cap"] for items in state.plans.values())
        total_B = sum(items[0]["long_cap"]  for items in state.plans.values())

        if total_A > firm_capA:
            scale_A = firm_capA / total_A
            for cat_items in state.plans.values():
                for item in cat_items:
                    item["short_cap"] = int(round(item["short_cap"] * scale_A))

        if total_B > firm_capB:
            scale_B = firm_capB / total_B
            for cat_items in state.plans.values():
                for item in cat_items:
                    item["long_cap"] = int(round(item["long_cap"] * scale_B))
                                 
    if not plan_details:
        return  # safety guard

    for cat_name, items in state.plans.items():
        details = plan_details.get(cat_name)
        if not details:
            continue

        capA = details["A"]
        capB = details["B"]
        max_capA, max_capB = details["caps"]

        # Compute usage ratio for THIS catalog
        usageA = (capA / max_capA) if capA > 0 else 0.0
        usageB = (capB / max_capB) if capB > 0 else 0.0
        usage = max(usageA, usageB)

        # Decide price adjustment (your rule)
        if usage >= 0.85:
            delta = -STEP
        elif usage <= 0.45:
            delta = +STEP
        else:
            delta = 0.0

        # Apply update to ALL items in this catalog
        for item in items:
            item["short_price"] = _fmt_price(item["short_price"] + delta)
            item["long_price"]  = _fmt_price(item["long_price"]  + delta)

# ============================================================
#  PUBLIC ENTRYPOINT — MULTI-ROUND WITH LOCKING
# ============================================================

def run_catalog_simulation(scenario: Dict) -> Dict:
    """
    Multi-round simulation with per-firm catalog adaptation
    and locking when choices stabilize.

    A firm is locked when it chooses the same non-opt-out plan
    for STABLE_THRESHOLD consecutive rounds.
    """

    MAX_ROUNDS = 10
    STABLE_THRESHOLD = 3

    firms_input = scenario.get("firms", [])

    # 1) Initial principal info, firm budgets, and cap-limited catalogs
    firm_states: List[FirmSimState] = []

    club_a_funds = float(scenario.get("club_a_funds", 0.0))
    club_b_funds = float(scenario.get("club_b_funds", 0.0))

    # ---- Pass 1: build principals (beliefs) for each firm ----
    firms_meta: List[Dict] = []
    for f in firms_input:
        firm_id = int(f["id"])
        p_raw = f["principal"]
        a_raw = f["actual"]

        D = float(p_raw["funding_need"])
        V1, V2 = p_raw["project_returns"]
        safe_score = float(p_raw["belief_safe_score"])
        true_type: FirmType = a_raw["type"]

        beliefs = _compute_beliefs_from_safe_score(safe_score)

        principal = PrincipalInfo(
            funding_need=D,
            project_returns=[V1, V2],
            belief_safe_score=safe_score,
            pi_safe=beliefs["pi_safe"],
            pi_mixed=beliefs["pi_mixed"],
            pi_risky=beliefs["pi_risky"],
        )

        firms_meta.append({
            "firm_id": firm_id,
            "principal": principal,
            "true_type": true_type,
        })

    # ---- Stage 1: allocate global pools across firms (budgets) ----
    budgets_by_firm = _allocate_firm_budgets(firms_meta, club_a_funds, club_b_funds)

    # ---- Stage 2: firm budgets -> catalog cap limits -> catalogs ----
    for fm in firms_meta:
        firm_id = fm["firm_id"]
        principal = fm["principal"]
        true_type = fm["true_type"]

        budgets = budgets_by_firm.get(firm_id, {"A": 0.0, "B": 0.0})
        cap_limits = _compute_catalog_cap_limits(budgets["A"], budgets["B"])

        scores = _orientation_scores(principal)
        plans = _build_plans_from_orientation(principal, scores, cap_limits)

        firm_states.append(
            FirmSimState(
                firm_id=firm_id,
                principal=principal,
                true_type=true_type,
                plans=plans,
                cap_limits=cap_limits,
                firm_budget={
                    "A": budgets["A"],
                    "B": budgets["B"],
                },
            )
        )

    welfare_hints: List[float] = []
    realized_payoffs: List[float] = []
    rounds_run = 0

    # 2) Iteration loop
    for r in range(1, MAX_ROUNDS + 1):
        rounds_run = r
        all_locked = True

        for state in firm_states:
            if state.locked:
                continue

            all_locked = False

            # compute orientation welfare hint this round
            scores = _orientation_scores(state.principal)
            welfare_hint = max(scores["expected_short"], scores["expected_long"])
            welfare_hints.append(welfare_hint)

            # simulate choice under current plans
            choice_info = _simulate_firm_choice(state.true_type, state.principal, state.plans)
            chosen = choice_info["chosen_plan"]

            # update lock / stability
            if chosen == state.last_chosen_plan and chosen != "opt_out":
                state.stable_rounds += 1
            else:
                state.stable_rounds = 1 if chosen != "opt_out" else 0

            if state.stable_rounds >= STABLE_THRESHOLD and chosen != "opt_out":
                state.locked = True

            state.last_chosen_plan = chosen

            # record history for this iteration
            snapshot_plans = copy.deepcopy(state.plans)


            state.history.append({
                "iteration": r,
                "plans": snapshot_plans,
                "choice": choice_info,
            })

            realized_payoffs.append(choice_info["chosen_payoff"])

            # if not locked, update catalog for next round
            if not state.locked:
                _update_plans_for_firm(state, choice_info)

        if all_locked:
            break

    # 3) Aggregate results
    avg_welfare_hint = sum(welfare_hints) / len(welfare_hints) if welfare_hints else 0.0
    avg_realized_payoff = sum(realized_payoffs) / len(realized_payoffs) if realized_payoffs else 0.0

    results: List[Dict] = []

    for state in firm_states:
        if state.history:
            final_choice = state.history[-1]["choice"]
        else:
            # fallback if no rounds ran for some reason
            final_choice = {
                "true_type": state.true_type,
                "principal_intended_plan": _principal_intended_plan_name(state.principal),
                "chosen_plan": "opt_out",
                "deviation": False,
                "non_participation": True,
                "payoffs": {"safe": 0.0, "mixed": 0.0, "risky": 0.0},
                "plan_details": {},
                "chosen_payoff": 0.0,
            }

        results.append({
            "firm_id": state.firm_id,
            "principal": {
                "funding_need": state.principal.funding_need,
                "project_returns": state.principal.project_returns,
                "belief_safe_score": state.principal.belief_safe_score,
                "pi_safe": state.principal.pi_safe,
                "pi_mixed": state.principal.pi_mixed,
                "pi_risky": state.principal.pi_risky,
            },
            "capacity": {
                "budget": state.firm_budget,
                "offered": {
                    "A": round(state.firm_budget["A"] * 0.90, 2),
                    "B": round(state.firm_budget["B"] * 0.90, 2),
                },
                "catalog_caps": state.cap_limits,
            },
            "plans": {
                name: items
                for name, items in state.plans.items()
            },
            "locked": state.locked,
            "expected_welfare_hint": None,  # per-round hints are in history if needed
            "choice": final_choice,
            "history": state.history,
        })

    return {
        "welfare": avg_welfare_hint,
        "realized_payoff": avg_realized_payoff,
        "iterations": rounds_run,
        "results": results,
    }
