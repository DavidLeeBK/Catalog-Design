from dataclasses import dataclass, asdict
from typing import List, Dict, Literal
import math


FirmType = Literal["safe", "mixed", "risky"]
Maturity = Literal["short", "long"]

# ============================================================
#  DATA CLASSES
# ============================================================

@dataclass
class PrincipalInfo:
    funding_need: float              # D
    project_returns: List[float]     # [V1, V2]
    belief_safe_score: float         # S in [0,100]
    pi_safe: float
    pi_mixed: float
    pi_risky: float


@dataclass
class PlanSide:
    zcb_price: float     # Zero-coupon bond price per unit face
    max_cap: float       # Max amount that can be borrowed


@dataclass
class Plan:
    name: str            # "safe", "mixed", "risky"
    club_A: PlanSide     # short-term
    club_B: PlanSide     # long-term


@dataclass
class FirmCatalogResult:
    firm_id: int
    principal: PrincipalInfo
    plans: Dict[str, Plan]     # "safe", "mixed", "risky"
    opt_out: bool
    expected_welfare_hint: float
    choice: Dict              # added in Step 5: firm selection info


# ============================================================
#  BELIEF DECOMPOSITION
# ============================================================

def _compute_beliefs_from_safe_score(safe_score: float) -> Dict[str, float]:
    """
    Map a SAFE score S in [0,100] into probabilities over
    safe / mixed / risky.

    Intuition:
      - High S  -> mostly safe
      - Low S   -> mostly risky
      - Middle S (~50) -> mostly mixed

    pi_mixed is high near S=50 and low near the extremes.
    The remaining probability is split between safe and risky
    according to S.
    """
    S = max(0.0, min(100.0, safe_score))  # clamp to [0,100]
    s = S / 100.0
    d = abs(S - 50.0)

    # Tunable hyperparameters:
    M_max = 0.5   # mixed probability when S = 50
    M_min = 0.01   # mixed probability when S = 0 or 100

    # 1) Mixed probability: high at center, low at extremes
    pi_mixed = M_min + (M_max - M_min) * (1.0 - d / 50.0)

    # 2) Remaining mass goes to safe vs risky
    remaining = 1.0 - pi_mixed
    safe_weight = s
    risky_weight = 1.0 - s

    pi_safe = remaining * safe_weight
    pi_risky = remaining * risky_weight

    return {
        "pi_safe": pi_safe,
        "pi_mixed": pi_mixed,
        "pi_risky": pi_risky,
    }


# ============================================================
#  PROJECT VALUE & WELFARE FUNCTIONS
# ============================================================

def _base_alpha_for_type(type_: FirmType) -> float:
    """
    Intrinsic preference for short-term (Year 1) vs long-term (Year 2).
    Higher alpha = more weight on Year 1. Lower alpha = more weight on Year 2.
    Assume that the true firm types have fixed alphas as follows.
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

    # -----------------------------
    # 1) Base alpha by true type
    # -----------------------------
    base_alpha = _base_alpha_for_type(type_)

    # 2) Belief-weighted alpha over all types
    alpha_safe  = _base_alpha_for_type("safe")
    alpha_mixed = _base_alpha_for_type("mixed")
    alpha_risky = _base_alpha_for_type("risky")

    belief_alpha = (
        principal.pi_safe  * alpha_safe +
        principal.pi_mixed * alpha_mixed +
        principal.pi_risky * alpha_risky
    )

    # 3) Blend true-type and belief view
    #    blend in [0,1]; e.g. 0.5 = equal weight
    blend = 0.5
    alpha = (1.0 - blend) * base_alpha + blend * belief_alpha

    # 4) Apply maturity tilt
    tilt = 0.2
    if maturity == "short":
        alpha = min(1.0, alpha + tilt)
    else:  # "long"
        alpha = max(0.0, alpha - tilt)

    # 5) Apply discounting to Y1/Y2
    discount_v1 = 1.1   # boost early returns
    discount_v2 = 0.85  # discount later returns

    v1_adj = discount_v1 * V1
    v2_adj = discount_v2 * V2   

    expected_multiplier = alpha * v1_adj + (1.0 - alpha) * v2_adj

    # 6) Type–maturity bonus:
    #    - SAFE + SHORT  gets bonus_factor
    #    - RISKY + LONG  gets bonus_factor
    bonus_factor = 1.05
    bonus = 1.0

    if type_ == "safe" and maturity == "short":
        bonus = bonus_factor
    elif type_ == "risky" and maturity == "long":
        bonus = bonus_factor

    # 7) Final project value
    return D * expected_multiplier * bonus



def _type_base_project_value(
    type_: FirmType,
    D: float,
    V1: float,
    V2: float,
) -> float:
    """
    Maturity-agnostic project value for STEP 5 (firm's payoff).

    Here we use ONLY the true type's intrinsic alpha and do NOT
    apply any maturity tilt or belief adjustments, because the firm
    knows its own type when evaluating its ex-post project value.
    """
    alpha = _base_alpha_for_type(type_)

    
    expected_multiplier = alpha * V1 + (1.0 - alpha) * V2
    return D * expected_multiplier


def _expected_welfare_for_type( 
    type_: FirmType,
    D: float,
    V1: float,
    V2: float,
    maturity: Maturity,
    principal: PrincipalInfo,
) -> float:
    """Welfare = project value - funding cost (D)."""
    expected_welfare = _simple_project_value(type_, D, V1, V2, maturity, principal)
    cost = D
    return expected_welfare - cost


def _orientation_scores(principal: PrincipalInfo) -> Dict:
    """
    For each type and maturity, compute welfare, plus expected welfare
    under the principal's belief distribution.
    """
    D = principal.funding_need
    V1, V2 = principal.project_returns

    Ws_short = _expected_welfare_for_type("safe",  D, V1, V2, "short", principal)
    Ws_long  = _expected_welfare_for_type("safe",  D, V1, V2, "long",  principal)
    
    Wm_short = _expected_welfare_for_type("mixed", D, V1, V2, "short", principal)
    Wm_long  = _expected_welfare_for_type("mixed", D, V1, V2, "long",  principal)
    
    Wr_short = _expected_welfare_for_type("risky", D, V1, V2, "short", principal)
    Wr_long  = _expected_welfare_for_type("risky", D, V1, V2, "long",  principal)

    E_short = (
        principal.pi_safe  * Ws_short +
        principal.pi_mixed * Wm_short +
        principal.pi_risky * Wr_short
    )
    E_long  = (
        principal.pi_safe  * Ws_long +
        principal.pi_mixed * Wm_long +
        principal.pi_risky * Wr_long
    )

    return {
        "safe":  {"short": Ws_short, "long": Ws_long},
        "mixed": {"short": Wm_short, "long": Wm_long},
        "risky": {"short": Wr_short, "long": Wr_long},
        "expected_short": E_short,
        "expected_long": E_long,
    }


# ============================================================
# BUILD CATALOG (SAFE / MIXED / RISKY PLANS)
# ============================================================

def _build_plans_from_orientation(principal: PrincipalInfo,
                                  scores: Dict) -> Dict[str, Plan]:
    """
    IC + Opt-Out aware catalog construction.

    Design goals:
      1. Use expected welfare orientation (short vs long) to tilt all plans slightly.
      2. Give each plan (safe/mixed/risky) base prices & caps that align with its
         intended true type:
             - SAFE: short-oriented, low leverage
             - MIXED: balanced
             - RISKY: long-oriented, high leverage
      3. Use the belief system (pi_safe, pi_mixed, pi_risky) to decide which plan is
         favored / neutral / unfavored ACROSS plans:
             - highest belief  -> favored
             - middle belief   -> neutral
             - lowest belief   -> unfavored
         This affects both ZCB price (small bonus/penalty) and caps (big effect).
      4. Ensure that for some belief–plan combinations, total caps < D so the firm
         CANNOT raise full funding, making that plan invalid and pushing the firm
         either to another plan or to Club C (opt-out).
    """

    D = principal.funding_need

    # --------------------------------------------------
    # 1) ORIENTATION TILT (short vs long) FROM EXPECTED WELFARE
    # --------------------------------------------------
    # Use the already computed expected welfares in 'scores'
    E_short = scores["expected_short"]
    E_long  = scores["expected_long"]
    delta = E_short - E_long  # >0 => world more short-oriented in expectation

    # Smoothly map delta -> [-1, 1]
    orientation_scale = 10.0
    orientation_strength = math.tanh(delta / orientation_scale)

    # Small price tilt so we don't destroy the IC base structure
    tilt_strength = 0.01
    orient_adjust_short = +tilt_strength * orientation_strength
    orient_adjust_long  = -tilt_strength * orientation_strength

    # --------------------------------------------------
    # 2) IC-CONSISTENT BASE TABLES (PLAN x MATURITY)
    # --------------------------------------------------
    # Higher price = cheaper borrowing = more attractive to the firm.
    # These are the "skeleton" incentives BEFORE beliefs or orientation tilt.

    BASE_PRICES = {
        # SAFE: short-oriented, modest long
        "safe": {
            "short": 0.95,
            "long":  0.88,
        },
        # MIXED: balanced, middle-of-the-road
        "mixed": {
            "short": 0.93,
            "long":  0.90,
        },
        # RISKY: long-oriented; long must be attractive to risky types
        "risky": {
            "short": 0.90,
            "long":  0.94,
        },
    }

    # Caps as multiples of D.
    # NOTE: totals are NOT huge; some will fail if also marked "unfavored".
    BASE_CAP_MULT = {
        # SAFE: decent short funding, small long; safe types don't want huge leverage.
        "safe": {
            "short": 0.70,   # 0.7D
            "long":  0.50,   # 0.5D  => total 1.2D before belief scaling
        },
        # MIXED: balanced, slightly more total capacity
        "mixed": {
            "short": 0.70,   # 0.7D
            "long":  0.70,   # 0.7D  => total 1.4D
        },
        # RISKY: long-heavy; risky types want big upside on long
        "risky": {
            "short": 0.50,   # 0.5D
            "long":  0.70,   # 0.9D  => total 1.4D
        },
    }

    # --------------------------------------------------
    # 3) BELIEF-DRIVEN FAVOR SYSTEM (ACROSS PLANS)
    # --------------------------------------------------
    # Use existing beliefs on types to decide which whole plan is favored.
    belief_list = [
        ("safe",  principal.pi_safe),
        ("mixed", principal.pi_mixed),
        ("risky", principal.pi_risky),
    ]
    belief_list.sort(key=lambda x: x[1], reverse=True)  # highest belief first

    # highest -> favored, middle -> neutral, lowest -> unfavored
    favor_levels: Dict[str, str] = {}
    if len(belief_list) == 3:
        favor_levels[belief_list[0][0]] = "favored"
        favor_levels[belief_list[1][0]] = "neutral"
        favor_levels[belief_list[2][0]] = "unfavored"
    else:
        # fallback (should not really happen)
        for name, _ in belief_list:
            favor_levels[name] = "neutral"

    # How favor level affects prices (small) and caps (large).
    FAVOR_PRICE_BONUS = {
        "favored":   +0.01,
        "neutral":    0.00,
        "unfavored": -0.01,
    }
    FAVOR_CAP_SCALE = {
        "favored":   1.20,   # +20% capacity
        "neutral":   1.00,   # unchanged
        "unfavored": 0.50,   # -50% capacity => often cannot reach D
    }

    # --------------------------------------------------
    # 4) HELPER: BUILD ONE PLAN-SIDE (SHORT or LONG)
    # --------------------------------------------------
    def build_side(plan_name: str, maturity: Maturity) -> PlanSide:
        # IC skeleton
        base_price = BASE_PRICES[plan_name][maturity]
        base_cap_mult = BASE_CAP_MULT[plan_name][maturity]

        # Orientation tilt: if world is short-oriented,
        # - short prices go up (cheaper)
        # - long prices go down (more expensive)
        if maturity == "short":
            price = base_price + orient_adjust_short
        else:
            price = base_price + orient_adjust_long

        # Belief-driven favor system (across plans)
        favor = favor_levels[plan_name]
        price += FAVOR_PRICE_BONUS[favor]

        # Clamp prices to a reasonable range
        price = max(0.5, min(0.995, price))

        # Caps = base cap * belief-scaling * D
        cap_mult = base_cap_mult * FAVOR_CAP_SCALE[favor]
        cap = cap_mult * D

        return PlanSide(
            zcb_price=round(price, 3),
            max_cap=round(cap, 2),
        )

    # --------------------------------------------------
    # 5) BUILD THE THREE PLANS
    # --------------------------------------------------
    safe_plan = Plan(
        name="safe",
        club_A=build_side("safe", "short"),
        club_B=build_side("safe", "long"),
    )

    mixed_plan = Plan(
        name="mixed",
        club_A=build_side("mixed", "short"),
        club_B=build_side("mixed", "long"),
    )

    risky_plan = Plan(
        name="risky",
        club_A=build_side("risky", "short"),
        club_B=build_side("risky", "long"),
    )

    return {
        "safe": safe_plan,
        "mixed": mixed_plan,
        "risky": risky_plan,
    }

# ============================================================
#  STEP 5: FIRM CHOICE GIVEN TRUE TYPE
# ============================================================

def _principal_intended_plan_name(principal: PrincipalInfo) -> str:
    """
    Heuristic: the principal's intended plan is the type with the highest belief weight.
    """
    weights = {
        "safe": principal.pi_safe,
        "mixed": principal.pi_mixed,
        "risky": principal.pi_risky,
    }
    # max by probability; ties resolved by order of dict
    return max(weights, key=weights.get)


def _compute_plan_payoff_for_true_type(
    true_type: FirmType,
    D: float,
    V1: float,
    V2: float,
    plan: Plan,
) -> float:
    """
    Compute the firm's payoff under a single plan, given its true type.

    Rules:
      - Firm must raise full D. If caps of A+B cannot reach D => plan invalid -> payoff = -inf.
      - If payoff < 0, firm will later choose Club C instead (0 payoff).
    """
    cap_A = plan.club_A.max_cap
    cap_B = plan.club_B.max_cap

    # Borrow as much as possible, prioritizing Club A first, then B
    borrow_A = min(cap_A, D)
    remaining = max(D - borrow_A, 0.0)
    borrow_B = min(cap_B, remaining)
    total_borrowed = borrow_A + borrow_B

    if total_borrowed + 1e-9 < D:
        # Cannot raise full funding => invalid plan
        return float("-inf")

    # Project value from the true type (no maturity tilt here)
    value = _type_base_project_value(true_type, D, V1, V2)

    # Cost of borrowing
    cost_A = plan.club_A.zcb_price * borrow_A
    cost_B = plan.club_B.zcb_price * borrow_B
    cost = cost_A + cost_B

    payoff = value - cost
    return payoff


def _simulate_firm_choice(true_type: FirmType,
                          principal: PrincipalInfo,
                          plans: Dict[str, Plan]) -> Dict:
    """
    For a single firm:
      - compute payoff under each plan (safe/mixed/risky),
      - compare them with Club C (0),
      - choose the best option.
    """
    D = principal.funding_need
    V1, V2 = principal.project_returns

    plan_payoffs: Dict[str, float] = {}
    for plan_name, plan in plans.items():
        p = _compute_plan_payoff_for_true_type(true_type, D, V1, V2, plan)
        plan_payoffs[plan_name] = p

    # Club C payoff = 0 (always available)
    opt_out_payoff = 0.0

    # Determine best plan among safe/mixed/risky
    # (we don't clamp negatives yet; choice vs Club C decides that)
    best_plan_name = None
    best_plan_payoff = float("-inf")
    for name, p in plan_payoffs.items():
        if p > best_plan_payoff:
            best_plan_payoff = p
            best_plan_name = name

    # Now compare with Club C
    if best_plan_payoff is None or best_plan_payoff <= 0.0:
        chosen_plan = "opt_out"
        chosen_payoff = opt_out_payoff
    else:
        chosen_plan = best_plan_name
        chosen_payoff = best_plan_payoff

    intended = _principal_intended_plan_name(principal)

    return {
        "true_type": true_type,
        "principal_intended_plan": intended,
        "chosen_plan": chosen_plan,
        "deviation": (chosen_plan not in [intended, "opt_out"]),
        "non_participation": (chosen_plan == "opt_out"),
        "payoffs": {
            "safe": plan_payoffs["safe"],
            "mixed": plan_payoffs["mixed"],
            "risky": plan_payoffs["risky"],
            "opt_out": opt_out_payoff,
        },
        "chosen_payoff": chosen_payoff,
    }


# ============================================================
#  PUBLIC ENTRYPOINT: RUN SIMULATION
# ============================================================

def run_catalog_simulation(scenario: Dict) -> Dict:
    """
    Steps:
      3. For each firm: from beliefs + project data, compute type probabilities
         and welfare orientation (short vs long).
      4. Build a 3-plan catalog (safe / mixed / risky) per firm.
      5. Simulate each firm (using its TRUE type) choosing the best plan,
         or Club C (non-participation) when all actual payoffs are <= 0.

    Input 'scenario':
      {
        "firms": [
          {
            "id": 1,
            "principal": {
               "funding_need": 120.0,
               "project_returns": [0.9, 1.2],
               "belief_safe_score": 60.0,
            },
            "actual": {
               "type": "risky"
            }
          },
          ...
        ]
      }
    """

    firms_input = scenario.get("firms", [])
    results: List[FirmCatalogResult] = []
    welfare_hints = []
    realized_payoffs = []

    for f in firms_input:
        firm_id = f["id"]
        p_raw = f["principal"]
        a_raw = f["actual"]

        D = float(p_raw["funding_need"])
        V1, V2 = p_raw["project_returns"]
        belief_safe_score = float(p_raw["belief_safe_score"])
        true_type: FirmType = a_raw["type"]

        # Belief decomposition
        beliefs = _compute_beliefs_from_safe_score(belief_safe_score)

        principal = PrincipalInfo(
            funding_need=D,
            project_returns=[V1, V2],
            belief_safe_score=belief_safe_score,
            pi_safe=beliefs["pi_safe"],
            pi_mixed=beliefs["pi_mixed"],
            pi_risky=beliefs["pi_risky"],
        )

        # Step 3: orientation / expected welfare
        scores = _orientation_scores(principal)
        expected_welfare_hint = max(scores["expected_short"], scores["expected_long"])
        welfare_hints.append(expected_welfare_hint)

        # Step 4: build plans
        plans = _build_plans_from_orientation(principal, scores)

        # Step 5: firm chooses plan (or opt-out)
        choice_info = _simulate_firm_choice(true_type, principal, plans)
        realized_payoffs.append(choice_info["chosen_payoff"])

        firm_result = FirmCatalogResult(
            firm_id=firm_id,
            principal=principal,
            plans=plans,
            opt_out=True,
            expected_welfare_hint=expected_welfare_hint,
            choice=choice_info,
        )
        results.append(firm_result)

    # Aggregate welfare hints and realized payoffs
    avg_welfare_hint = sum(welfare_hints) / len(welfare_hints) if welfare_hints else 0.0
    avg_realized_payoff = sum(realized_payoffs) / len(realized_payoffs) if realized_payoffs else 0.0

    # Convert dataclasses to dicts
    result_dicts = []
    for r in results:
        result_dicts.append({
            "firm_id": r.firm_id,
            "principal": {
                "funding_need": r.principal.funding_need,
                "project_returns": r.principal.project_returns,
                "belief_safe_score": r.principal.belief_safe_score,
                "pi_safe": r.principal.pi_safe,
                "pi_mixed": r.principal.pi_mixed,
                "pi_risky": r.principal.pi_risky,
            },
            "plans": {
                name: {
                    "club_A": asdict(plan.club_A),
                    "club_B": asdict(plan.club_B),
                }
                for name, plan in r.plans.items()
            },
            "opt_out": r.opt_out,
            "expected_welfare_hint": r.expected_welfare_hint,
            "choice": r.choice,
        })

    return {
        "welfare": avg_welfare_hint,              # principal's ex-ante welfare hint
        "realized_payoff": avg_realized_payoff,   # average ex-post firm payoff
        "iterations": 1,
        "results": result_dicts,
    }
