from dataclasses import dataclass, asdict, field
from typing import List, Dict, Literal
import math


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
    true_type: FirmType
    plans: Dict[str, Plan]
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


def _type_base_project_value(
    type_: FirmType,
    D: float,
    V1: float,
    V2: float,
) -> float:
    """
    Maturity-agnostic project value for the firm's payoff.

    Uses ONLY the true type's intrinsic alpha and no belief adjustments,
    because the firm knows its own type when evaluating its project.
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


def _type_plan_multiplier(true_type: FirmType, plan_name: str) -> float:
    return TYPE_PLAN_MULTIPLIER[true_type][plan_name]


# ============================================================
#  INITIAL CATALOG CONSTRUCTION
# ============================================================

def _build_plans_from_orientation(principal: PrincipalInfo,
                                  scores: Dict) -> Dict[str, Plan]:
    """
    Builds safe/mixed/risky plans with:
      - Base IC-inspired structure (different prices & caps).
      - Belief-driven favoring/unfavoring.
      - Short vs long tilt based on expected_short vs expected_long.
    """

    D = principal.funding_need
    E_short = scores["expected_short"]
    E_long = scores["expected_long"]

    # How much the environment "likes" short vs long, from principal's POV
    delta = E_short - E_long
    orientation_strength = math.tanh(delta / 10.0)

    tilt = 0.01 * orientation_strength
    orient_short = +tilt
    orient_long = -tilt

    # Base prices and caps for each type/maturity (IC skeleton)
    BASE_PRICES = {
        "safe": {"short": 0.95, "long": 0.88},
        "mixed": {"short": 0.93, "long": 0.90},
        "risky": {"short": 0.90, "long": 0.94},
    }

    BASE_CAP_MULT = {
        "safe": {"short": 0.70, "long": 0.50},
        "mixed": {"short": 0.70, "long": 0.70},
        "risky": {"short": 0.50, "long": 0.70},
    }

    # Favor levels by belief ordering
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

    # Belief-based tweaks
    FAVOR_PRICE = {
        "favored": +0.01,  
        "neutral": 0.0,
        "unfavored": -0.01, 
    }
    FAVOR_CAP = {
        "favored": 1.10,
        "neutral": 1.00,
        "unfavored": 0.90,
    }

    def side(plan_name: str, maturity: Maturity) -> PlanSide:
        base_price = BASE_PRICES[plan_name][maturity]
        base_cap = BASE_CAP_MULT[plan_name][maturity] * D

        price = base_price + (orient_short if maturity == "short" else orient_long)
        price += FAVOR_PRICE[favor_levels[plan_name]]

        price = max(0.5, min(0.995, price))
        cap = base_cap * FAVOR_CAP[favor_levels[plan_name]]

        return PlanSide(
            zcb_price=round(price, 3),
            max_cap=round(cap)
        )

    return {
        "safe": Plan("safe", side("safe", "short"), side("safe", "long")),
        "mixed": Plan("mixed", side("mixed", "short"), side("mixed", "long")),
        "risky": Plan("risky", side("risky", "short"), side("risky", "long")),
    }


# ============================================================
#  PAYOFF ENGINE — GRID SEARCH, EXACT D FUNDING
# ============================================================

def _compute_plan_payoff_for_true_type(
    true_type: FirmType,
    D: float,
    V1: float,
    V2: float,
    plan: Plan,
) -> Dict:
    """
    - Exact D funding required.
    - Search all feasible (A,B) combos that respect caps.
    - Choose minimum debt cost.
    - Apply type plan fit multiplier to project value.
    """

    cap_A = int(plan.club_A.max_cap)
    cap_B = int(plan.club_B.max_cap)

    best_cost = float("inf")
    best_a = None
    best_b = None

    for a in range(0, cap_A + 1):
        b = D - a
        if b < 0:
            break
        if 0 <= b <= cap_B:
            cost = plan.club_A.zcb_price * a + plan.club_B.zcb_price * b
            if cost < best_cost:
                best_cost = cost
                best_a = a
                best_b = b

    if best_a is None:
        # Cannot raise full D under this plan
        return {
            "payoff": float("Insufficient-funding"),
            "valid": False,
            "borrow_A": 0,
            "borrow_B": 0,
            "cap_A": cap_A,
            "cap_B": cap_B,
            "raw_value": 0,
            "fit_multiplier": 0,
            "final_value": 0,
            "cost": None,
        }

    raw_value = _type_base_project_value(true_type, D, V1, V2)
    fit = _type_plan_multiplier(true_type, plan.name)
    final_value = raw_value * fit
    payoff = final_value - best_cost

    return {
        "payoff": payoff,
        "valid": True,
        "borrow_A": best_a,
        "borrow_B": best_b,
        "cap_A": cap_A,
        "cap_B": cap_B,
        "raw_value": raw_value,
        "fit_multiplier": fit,
        "final_value": final_value,
        "cost": best_cost,
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
                          plans: Dict[str, Plan]) -> Dict:
    """
    One-shot choice among safe/mixed/risky vs opt-out
    given the CURRENT plans.
    """

    D = principal.funding_need
    V1, V2 = principal.project_returns

    plan_payoffs: Dict[str, float] = {}
    plan_details: Dict[str, Dict] = {}

    for name, plan in plans.items():
        details = _compute_plan_payoff_for_true_type(true_type, D, V1, V2, plan)
        plan_payoffs[name] = details["payoff"]
        plan_details[name] = details

    best_plan = max(plan_payoffs, key=lambda k: plan_payoffs[k])
    best_payoff = plan_payoffs[best_plan]

    if best_payoff <= 0:
        chosen_plan = "opt_out"
        chosen_payoff = 0.0
    else:
        chosen_plan = best_plan
        chosen_payoff = best_payoff

    intended = _principal_intended_plan_name(principal)

    return {
        "true_type": true_type,
        "principal_intended_plan": intended,
        "chosen_plan": chosen_plan,
        "deviation": (chosen_plan not in [intended, "opt_out"]),
        "non_participation": (chosen_plan == "opt_out"),
        "payoffs": plan_payoffs,
        "plan_details": plan_details,
        "chosen_payoff": chosen_payoff,
    }


# ============================================================
#  CATALOG UPDATE RULES
# ============================================================

def _adjust_price(side: PlanSide, delta: float) -> None:
    side.zcb_price = max(0.5, min(0.995, side.zcb_price + delta))


def _adjust_cap(side: PlanSide, mult: float, D: float) -> None:
    new_cap = side.max_cap * mult
    side.max_cap = max(0.0, min(2.0 * D, new_cap))


def _update_plans_for_firm(state: FirmSimState,
                           choice_info: Dict) -> None:
    """
    Very simple update rule for now:

    - If firm opts out:
        -> Make ALL plans more attractive:
           lower prices slightly, increase caps slightly.

    - If firm chooses a plan P:
        -> Make P slightly WORSE (raise price a bit),
        -> Make all other plans slightly BETTER (lower price a bit),
        -> Keep caps unchanged for now.

    This fits your idea:
      "Offer better alternatives; if firm still sticks, we infer type".
    """

    D = state.principal.funding_need
    chosen = choice_info["chosen_plan"]

    if chosen == "opt_out":
        for plan in state.plans.values():
            _adjust_price(plan.club_A, delta=-0.01)
            _adjust_price(plan.club_B, delta=-0.01)
            _adjust_cap(plan.club_A, mult=1.10, D=D)
            _adjust_cap(plan.club_B, mult=1.10, D=D)
        return

    # Non-opt-out: chosen plan vs rivals
    for name, plan in state.plans.items():
        if name == chosen:
            # chosen plan: make slightly worse
            _adjust_price(plan.club_A, delta=+0.005)
            _adjust_price(plan.club_B, delta=+0.005)
        else:
            # rivals: make slightly better
            _adjust_price(plan.club_A, delta=-0.005)
            _adjust_price(plan.club_B, delta=-0.005)
    # caps unchanged here (you can experiment later)


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

    # 1) Initial principal info & catalogs
    firm_states: List[FirmSimState] = []

    for f in firms_input:
        firm_id = f["id"]
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

        scores = _orientation_scores(principal)
        plans = _build_plans_from_orientation(principal, scores)

        firm_states.append(
            FirmSimState(
                firm_id=firm_id,
                principal=principal,
                true_type=true_type,
                plans=plans,
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
            snapshot_plans = {
                name: {
                    "club_A": {
                        "zcb_price": p.club_A.zcb_price,
                        "max_cap": p.club_A.max_cap,
                    },
                    "club_B": {
                        "zcb_price": p.club_B.zcb_price,
                        "max_cap": p.club_B.max_cap,
                    },
                }
                for name, p in state.plans.items()
            }

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
            "plans": {
                name: {
                    "club_A": asdict(plan.club_A),
                    "club_B": asdict(plan.club_B),
                }
                for name, plan in state.plans.items()
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
