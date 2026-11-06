"""waterloo-jane-street-qr — quant research toolkit for Hermes Agent.

Tools
-----
black_scholes       European call/put prices + Greeks (Delta, Gamma, Vega, Theta, Rho),
                    put-call parity check, optional implied-vol solve.
ev_calculator       Exact (rational) solvers for classic trading-interview puzzles:
                    custom lotteries, Bayes, dice re-roll / max, red-black cards,
                    coin pattern waiting times, Penney races, gambler's ruin,
                    coupon collector, Kelly sizing.
order_book_metrics  Mid, spread (abs / bps), microprice, imbalance, depth VWAPs and
                    walk-the-book impact for a given quantity.

Context
-------
A system-prompt section injects the "Waterloo QR" persona: strict probabilistic
reasoning, rigorous math, desk-style clarity, and an explicit EV verdict.

Every handler takes the LLM's args dict and returns a JSON string. Handlers never raise.
"""

from __future__ import annotations

import json
import logging
import math
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

TOOLSET = "waterloo_qr"

# ════════════════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════════════════


class ToolInputError(ValueError):
    """Bad user/LLM input — reported back as a clean JSON error."""


def _clean(obj: Any) -> Any:
    """Make a structure strict-JSON safe (no NaN/inf, Fractions rendered)."""
    if isinstance(obj, Fraction):
        return _frac_out(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _ok(payload: Dict[str, Any]) -> str:
    return json.dumps(_clean({"ok": True, **payload}), allow_nan=False)


def _err(message: str, **extra: Any) -> str:
    return json.dumps(_clean({"ok": False, "error": message, **extra}), allow_nan=False)


def _safe(fn: Callable[[dict], Dict[str, Any]]) -> Callable[..., str]:
    """Wrap a pure dict->dict computation as a never-raising Hermes tool handler."""

    def handler(args: Optional[dict] = None, **_kwargs: Any) -> str:
        try:
            if args is None:
                args = {}
            if not isinstance(args, dict):
                raise ToolInputError("arguments must be a JSON object")
            return _ok(fn(args))
        except ToolInputError as e:
            return _err(str(e))
        except (ZeroDivisionError, OverflowError) as e:
            return _err(f"numerical error: {e}")
        except Exception as e:  # pragma: no cover — last-resort guard
            logger.exception("waterloo-jane-street-qr handler %s failed", fn.__name__)
            return _err(f"internal error: {type(e).__name__}: {e}")

    handler.__name__ = fn.__name__
    handler.__doc__ = fn.__doc__
    return handler


def _num(args: dict, key: str, default: Any = None, *, required: bool = False,
         lo: Optional[float] = None, hi: Optional[float] = None,
         lo_inclusive: bool = True) -> float:
    val = args.get(key, default)
    if val is None:
        if required:
            raise ToolInputError(f"'{key}' is required")
        return default
    try:
        x = float(Fraction(val.strip())) if isinstance(val, str) else float(val)
    except (TypeError, ValueError, ZeroDivisionError):
        raise ToolInputError(f"'{key}' must be a number, got {val!r}")
    if math.isnan(x) or math.isinf(x):
        raise ToolInputError(f"'{key}' must be finite")
    if lo is not None and (x < lo or (x == lo and not lo_inclusive)):
        raise ToolInputError(f"'{key}' must be {'>=' if lo_inclusive else '>'} {lo}, got {x}")
    if hi is not None and x > hi:
        raise ToolInputError(f"'{key}' must be <= {hi}, got {x}")
    return x


def _int(args: dict, key: str, default: Optional[int] = None, *, lo: int = 0,
         hi: int = 10**6, required: bool = False) -> int:
    val = args.get(key, default)
    if val is None:
        if required:
            raise ToolInputError(f"'{key}' is required")
        return default  # type: ignore[return-value]
    try:
        f = float(val)
    except (TypeError, ValueError):
        raise ToolInputError(f"'{key}' must be an integer, got {val!r}")
    if not f.is_integer():
        raise ToolInputError(f"'{key}' must be an integer, got {val!r}")
    n = int(f)
    if not lo <= n <= hi:
        raise ToolInputError(f"'{key}' must be in [{lo}, {hi}], got {n}")
    return n


def _frac(val: Any, name: str = "value") -> Fraction:
    """Exact rational from int / float / '1/6' / '0.35'."""
    try:
        if isinstance(val, bool):
            raise TypeError
        if isinstance(val, Fraction):
            return val
        if isinstance(val, int):
            return Fraction(val)
        if isinstance(val, float):
            if math.isnan(val) or math.isinf(val):
                raise ValueError
            return Fraction(repr(val))  # shortest decimal repr -> exact intent (0.1 -> 1/10)
        if isinstance(val, str):
            return Fraction(val.strip())
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    raise ToolInputError(f"{name} must be a number or fraction string like '1/6', got {val!r}")


def _prob(val: Any, name: str) -> Fraction:
    p = _frac(val, name)
    if not 0 <= p <= 1:
        raise ToolInputError(f"{name} must be a probability in [0, 1], got {p}")
    return p


def _frac_out(x: Fraction) -> Dict[str, Any]:
    out: Dict[str, Any] = {"decimal": float(x)}
    # Exact form only when it is human-readable.
    if len(str(x.numerator)) + len(str(x.denominator)) <= 40:
        out["exact"] = str(x)
    return out


def _solve_linear(a: List[List[Fraction]], b: List[Fraction]) -> List[Fraction]:
    """Exact Gauss-Jordan elimination over the rationals."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = next((r for r in range(col, n) if m[r][col] != 0), None)
        if piv is None:
            raise ToolInputError("singular system — the absorbing state is unreachable "
                                 "(e.g. a probability of 0 or 1 makes the pattern impossible)")
        m[col], m[piv] = m[piv], m[col]
        inv = 1 / m[col][col]
        m[col] = [v * inv for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [rv - f * cv for rv, cv in zip(m[r], m[col])]
    return [m[i][n] for i in range(n)]


# ════════════════════════════════════════════════════════════════════════
# 1. Black-Scholes
# ════════════════════════════════════════════════════════════════════════

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / _SQRT2)


def _npdf(x: float) -> float:
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _bs_core(S: float, K: float, T: float, r: float, sigma: float, q: float) -> Dict[str, Any]:
    disc_r, disc_q = math.exp(-r * T), math.exp(-q * T)
    forward = S * math.exp((r - q) * T)

    if T == 0 or sigma == 0:
        # Degenerate (deterministic) limit: payoff on the forward, discounted.
        call_px = disc_r * max(forward - K, 0.0)
        put_px = disc_r * max(K - forward, 0.0)
        itm_c = 1.0 if forward > K else (0.5 if forward == K else 0.0)
        itm_p = 1.0 - itm_c if forward != K else 0.5
        if T == 0:
            theta_c = theta_p = None  # undefined at expiry
            rho_c = rho_p = 0.0
        else:
            theta_c = (q * S * disc_q - r * K * disc_r) * itm_c
            theta_p = (r * K * disc_r - q * S * disc_q) * itm_p
            rho_c = K * T * disc_r * itm_c
            rho_p = -K * T * disc_r * itm_p
        return {
            "d1": None, "d2": None, "forward": forward, "degenerate": True,
            "call": {"price": call_px, "delta": disc_q * itm_c, "gamma": 0.0, "vega": 0.0,
                     "theta": theta_c, "rho": rho_c, "prob_itm_risk_neutral": itm_c},
            "put": {"price": put_px, "delta": -disc_q * itm_p, "gamma": 0.0, "vega": 0.0,
                    "theta": theta_p, "rho": rho_p, "prob_itm_risk_neutral": itm_p},
        }

    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    nd1, nd2, nmd1, nmd2 = _ncdf(d1), _ncdf(d2), _ncdf(-d1), _ncdf(-d2)
    pdf_d1 = _npdf(d1)

    gamma = disc_q * pdf_d1 / (S * vol_sqrt_t)
    vega = S * disc_q * pdf_d1 * math.sqrt(T)  # per 1.00 (100 vol points)
    decay = -S * disc_q * pdf_d1 * sigma / (2.0 * math.sqrt(T))

    return {
        "d1": d1, "d2": d2, "forward": forward, "degenerate": False,
        "call": {
            "price": S * disc_q * nd1 - K * disc_r * nd2,
            "delta": disc_q * nd1,
            "gamma": gamma,
            "vega": vega,
            "theta": decay - r * K * disc_r * nd2 + q * S * disc_q * nd1,
            "rho": K * T * disc_r * nd2,
            "prob_itm_risk_neutral": nd2,
        },
        "put": {
            "price": K * disc_r * nmd2 - S * disc_q * nmd1,
            "delta": -disc_q * nmd1,
            "gamma": gamma,
            "vega": vega,
            "theta": decay + r * K * disc_r * nmd2 - q * S * disc_q * nmd1,
            "rho": -K * T * disc_r * nmd2,
            "prob_itm_risk_neutral": nmd2,
        },
    }


def _implied_vol(target: float, kind: str, S: float, K: float, T: float, r: float,
                 q: float) -> Dict[str, Any]:
    if T <= 0:
        return {"error": "implied vol undefined at T = 0"}
    lo_px = _bs_core(S, K, T, r, 0.0, q)[kind]["price"]  # no-arb lower bound
    hi_bound = S * math.exp(-q * T) if kind == "call" else K * math.exp(-r * T)
    if not lo_px - 1e-12 <= target < hi_bound:
        return {"error": f"market price {target} violates no-arbitrage bounds "
                         f"[{lo_px:.6g}, {hi_bound:.6g}) for a {kind}"}
    lo, hi = 1e-9, 5.0
    while _bs_core(S, K, T, r, hi, q)[kind]["price"] < target and hi < 100:
        hi *= 2
    for _ in range(200):  # bisection: monotone in sigma, guaranteed convergence
        mid = 0.5 * (lo + hi)
        if _bs_core(S, K, T, r, mid, q)[kind]["price"] < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return {"implied_volatility": 0.5 * (lo + hi)}


def _black_scholes(args: dict) -> Dict[str, Any]:
    """European option pricing + Greeks under Black-Scholes-Merton."""
    S = _num(args, "spot", required=True, lo=0, lo_inclusive=False)
    K = _num(args, "strike", required=True, lo=0, lo_inclusive=False)
    T = _num(args, "time_to_maturity", required=True, lo=0)
    r = _num(args, "risk_free_rate", required=True, lo=-1, hi=5)
    sigma = _num(args, "volatility", required=True, lo=0, hi=20)
    q = _num(args, "dividend_yield", 0.0, lo=-1, hi=5)
    kind = str(args.get("option_type", "both")).strip().lower()
    if kind not in ("call", "put", "both"):
        raise ToolInputError("option_type must be 'call', 'put' or 'both'")

    warnings: List[str] = []
    if sigma > 3:
        warnings.append(f"volatility={sigma} looks like a percentage; inputs are decimals (0.20 = 20%).")
    if abs(r) > 0.5:
        warnings.append(f"risk_free_rate={r} looks like a percentage; inputs are decimals (0.05 = 5%).")
    if T > 50:
        warnings.append(f"time_to_maturity={T} years — did you pass days? Use days/365.")

    core = _bs_core(S, K, T, r, sigma, q)

    def present(leg: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(leg)
        out["vega_per_vol_point"] = leg["vega"] / 100.0  # per 1% absolute vol move
        out["theta_per_day"] = None if leg["theta"] is None else leg["theta"] / 365.0
        out["rho_per_bp"] = leg["rho"] / 10_000.0
        return out

    result: Dict[str, Any] = {
        "model": "Black-Scholes-Merton (European, continuous dividend yield)",
        "inputs": {"spot": S, "strike": K, "time_to_maturity_years": T,
                   "risk_free_rate": r, "volatility": sigma, "dividend_yield": q},
        "d1": core["d1"], "d2": core["d2"], "forward": core["forward"],
        "moneyness_ln_F_over_K": math.log(core["forward"] / K),
        "units": {"vega": "price change per 1.00 vol (vega_per_vol_point: per 1 vol pt)",
                  "theta": "price change per year (theta_per_day: per calendar day /365)",
                  "rho": "price change per 1.00 rate (rho_per_bp: per 1bp)"},
    }
    if core["degenerate"]:
        warnings.append("T = 0 or sigma = 0: deterministic limit; Gamma/Vega are 0 (Dirac at the strike).")
    for leg in ("call", "put"):
        if kind in (leg, "both"):
            result[leg] = present(core[leg])

    parity = (core["call"]["price"] - core["put"]["price"]) - (
        S * math.exp(-q * T) - K * math.exp(-r * T))
    result["put_call_parity_residual"] = parity  # ≈ 0 by construction; sanity check

    mkt = args.get("market_price")
    if mkt is not None:
        target = _num(args, "market_price", lo=0)
        iv_kind = kind if kind in ("call", "put") else "call"
        result["implied_vol_solve"] = {"option_type": iv_kind, "market_price": target,
                                       **_implied_vol(target, iv_kind, S, K, T, r, q)}
    if warnings:
        result["warnings"] = warnings
    return result


BLACK_SCHOLES_SCHEMA = {
    "name": "black_scholes",
    "description": (
        "Price European call/put options with Black-Scholes-Merton and return the Greeks "
        "(Delta, Gamma, Vega, Theta, Rho), d1/d2, forward, risk-neutral P(ITM) and a put-call "
        "parity check. All rates/vols are DECIMALS (20% vol = 0.20), time in YEARS. "
        "Optionally pass market_price to back out implied volatility."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "spot": {"type": "number", "description": "Underlying spot price S (> 0)."},
            "strike": {"type": "number", "description": "Strike price K (> 0)."},
            "time_to_maturity": {"type": "number", "description": "Time to expiry T in years (e.g. 30 days = 30/365)."},
            "risk_free_rate": {"type": "number", "description": "Continuously compounded risk-free rate r as a decimal (0.05 = 5%)."},
            "volatility": {"type": "number", "description": "Annualized volatility sigma as a decimal (0.2 = 20%)."},
            "dividend_yield": {"type": "number", "description": "Continuous dividend/borrow yield q as a decimal. Default 0."},
            "option_type": {"type": "string", "enum": ["call", "put", "both"], "description": "Which leg(s) to return. Default 'both'."},
            "market_price": {"type": "number", "description": "Optional observed option price; solves implied vol for option_type (call if 'both')."},
        },
        "required": ["spot", "strike", "time_to_maturity", "risk_free_rate", "volatility"],
    },
}


# ════════════════════════════════════════════════════════════════════════
# 2. EV / probability puzzle solver
# ════════════════════════════════════════════════════════════════════════

def _ev_custom(args: dict) -> Dict[str, Any]:
    outcomes = args.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise ToolInputError("custom: 'outcomes' must be a non-empty list of payoffs")
    if len(outcomes) > 10_000:
        raise ToolInputError("custom: at most 10,000 outcomes")
    xs = [_frac(v, "outcome") for v in outcomes]
    probs = args.get("probabilities")
    if probs is None:
        ps = [Fraction(1, len(xs))] * len(xs)
    else:
        if not isinstance(probs, list) or len(probs) != len(xs):
            raise ToolInputError("custom: 'probabilities' must be a list the same length as 'outcomes'")
        ps = [_prob(p, "probability") for p in probs]
    total = sum(ps)
    if total != 1:
        if abs(float(total) - 1) > 1e-9:
            raise ToolInputError(f"custom: probabilities sum to {float(total):.12g}, not 1")
        ps = [p / total for p in ps]  # renormalize float round-off
    cost = _frac(args.get("cost", 0), "cost")
    net = [x - cost for x in xs]
    ev = sum(p * x for p, x in zip(ps, net))
    var = sum(p * (x - ev) ** 2 for p, x in zip(ps, net))
    p_win = sum(p for p, x in zip(ps, net) if x > 0)
    p_loss = sum(p for p, x in zip(ps, net) if x < 0)
    return {
        "expected_value": ev, "gross_expected_payoff": ev + cost, "cost": cost,
        "variance": var, "std_dev": math.sqrt(float(var)),
        "sharpe_per_play": (float(ev) / math.sqrt(float(var))) if var > 0 else None,
        "p_win": p_win, "p_loss": p_loss, "p_push": 1 - p_win - p_loss,
        "verdict": "EV > 0 — take it (size it)" if ev > 0 else ("EV = 0 — pass, no edge" if ev == 0 else "EV < 0 — pass"),
    }


def _ev_bayes(args: dict) -> Dict[str, Any]:
    priors, likes = args.get("priors"), args.get("likelihoods")
    if not isinstance(priors, list) or not isinstance(likes, list) or len(priors) != len(likes) or not priors:
        raise ToolInputError("bayes: 'priors' and 'likelihoods' must be equal-length non-empty lists "
                             "(likelihoods[i] = P(evidence | H_i))")
    pri = [_prob(p, "prior") for p in priors]
    if abs(float(sum(pri)) - 1) > 1e-9:
        raise ToolInputError(f"bayes: priors sum to {float(sum(pri)):.12g}, not 1")
    pri = [p / sum(pri) for p in pri]
    lik = [_prob(l, "likelihood") for l in likes]
    k = _int(args, "num_observations", 1, lo=0, hi=10_000)
    lik_k = [l ** k for l in lik]
    joint = [p * l for p, l in zip(pri, lik_k)]
    evidence = sum(joint)
    if evidence == 0:
        raise ToolInputError("bayes: evidence has probability 0 under every hypothesis")
    post = [j / evidence for j in joint]
    labels = args.get("hypotheses")
    if not isinstance(labels, list) or len(labels) != len(pri):
        labels = [f"H{i + 1}" for i in range(len(pri))]
    # Predictive probability the NEXT observation repeats the evidence.
    next_pred = sum(p * l for p, l in zip(post, lik))
    return {
        "num_observations": k,
        "p_evidence": evidence,
        "posteriors": [{"hypothesis": str(h), "prior": p, "likelihood_per_obs": l, "posterior": po}
                       for h, p, l, po in zip(labels, pri, lik, post)],
        "p_next_observation_same": next_pred,
    }


def _ev_dice_reroll(args: dict) -> Dict[str, Any]:
    n = _int(args, "faces", 6, lo=2, hi=10_000)
    k = _int(args, "rolls", 2, lo=1, hi=1_000)
    c = _frac(args.get("cost_per_roll", 0), "cost_per_roll")
    # V[j] = value of the game with j rolls remaining (paying c for each roll taken).
    faces = [Fraction(i) for i in range(1, n + 1)]
    v_prev: Optional[Fraction] = None
    values, policy = [], []
    for j in range(1, k + 1):
        if v_prev is None:
            v = sum(faces) / n - c
            policy.append({"rolls_left_after_this": 0, "keep_if_at_least": 1})
        else:
            v = sum(max(f, v_prev) for f in faces) / n - c
            thr = next((int(f) for f in faces if f >= v_prev), None)
            policy.append({"rolls_left_after_this": j - 1, "keep_if_at_least": thr,
                           "continuation_value": v_prev})
        values.append({"max_rolls": j, "value": v})
        v_prev = v
    play = v_prev if v_prev is not None and v_prev > 0 else Fraction(0)
    return {
        "faces": n, "max_rolls": k, "cost_per_roll": c,
        "expected_value": v_prev, "value_if_optional_to_play": play,
        "values_by_rolls": values,
        "optimal_policy": "Stop as soon as the face >= the value of continuing (backward induction).",
        "thresholds": list(reversed(policy)),
    }


def _ev_dice_max(args: dict) -> Dict[str, Any]:
    n = _int(args, "faces", 6, lo=1, hi=100_000)
    k = _int(args, "dice", 2, lo=1, hi=10_000)
    e_max = sum(1 - Fraction(m - 1, n) ** k for m in range(1, n + 1))
    e_min = sum(Fraction(n - m + 1, n) ** k for m in range(1, n + 1))
    return {"faces": n, "dice": k, "expected_max": e_max, "expected_min": e_min,
            "expected_sum": Fraction(k * (n + 1), 2),
            "identity": "E[max] = sum_{m=1..n} (1 - ((m-1)/n)^k);  E[min] = sum_{m=1..n} ((n-m+1)/n)^k"}


def _ev_card_red_black(args: dict) -> Dict[str, Any]:
    R = _int(args, "red", 26, lo=0, hi=200)
    B = _int(args, "black", 26, lo=0, hi=200)
    # V(r,b) = max(0, r/(r+b)*(1+V(r-1,b)) + b/(r+b)*(-1+V(r,b-1)))  — stop any time.
    use_exact = R * B <= 2_704  # exact rationals up to 52x52; floats beyond
    one = Fraction(1) if use_exact else 1.0
    zero = Fraction(0) if use_exact else 0.0
    V = [[zero] * (B + 1) for _ in range(R + 1)]
    for r in range(R + 1):
        for b in range(B + 1):
            if r == 0:
                V[r][b] = zero
            elif b == 0:
                V[r][b] = r * one
            else:
                tot = r + b
                draw = (one * r / tot) * (1 + V[r - 1][b]) + (one * b / tot) * (-1 + V[r][b - 1])
                V[r][b] = draw if draw > 0 else zero
    val = V[R][B]
    first = "draw" if R and B and V[R][B] > 0 else ("draw" if B == 0 and R > 0 else "stop")
    out: Dict[str, Any] = {"red": R, "black": B,
                           "rules": "Draw without replacement; +1 for red, -1 for black; stop any time.",
                           "expected_value": val if use_exact else float(val),
                           "first_action": first}
    if not use_exact:
        out["note"] = "large deck: computed in floating point"
    return out


def _check_pattern(p: Any, name: str) -> str:
    if not isinstance(p, str):
        raise ToolInputError(f"{name} must be a string of H/T")
    s = p.strip().upper()
    if not s or any(ch not in "HT" for ch in s) or len(s) > 24:
        raise ToolInputError(f"{name} must be 1-24 characters of H/T, got {p!r}")
    return s


def _pattern_chain(patterns: Sequence[str], p: Fraction):
    """States = distinct proper prefixes of the patterns; absorbing on completion."""
    prefixes = sorted({pat[:i] for pat in patterns for i in range(len(pat))}, key=lambda s: (len(s), s))
    idx = {s: i for i, s in enumerate(prefixes)}

    def step(state: str, ch: str) -> Tuple[str, Optional[List[int]]]:
        t = state + ch
        hits = [i for i, pat in enumerate(patterns) if t.endswith(pat)]
        if hits:
            return t, hits
        for start in range(len(t) + 1):
            if t[start:] in idx:
                return t[start:], None
        return "", None

    q = 1 - p
    trans = []
    for s in prefixes:
        row = []
        for ch, pr in (("H", p), ("T", q)):
            if pr == 0:
                continue
            nxt, hits = step(s, ch)
            row.append((pr, nxt, hits))
        trans.append(row)
    return prefixes, idx, trans


def _ev_coin_pattern(args: dict) -> Dict[str, Any]:
    pat = _check_pattern(args.get("pattern"), "pattern")
    p = _prob(args.get("p_heads", "1/2"), "p_heads")
    prefixes, idx, trans = _pattern_chain([pat], p)
    n = len(prefixes)
    # E_s = 1 + sum_{non-absorbing} P * E_next
    A = [[Fraction(0)] * n for _ in range(n)]
    b = [Fraction(1)] * n
    for i, row in enumerate(trans):
        A[i][i] += 1
        for pr, nxt, hits in row:
            if hits is None:
                A[i][idx[nxt]] -= pr
    E = _solve_linear(A, b)
    out = {"pattern": pat, "p_heads": p, "expected_flips": E[idx[""]]}
    if p == Fraction(1, 2):
        # Conway's leading-number identity: E = sum of 2^k over k where prefix_k == suffix_k.
        out["conway_check"] = sum(2 ** k for k in range(1, len(pat) + 1) if pat[:k] == pat[-k:])
    return out


def _ev_coin_race(args: dict) -> Dict[str, Any]:
    a = _check_pattern(args.get("pattern_a"), "pattern_a")
    bpat = _check_pattern(args.get("pattern_b"), "pattern_b")
    if a == bpat:
        raise ToolInputError("coin_race: patterns must differ")
    p = _prob(args.get("p_heads", "1/2"), "p_heads")
    prefixes, idx, trans = _pattern_chain([a, bpat], p)
    n = len(prefixes)

    def solve(reward: Callable[[List[int]], Fraction], step_cost: Fraction) -> List[Fraction]:
        A = [[Fraction(0)] * n for _ in range(n)]
        rhs = [step_cost] * n
        for i, row in enumerate(trans):
            A[i][i] += 1
            for pr, nxt, hits in row:
                if hits is None:
                    A[i][idx[nxt]] -= pr
                else:
                    rhs[i] += pr * reward(hits)
        return _solve_linear(A, rhs)

    pa = solve(lambda h: Fraction(1) if h == [0] else Fraction(0), Fraction(0))[idx[""]]
    pb = solve(lambda h: Fraction(1) if h == [1] else Fraction(0), Fraction(0))[idx[""]]
    tie = 1 - pa - pb
    e_len = solve(lambda h: Fraction(0), Fraction(1))[idx[""]]
    return {"pattern_a": a, "pattern_b": bpat, "p_heads": p,
            "p_a_first": pa, "p_b_first": pb, "p_tie": tie,
            "odds_a_to_b": (str(pa / pb) if pb else None),
            "expected_flips_until_either": e_len}


def _ev_gamblers_ruin(args: dict) -> Dict[str, Any]:
    i = _int(args, "start", required=True, lo=0, hi=1_000_000)
    N = _int(args, "target", required=True, lo=1, hi=1_000_000)
    if i > N:
        raise ToolInputError("gamblers_ruin: start must be <= target")
    p = float(_prob(args.get("p_win", "1/2"), "p_win"))
    q = 1 - p
    if i in (0, N):  # already absorbed
        win = 1.0 if i == N else 0.0
        return {"start": i, "target": N, "p_win_each_step": p,
                "p_reach_target": win, "p_ruin": 1 - win, "expected_steps": 0.0}
    if p in (0.0, 1.0):  # deterministic walk
        win = p
        return {"start": i, "target": N, "p_win_each_step": p, "p_reach_target": win,
                "p_ruin": 1 - win, "expected_steps": float(N - i) if p == 1 else float(i)}
    if abs(p - q) < 1e-15:
        win, dur = i / N, float(i * (N - i))
    else:
        rho = q / p
        # log-space for numerical stability with large N
        def frac_pow(k: int) -> float:
            # (1 - rho^k) / (1 - rho^N)
            if rho < 1:
                return (1 - rho ** k) / (1 - rho ** N)
            lr = math.log(rho)
            return math.exp((k - N) * lr) * (-math.expm1(-k * lr)) / (-math.expm1(-N * lr))
        win = frac_pow(i)
        dur = i / (q - p) - (N / (q - p)) * win
    return {"start": i, "target": N, "p_win_each_step": p,
            "p_reach_target": win, "p_ruin": 1 - win, "expected_steps": dur}


def _ev_coupon_collector(args: dict) -> Dict[str, Any]:
    n = _int(args, "coupons", required=True, lo=1, hi=100_000)
    m = _int(args, "collect", n, lo=1, hi=n)
    exact = n <= 2_000
    if exact:
        e = sum(Fraction(n, n - k) for k in range(m))
        var = sum(Fraction(k, n) / (Fraction(n - k, n) ** 2) for k in range(m))
    else:
        e = sum(n / (n - k) for k in range(m))
        var = sum((k / n) / ((n - k) / n) ** 2 for k in range(m))
    return {"coupons": n, "collect_distinct": m,
            "expected_draws": e if exact else float(e),
            "variance": var if exact else float(var),
            "std_dev": math.sqrt(float(var)),
            "identity": "E = sum_{k=0}^{m-1} n/(n-k)  (= n*H_n for a full set)"}


def _ev_kelly(args: dict) -> Dict[str, Any]:
    p = _prob(args.get("p_win"), "p_win") if args.get("p_win") is not None else None
    if p is None:
        raise ToolInputError("kelly: 'p_win' is required")
    b = _frac(args.get("odds", 1), "odds")
    a = _frac(args.get("loss_fraction", 1), "loss_fraction")
    if b <= 0 or a <= 0:
        raise ToolInputError("kelly: odds and loss_fraction must be > 0")
    q = 1 - p
    edge = p * b - q * a
    f_star = p / a - q / b
    f = max(Fraction(0), min(f_star, 1 / a))

    def growth(frac: Fraction) -> Optional[float]:
        fb, fa = float(frac), float(frac)
        if 1 - fa * float(a) <= 0:
            return None
        return float(p) * math.log1p(fb * float(b)) + float(q) * math.log1p(-fa * float(a))

    return {"p_win": p, "odds_b_to_1": b, "loss_fraction": a,
            "ev_per_unit_staked": edge,
            "kelly_fraction": f, "raw_kelly_unclipped": f_star,
            "half_kelly": f / 2,
            "log_growth_full_kelly": growth(f), "log_growth_half_kelly": growth(f / 2),
            "verdict": "EV > 0 — bet kelly_fraction (desks usually run half-Kelly)" if edge > 0
                       else "EV <= 0 — do not bet"}


_EV_SOLVERS: Dict[str, Callable[[dict], Dict[str, Any]]] = {
    "custom": _ev_custom,
    "bayes": _ev_bayes,
    "dice_reroll": _ev_dice_reroll,
    "dice_max": _ev_dice_max,
    "card_red_black": _ev_card_red_black,
    "coin_pattern": _ev_coin_pattern,
    "coin_race": _ev_coin_race,
    "gamblers_ruin": _ev_gamblers_ruin,
    "coupon_collector": _ev_coupon_collector,
    "kelly": _ev_kelly,
}


def _ev_calculator(args: dict) -> Dict[str, Any]:
    """Dispatch to an exact puzzle solver."""
    puzzle = str(args.get("puzzle", "")).strip().lower()
    solver = _EV_SOLVERS.get(puzzle)
    if solver is None:
        raise ToolInputError(f"unknown puzzle {puzzle!r}; choose one of: {', '.join(_EV_SOLVERS)}")
    return {"puzzle": puzzle, **solver(args)}


EV_CALCULATOR_SCHEMA = {
    "name": "ev_calculator",
    "description": (
        "Exact solver for quant-interview probability / expected-value puzzles. Results are exact "
        "rationals ('exact') plus decimals. Probabilities accept numbers or fraction strings ('1/6'). "
        "Pick 'puzzle' and pass only its fields:\n"
        "- custom: outcomes[], probabilities[] (default uniform), cost -> EV, variance, P(win), verdict\n"
        "- bayes: priors[], likelihoods[] (P(E|H_i)), num_observations (iid repeats), hypotheses[] -> posteriors\n"
        "- dice_reroll: faces, rolls (max rolls, may stop anytime, paid face), cost_per_roll -> optimal EV + thresholds\n"
        "- dice_max: faces, dice -> E[max], E[min], E[sum]\n"
        "- card_red_black: red, black (+1 red/-1 black, stop anytime) -> optimal EV\n"
        "- coin_pattern: pattern ('HTH'), p_heads -> expected flips until pattern\n"
        "- coin_race: pattern_a, pattern_b, p_heads -> P(A first), P(B first), expected length (Penney's game)\n"
        "- gamblers_ruin: start, target, p_win -> P(reach target), expected steps\n"
        "- coupon_collector: coupons, collect (distinct wanted, default all) -> expected draws, variance\n"
        "- kelly: p_win, odds (net b:1), loss_fraction (default 1) -> Kelly fraction, edge, log-growth"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "puzzle": {"type": "string", "enum": list(_EV_SOLVERS), "description": "Which solver to run."},
            "outcomes": {"type": "array", "items": {"type": ["number", "string"]}, "description": "custom: payoff of each outcome."},
            "probabilities": {"type": "array", "items": {"type": ["number", "string"]}, "description": "custom: probability of each outcome (default uniform)."},
            "cost": {"type": ["number", "string"], "description": "custom: price paid to play (subtracted from every payoff)."},
            "priors": {"type": "array", "items": {"type": ["number", "string"]}, "description": "bayes: prior P(H_i)."},
            "likelihoods": {"type": "array", "items": {"type": ["number", "string"]}, "description": "bayes: P(evidence | H_i) for ONE observation."},
            "num_observations": {"type": "integer", "description": "bayes: number of iid observations of the evidence (default 1)."},
            "hypotheses": {"type": "array", "items": {"type": "string"}, "description": "bayes: optional labels."},
            "faces": {"type": "integer", "description": "dice_*: sides per die (default 6)."},
            "rolls": {"type": "integer", "description": "dice_reroll: maximum number of rolls (default 2)."},
            "cost_per_roll": {"type": ["number", "string"], "description": "dice_reroll: fee charged per roll (default 0)."},
            "dice": {"type": "integer", "description": "dice_max: number of dice (default 2)."},
            "red": {"type": "integer", "description": "card_red_black: red cards (default 26)."},
            "black": {"type": "integer", "description": "card_red_black: black cards (default 26)."},
            "pattern": {"type": "string", "description": "coin_pattern: target sequence of H/T."},
            "pattern_a": {"type": "string", "description": "coin_race: first player's H/T sequence."},
            "pattern_b": {"type": "string", "description": "coin_race: second player's H/T sequence."},
            "p_heads": {"type": ["number", "string"], "description": "coin_*: P(heads), default 1/2."},
            "start": {"type": "integer", "description": "gamblers_ruin: starting bankroll (units)."},
            "target": {"type": "integer", "description": "gamblers_ruin: target bankroll (absorbing)."},
            "p_win": {"type": ["number", "string"], "description": "gamblers_ruin / kelly: win probability per bet."},
            "coupons": {"type": "integer", "description": "coupon_collector: number of distinct coupon types."},
            "collect": {"type": "integer", "description": "coupon_collector: distinct types wanted (default all)."},
            "odds": {"type": ["number", "string"], "description": "kelly: net odds b (win b per 1 staked)."},
            "loss_fraction": {"type": ["number", "string"], "description": "kelly: fraction of stake lost on a loss (default 1)."},
        },
        "required": ["puzzle"],
    },
}


# ════════════════════════════════════════════════════════════════════════
# 3. Order book metrics
# ════════════════════════════════════════════════════════════════════════

def _parse_levels(raw: Any, side: str) -> List[Tuple[float, float]]:
    if not isinstance(raw, list) or not raw:
        raise ToolInputError(f"'{side}' must be a non-empty list of [price, size] levels")
    if len(raw) > 10_000:
        raise ToolInputError(f"'{side}': at most 10,000 levels")
    levels: Dict[float, float] = {}
    for i, lvl in enumerate(raw):
        if isinstance(lvl, dict):
            price, size = lvl.get("price"), lvl.get("size", lvl.get("quantity", lvl.get("qty")))
        elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
            price, size = lvl[0], lvl[1]
        else:
            raise ToolInputError(f"{side}[{i}] must be [price, size] or {{'price','size'}}")
        try:
            px, sz = float(price), float(size)
        except (TypeError, ValueError):
            raise ToolInputError(f"{side}[{i}]: price and size must be numbers")
        if not (math.isfinite(px) and math.isfinite(sz)) or px <= 0 or sz < 0:
            raise ToolInputError(f"{side}[{i}]: need price > 0 and size >= 0")
        if sz > 0:
            levels[px] = levels.get(px, 0.0) + sz  # aggregate duplicate prices
    if not levels:
        raise ToolInputError(f"'{side}' has no levels with positive size")
    return sorted(levels.items(), key=lambda t: -t[0] if side == "bids" else t[0])


def _vwap(levels: List[Tuple[float, float]]) -> Optional[float]:
    vol = sum(s for _, s in levels)
    return sum(p * s for p, s in levels) / vol if vol > 0 else None


def _walk(levels: List[Tuple[float, float]], qty: float) -> Dict[str, Any]:
    left, cost, worst = qty, 0.0, None
    for px, sz in levels:
        take = min(left, sz)
        cost += take * px
        left -= take
        worst = px
        if left <= 1e-12:
            break
    filled = qty - max(left, 0.0)
    return {"requested": qty, "filled": filled, "unfilled": max(left, 0.0),
            "avg_price": cost / filled if filled > 0 else None, "worst_price": worst,
            "notional": cost}


def _order_book_metrics(args: dict) -> Dict[str, Any]:
    """Top-of-book and depth metrics for a limit order book snapshot."""
    bids = _parse_levels(args.get("bids"), "bids")
    asks = _parse_levels(args.get("asks"), "asks")
    depth_n = _int(args, "levels", 5, lo=1, hi=10_000)
    tick = _num(args, "tick_size", None, lo=0, lo_inclusive=False)

    (pb, qb), (pa, qa) = bids[0], asks[0]
    mid = 0.5 * (pb + pa)
    spread = pa - pb
    imbalance = (qb - qa) / (qb + qa)
    microprice = (pb * qa + pa * qb) / (qb + qa)  # size-weighted mid

    warnings: List[str] = []
    state = "normal"
    if spread < 0:
        state = "crossed"
        warnings.append("Crossed book (best bid > best ask): stale data or arbitrage — metrics are suspect.")
    elif spread == 0:
        state = "locked"
        warnings.append("Locked book (best bid == best ask).")

    bd, ad = bids[:depth_n], asks[:depth_n]
    bid_vol, ask_vol = sum(s for _, s in bd), sum(s for _, s in ad)
    bid_vwap, ask_vwap = _vwap(bd), _vwap(ad)
    depth_wmid = ((bid_vwap * ask_vol + ask_vwap * bid_vol) / (bid_vol + ask_vol)
                  if bid_vwap is not None and ask_vwap is not None else None)

    out: Dict[str, Any] = {
        "book_state": state,
        "best_bid": {"price": pb, "size": qb},
        "best_ask": {"price": pa, "size": qa},
        "mid_price": mid,
        "spread": spread,
        "spread_bps": spread / mid * 1e4,
        "microprice": microprice,
        "microprice_minus_mid": microprice - mid,
        "microprice_signal": ("up-tick pressure (bid-heavy)" if microprice > mid else
                              "down-tick pressure (ask-heavy)" if microprice < mid else "balanced"),
        "top_of_book_imbalance": imbalance,
        "depth": {
            "levels_used": {"bids": len(bd), "asks": len(ad)},
            "bid_volume": bid_vol, "ask_volume": ask_vol,
            "depth_imbalance": (bid_vol - ask_vol) / (bid_vol + ask_vol),
            "bid_vwap": bid_vwap, "ask_vwap": ask_vwap,
            "depth_weighted_mid": depth_wmid,
        },
        "definitions": {
            "microprice": "(bid*ask_size + ask*bid_size)/(bid_size+ask_size) — top-of-book size-weighted mid",
            "imbalance": "(bid_size - ask_size)/(bid_size + ask_size), in [-1, 1]",
            "depth_weighted_mid": "same weighting as microprice using VWAPs/volumes over the first N levels",
        },
    }
    if tick:
        out["spread_ticks"] = spread / tick

    qty = args.get("impact_quantity")
    if qty is not None:
        q = _num(args, "impact_quantity", lo=0, lo_inclusive=False)
        buy, sell = _walk(asks, q), _walk(bids, q)
        for leg, ref in ((buy, 1), (sell, -1)):
            if leg["avg_price"] is not None:
                leg["slippage_vs_mid"] = ref * (leg["avg_price"] - mid)
                leg["slippage_bps"] = leg["slippage_vs_mid"] / mid * 1e4
            if leg["unfilled"] > 0:
                warnings.append(f"Book too thin: {leg['unfilled']:g} of {q:g} unfilled on one side.")
        out["market_impact"] = {"buy_market_order": buy, "sell_market_order": sell}

    if warnings:
        out["warnings"] = warnings
    return out


ORDER_BOOK_SCHEMA = {
    "name": "order_book_metrics",
    "description": (
        "Compute limit-order-book metrics from bid/ask ladders: best bid/ask, mid-price, spread "
        "(absolute, bps, ticks), size-weighted microprice and its deviation from mid, top-of-book "
        "and depth imbalance, depth VWAPs / depth-weighted mid over N levels, crossed/locked "
        "detection, and optional walk-the-book slippage for a market order of impact_quantity."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "bids": {"type": "array", "description": "Bid levels as [[price, size], ...] or [{price, size}, ...]; any order.",
                     "items": {"type": ["array", "object"]}},
            "asks": {"type": "array", "description": "Ask levels as [[price, size], ...] or [{price, size}, ...]; any order.",
                     "items": {"type": ["array", "object"]}},
            "levels": {"type": "integer", "description": "Depth levels per side for depth metrics (default 5)."},
            "tick_size": {"type": "number", "description": "Optional tick size to express spread in ticks."},
            "impact_quantity": {"type": "number", "description": "Optional size of a hypothetical market order to estimate slippage both ways."},
        },
        "required": ["bids", "asks"],
    },
}


# ════════════════════════════════════════════════════════════════════════
# Persona — elite Waterloo quantitative researcher
# ════════════════════════════════════════════════════════════════════════

PERSONA_SECTION_ID = "waterloo-jane-street-qr.persona"

PERSONA_PROMPT = """\
# Quant Research Mode (waterloo-jane-street-qr)
You are a quantitative researcher: Waterloo Math/CS, IMO medallist, ICPC World Finalist, Putnam \
top-scorer, now on a Jane Street / Citadel style trading desk. Apply this standard to every \
quantitative, probabilistic, trading or decision question:

1. Model first. State the random variables, sample space, assumptions (independence, \
replacement, risk-neutral vs physical measure) and what is being optimised. Flag any ambiguity \
in one line and pick the standard interpretation.
2. Rigour over vibes. Derive with exact math — conditioning, linearity of expectation, \
symmetry, recursion / backward induction, martingales, generating functions, Markov chains. \
Prefer exact fractions, then decimals. Never assert a number you have not derived or computed.
3. Verify. Sanity-check every answer two ways when cheap: limiting cases, a bound, a symmetry \
argument, parity, or a tool call. Use the plugin tools (black_scholes, ev_calculator, \
order_book_metrics) for computation instead of mental arithmetic, and cite their outputs.
4. Think in distributions, not points. Report EV together with variance / tail risk; separate \
edge from sizing (Kelly, fractional Kelly); call out adverse selection, fees, slippage, and \
model risk.
5. Desk clarity. Lead with the answer (one line, exact value), then a tight derivation, then \
the trade-off. No filler, no hedging adjectives, no hand-waving. Quote probabilities as \
numbers, not words.
6. Decision rule. End every decision-type answer with an explicit verdict: \
"EV = <value> -> TAKE (EV > 0)" or "EV = <value> -> PASS (EV <= 0)", plus a sizing note when \
relevant. If the information is insufficient for a sign on EV, say exactly which quantity \
would flip it.
7. Calibrated honesty. When uncertain, give a confidence interval or the assumption that \
breaks the answer. Being precisely wrong is worse than being approximately right — and both \
are worse than saying what you don't know.

Non-quant requests are handled normally, but stay concise and precise.
"""


def _persona_context_hook(**_kwargs: Any) -> Optional[Dict[str, str]]:
    """Fallback for hosts without system-prompt sections: per-turn context injection."""
    return {"context": PERSONA_PROMPT}


# ════════════════════════════════════════════════════════════════════════
# Registration
# ════════════════════════════════════════════════════════════════════════

black_scholes = _safe(_black_scholes)
ev_calculator = _safe(_ev_calculator)
order_book_metrics = _safe(_order_book_metrics)

_TOOLS = (
    ("black_scholes", BLACK_SCHOLES_SCHEMA, black_scholes, "📈"),
    ("ev_calculator", EV_CALCULATOR_SCHEMA, ev_calculator, "🎲"),
    ("order_book_metrics", ORDER_BOOK_SCHEMA, order_book_metrics, "📊"),
)


def register(ctx) -> None:
    """Called once by Hermes at plugin load."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset=TOOLSET,
            schema=schema,
            handler=handler,
            description=schema["description"].split("\n", 1)[0],
            emoji=emoji,
        )

    # Persona: prefer the cache-safe system-prompt section (frozen per session, so it never
    # breaks prompt caching). Fall back to a pre_llm_call context hook on older hosts.
    if hasattr(ctx, "register_system_prompt_section"):
        ctx.register_system_prompt_section(PERSONA_SECTION_ID, PERSONA_PROMPT, max_chars=4000)
    elif hasattr(ctx, "register_hook"):
        ctx.register_hook("pre_llm_call", _persona_context_hook)
