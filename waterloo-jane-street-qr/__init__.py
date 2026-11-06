"""waterloo-jane-street-qr — quant research toolkit for Hermes Agent.

Tools
-----
Derivatives & volatility
  black_scholes              European BSM prices, Greeks, parity check, implied vol.
  binomial_tree_american     CRR tree for American options: price, early-exercise premium,
                             exercise boundary, tree Greeks, BS control-variate correction.
  heston_price               Stochastic-volatility European prices (Lewis/Fourier, "little
                             Heston trap" CF), implied-vol smile, Feller check, Delta/Gamma.
  monte_carlo_engine         GBM Monte Carlo with antithetic variates (+ control variates)
                             for Asian, barrier and European options with analytic benchmarks.
Market making & microstructure
  order_book_metrics         Mid, spread, microprice, imbalance, depth, walk-the-book impact.
  avellaneda_stoikov_quotes  Reservation price + optimal bid/ask (AS 2008, GLFT asymptotics).
  order_flow_toxicity        Volume-bucketed VPIN (BVC / tick rule / given sides), Kyle lambda.
Probability, stopping & games
  ev_calculator              Exact interview-puzzle EVs (Bayes, dice, cards, coins, Kelly, ...).
  optimal_stopping           Secretary problem; iid-offer threshold stopping (finite/infinite
                             horizon, search cost, discounting).
  matrix_game_solver         Exact mixed-strategy Nash equilibrium of zero-sum matrix games
                             (rational simplex + LP duality certificate).

Context & enforcement
---------------------
* A system-prompt section injects the elite desk persona and the mandatory answer
  contract: state space -> transitions -> terminal payoffs -> EV -> [TAKE]/[PASS]/[HEDGE].
* ``pre_llm_call`` adds a one-line protocol reminder each turn; ``transform_llm_output``
  audits the final answer and appends a compliance flag when EV/verdict are missing.
  Mode: ``plugins.entries.waterloo-jane-street-qr.settings.enforcement`` = strict|reminder|off.

Every tool handler takes the LLM's args dict and returns a JSON string. Handlers never raise.
"""

from __future__ import annotations

import cmath
import json
import logging
import math
import random
import re
import time
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
        if isinstance(val, bool):
            raise TypeError
        x = float(_frac(val, key)) if isinstance(val, str) else float(val)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError, ToolInputError):
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
        if isinstance(val, bool):
            raise TypeError
        f = float(val)
    except (TypeError, ValueError, OverflowError):
        raise ToolInputError(f"'{key}' must be an integer, got {val!r}")
    if not math.isfinite(f):
        raise ToolInputError(f"'{key}' must be finite")
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
        if isinstance(val, str) and len(val) <= 64:
            f = Fraction(val.strip())
            if f.numerator.bit_length() > 4096 or f.denominator.bit_length() > 4096:
                raise ValueError
            return f
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        pass
    raise ToolInputError(f"{name} must be a number or fraction string like '1/6', got {val!r}")


def _choice(args: dict, key: str, choices: Sequence[str], default: str) -> str:
    val = args.get(key, default)
    if val is None:
        val = default
    s = str(val).strip().lower().replace("_", "-")
    norm = {c.replace("_", "-"): c for c in choices}
    if s not in norm:
        raise ToolInputError(f"'{key}' must be one of {list(choices)}, got {val!r}")
    return norm[s]


def _bool(args: dict, key: str, default: bool) -> bool:
    val = args.get(key, default)
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)) and val in (0, 1):
        return bool(val)
    if isinstance(val, str) and val.strip().lower() in ("true", "false", "yes", "no", "1", "0"):
        return val.strip().lower() in ("true", "yes", "1")
    raise ToolInputError(f"'{key}' must be a boolean, got {val!r}")


def _matrix(raw: Any, name: str, max_dim: int = 25) -> List[List[Fraction]]:
    if not isinstance(raw, list) or not raw or not all(isinstance(r, list) and r for r in raw):
        raise ToolInputError(f"'{name}' must be a non-empty list of non-empty rows")
    width = len(raw[0])
    if any(len(r) != width for r in raw):
        raise ToolInputError(f"'{name}' rows must all have the same length")
    if len(raw) > max_dim or width > max_dim:
        raise ToolInputError(f"'{name}' is limited to {max_dim}x{max_dim}")
    return [[_frac(v, f"{name} entry") for v in r] for r in raw]


def _mean_var(xs: Sequence[float]) -> Tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = math.fsum(xs) / n
    if n < 2:
        return m, 0.0
    return m, math.fsum((x - m) ** 2 for x in xs) / (n - 1)


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
# 4. American options — Cox-Ross-Rubinstein binomial tree
# ════════════════════════════════════════════════════════════════════════

def _crr_tree(S: float, K: float, T: float, r: float, q: float, sigma: float, N: int,
              kind: str, with_european: bool = True, with_boundary: bool = True) -> Dict[str, Any]:
    """One backward pass pricing American (and European) options on the same CRR lattice."""
    dt = T / N
    vs = sigma * math.sqrt(dt)
    u = math.exp(vs)
    d = 1.0 / u
    growth = math.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ToolInputError(
            f"risk-neutral up-probability p={p:.6g} is outside (0,1): |r-q| is too large relative to "
            f"sigma*sqrt(dt). Increase 'steps' (now {N}) or check inputs.")
    disc = math.exp(-r * dt)
    pu, pd = disc * p, disc * (1.0 - p)
    sgn = 1.0 if kind == "call" else -1.0
    # spot_at[k + N] = S * u^k for k in [-N, N]
    spot_at = [S * math.exp(k * vs) for k in range(-N, N + 1)]

    am = [max(sgn * (spot_at[2 * j] - K), 0.0) for j in range(N + 1)]  # j up-moves at step N
    eu = am[:] if with_european else None
    boundary: List[Optional[float]] = [None] * (N + 1)
    itm_terminal = [spot_at[2 * j] for j in range(N + 1) if sgn * (spot_at[2 * j] - K) > 0]
    if itm_terminal:
        boundary[N] = K  # at expiry: exercise iff in the money
    snap: Dict[int, Tuple[List[float], Optional[List[float]]]] = {}

    for i in range(N - 1, -1, -1):
        crit: Optional[float] = None
        base = N - i  # node (i, j) price index: (2j - i) + N = base + 2j
        for j in range(i + 1):
            cont = pu * am[j + 1] + pd * am[j]
            ex = sgn * (spot_at[base + 2 * j] - K)
            if ex > cont and ex > 0.0:
                am[j] = ex
                if with_boundary:
                    s_node = spot_at[base + 2 * j]
                    if crit is None or (s_node > crit if sgn < 0 else s_node < crit):
                        crit = s_node
            else:
                am[j] = cont
            if eu is not None:
                eu[j] = pu * eu[j + 1] + pd * eu[j]
        boundary[i] = crit
        if i <= 2:
            snap[i] = (am[: i + 1], eu[: i + 1] if eu is not None else None)

    return {"american": am[0], "european": eu[0] if eu is not None else None, "p_up": p, "u": u,
            "d": d, "dt": dt, "boundary": boundary, "snap": snap}


def _binomial_tree_american(args: dict) -> Dict[str, Any]:
    """American option on a CRR lattice with exercise boundary and Greeks."""
    S = _num(args, "spot", required=True, lo=0, lo_inclusive=False)
    K = _num(args, "strike", required=True, lo=0, lo_inclusive=False)
    T = _num(args, "time_to_maturity", required=True, lo=0, lo_inclusive=False, hi=100)
    r = _num(args, "risk_free_rate", required=True, lo=-1, hi=5)
    sigma = _num(args, "volatility", required=True, lo=0, lo_inclusive=False, hi=20)
    q = _num(args, "dividend_yield", 0.0, lo=-1, hi=5)
    kind = _choice(args, "option_type", ("call", "put"), "put")
    N = _int(args, "steps", 500, lo=2, hi=2000)
    npts = _int(args, "boundary_points", 20, lo=0, hi=200)
    want_greeks = _bool(args, "greeks", True)
    if not 1e-6 <= S / K <= 1e6:
        raise ToolInputError("spot/strike ratio must be within [1e-6, 1e6]")

    t0 = time.perf_counter()
    tree = _crr_tree(S, K, T, r, q, sigma, N, kind)
    am, eu_tree = tree["american"], tree["european"]
    bs = _bs_core(S, K, T, r, sigma, q)[kind]["price"]
    intrinsic = max((S - K) if kind == "call" else (K - S), 0.0)

    result: Dict[str, Any] = {
        "model": "Cox-Ross-Rubinstein binomial tree",
        "inputs": {"spot": S, "strike": K, "time_to_maturity_years": T, "risk_free_rate": r,
                   "volatility": sigma, "dividend_yield": q, "option_type": kind, "steps": N},
        "lattice": {"u": tree["u"], "d": tree["d"], "p_up_risk_neutral": tree["p_up"], "dt": tree["dt"]},
        "american_price": am,
        "european_price_tree": eu_tree,
        "european_price_black_scholes": bs,
        # Hull's control variate: the lattice error is nearly identical for the American and
        # European claims, so correct the American price by the known European error.
        "american_price_control_variate": am + (bs - eu_tree),
        "early_exercise_premium": am - eu_tree,
        "intrinsic_value": intrinsic,
        "exercise_now_optimal": intrinsic > 0 and intrinsic >= am - 1e-12,
    }

    snap = tree["snap"]
    u, d = tree["u"], tree["d"]
    if want_greeks and 2 in snap and 1 in snap:
        v1, v2 = snap[1][0], snap[2][0]
        s_u, s_d, s_uu, s_dd = S * u, S * d, S * u * u, S * d * d
        delta = (v1[1] - v1[0]) / (s_u - s_d)
        gamma = ((v2[2] - v2[1]) / (s_uu - S) - (v2[1] - v2[0]) / (S - s_dd)) / (0.5 * (s_uu - s_dd))
        theta = (v2[1] - am) / (2 * tree["dt"])
        greeks: Dict[str, Any] = {"delta": delta, "gamma": gamma, "theta": theta,
                                  "theta_per_day": theta / 365.0}
        if N <= 1000:  # bump-and-reprice (2 trees each) only when cheap
            hv, hr = max(1e-4, 0.01 * sigma), 1e-4
            f = lambda **kw: _crr_tree(**{**dict(S=S, K=K, T=T, r=r, q=q, sigma=sigma, N=N, kind=kind,
                                                with_european=False, with_boundary=False), **kw})["american"]
            greeks["vega"] = (f(sigma=sigma + hv) - f(sigma=sigma - hv)) / (2 * hv)
            greeks["vega_per_vol_point"] = greeks["vega"] / 100.0
            greeks["rho"] = (f(r=r + hr) - f(r=r - hr)) / (2 * hr)
            greeks["method"] = "delta/gamma/theta from lattice nodes; vega/rho by central bump-and-reprice"
        else:
            greeks["method"] = "delta/gamma/theta from lattice nodes (vega/rho skipped for steps > 1000)"
        result["greeks"] = greeks

    bnd = tree["boundary"]
    has_early = any(b is not None for b in bnd[:-1])
    if npts and has_early:
        idx = sorted({round(k * N / max(npts - 1, 1)) for k in range(npts)}) if npts > 1 else [0]
        pts = [{"time": i * tree["dt"], "time_to_expiry": T - i * tree["dt"], "critical_spot": bnd[i]}
               for i in idx]
        result["early_exercise_boundary"] = {
            "rule": ("exercise the put when spot <= critical_spot" if kind == "put"
                     else "exercise the call when spot >= critical_spot"),
            "points": pts,
            "note": "critical_spot is the lattice node on the exercise frontier (resolution ~ one tree step); "
                    "null = no exercise region at that date",
        }
    elif not has_early:
        result["early_exercise_boundary"] = {
            "points": [],
            "note": ("never optimal to exercise early (American call on a non-dividend asset = European)"
                     if kind == "call" and q <= 0 else "no early-exercise region on this lattice"),
        }
    result["runtime_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


BINOMIAL_SCHEMA = {
    "name": "binomial_tree_american",
    "description": (
        "Price an American call/put on a Cox-Ross-Rubinstein binomial tree. Returns American and European "
        "lattice prices, the Black-Scholes control-variate-corrected American price, early-exercise "
        "premium, whether exercising now is optimal, the early-exercise boundary (critical spot vs time), "
        "and Greeks (lattice Delta/Gamma/Theta; bumped Vega/Rho). Rates/vols are decimals, time in years."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "spot": {"type": "number", "description": "Underlying spot S (> 0)."},
            "strike": {"type": "number", "description": "Strike K (> 0)."},
            "time_to_maturity": {"type": "number", "description": "Years to expiry (> 0)."},
            "risk_free_rate": {"type": "number", "description": "Continuously compounded r, decimal."},
            "volatility": {"type": "number", "description": "Annualized sigma, decimal (> 0)."},
            "dividend_yield": {"type": "number", "description": "Continuous dividend yield q, decimal. Default 0."},
            "option_type": {"type": "string", "enum": ["call", "put"], "description": "Default 'put'."},
            "steps": {"type": "integer", "description": "Tree steps N (2-2000, default 500)."},
            "boundary_points": {"type": "integer", "description": "Points to sample on the exercise boundary (default 20, 0 = omit)."},
            "greeks": {"type": "boolean", "description": "Compute Greeks (default true)."},
        },
        "required": ["spot", "strike", "time_to_maturity", "risk_free_rate", "volatility"],
    },
}


# ════════════════════════════════════════════════════════════════════════
# 5. Heston stochastic volatility
# ════════════════════════════════════════════════════════════════════════

def _gauss_legendre(n: int) -> Tuple[List[float], List[float]]:
    """Nodes/weights on [-1, 1] by Newton iteration on P_n."""
    xs, ws = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1.0)
            dx = p1 / dp
            x -= dx
            if abs(dx) < 1e-16:
                break
        p0, p1 = 1.0, x
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1.0)
        xs.append(x)
        ws.append(2.0 / ((1.0 - x * x) * dp * dp))
    return xs, ws


_GL_X, _GL_W = _gauss_legendre(16)


def _heston_cf(z: complex, x0: float, T: float, v0: float, kappa: float, theta: float,
               eta: float, rho: float) -> complex:
    """E[exp(i z ln S_T)] — Albrecher et al. 'little Heston trap' form (no branch-cut jumps).
    x0 = ln S + (r - q) T."""
    iz = 1j * z
    b = kappa - rho * eta * iz
    dd = cmath.sqrt(b * b + eta * eta * (iz + z * z))
    g = (b - dd) / (b + dd)
    e = cmath.exp(-dd * T)
    eta2 = eta * eta
    C = (kappa * theta / eta2) * ((b - dd) * T - 2.0 * cmath.log((1.0 - g * e) / (1.0 - g)))
    D = ((b - dd) / eta2) * ((1.0 - e) / (1.0 - g * e))
    return cmath.exp(iz * x0 + C + D * v0)


def _heston_call(S: float, K: float, T: float, r: float, q: float, v0: float, kappa: float,
                 theta: float, eta: float, rho: float) -> Tuple[float, Dict[str, Any]]:
    """Lewis (2001) single-integral formula:
    C = S e^{-qT} - sqrt(K) e^{-rT}/pi * int_0^inf Re[e^{-iu ln K} phi(u - i/2)] / (u^2 + 1/4) du."""
    x0 = math.log(S) + (r - q) * T
    lnK = math.log(K)

    def f(u: float) -> float:
        val = cmath.exp(-1j * u * lnK) * _heston_cf(complex(u, -0.5), x0, T, v0, kappa, theta, eta, rho)
        return val.real / (u * u + 0.25)

    total, a, width, u_max = 0.0, 0.0, 2.0, 20_000.0
    scale = math.sqrt(S * K) + 1.0
    panels = 0
    truncated = True
    while a < u_max:
        b = a + width
        half, mid = 0.5 * width, a + 0.5 * width
        vals = [f(mid + half * x) for x in _GL_X]
        total += half * math.fsum(w * v for w, v in zip(_GL_W, vals))
        panels += 1
        a = b
        # Tail bound: integrand decays at least like 1/u^2 from here on.
        if b >= 20.0 and max(abs(v) for v in vals) * b < 1e-13 * scale:
            truncated = False
            break
    call = S * math.exp(-q * T) - math.sqrt(K) * math.exp(-r * T) / math.pi * total
    return call, {"integration_upper_limit": a, "panels": panels, "truncated": truncated}


def _heston_price(args: dict) -> Dict[str, Any]:
    """European options under Heston (1993) stochastic volatility."""
    S = _num(args, "spot", required=True, lo=0, lo_inclusive=False)
    K = _num(args, "strike", required=True, lo=0, lo_inclusive=False)
    T = _num(args, "time_to_maturity", required=True, lo=0, lo_inclusive=False, hi=50)
    r = _num(args, "risk_free_rate", required=True, lo=-1, hi=5)
    q = _num(args, "dividend_yield", 0.0, lo=-1, hi=5)
    v0 = _num(args, "v0", required=True, lo=0, hi=25)
    kappa = _num(args, "kappa", required=True, lo=0, lo_inclusive=False, hi=100)
    theta = _num(args, "theta", required=True, lo=0, hi=25)
    eta = _num(args, "vol_of_vol", required=True, lo=1e-4, hi=10)
    rho = _num(args, "rho", required=True, lo=-1, hi=1)
    kind = _choice(args, "option_type", ("call", "put", "both"), "both")
    strikes_raw = args.get("strikes")
    want_greeks = _bool(args, "greeks", True)
    rho = max(-0.999999, min(0.999999, rho))

    def price_pair(s: float, k: float, v: float = v0) -> Tuple[float, float, Dict[str, Any]]:
        if not 1e-4 <= k / s <= 1e4:
            raise ToolInputError("strike/spot ratio must be within [1e-4, 1e4]")
        c, info = _heston_call(s, k, T, r, q, v, kappa, theta, eta, rho)
        lo_c = max(s * math.exp(-q * T) - k * math.exp(-r * T), 0.0)
        c = min(max(c, lo_c), s * math.exp(-q * T))  # clip quadrature noise into no-arb bounds
        p = c - s * math.exp(-q * T) + k * math.exp(-r * T)
        return c, max(p, 0.0), info

    t0 = time.perf_counter()
    call, put, info = price_pair(S, K)
    feller = 2 * kappa * theta - eta * eta
    out: Dict[str, Any] = {
        "model": "Heston (1993) stochastic volatility; Lewis Fourier integral, Gauss-Legendre quadrature",
        "inputs": {"spot": S, "strike": K, "time_to_maturity_years": T, "risk_free_rate": r,
                   "dividend_yield": q, "v0": v0, "kappa": kappa, "theta": theta, "vol_of_vol": eta, "rho": rho},
        "feller_condition": {"two_kappa_theta_minus_eta_sq": feller, "satisfied": feller > 0,
                             "note": "if violated, variance can touch 0 (prices remain valid)"},
        "expected_avg_variance": theta + (v0 - theta) * (1 - math.exp(-kappa * T)) / (kappa * T),
        "numerics": info,
    }
    out["expected_avg_vol_proxy"] = math.sqrt(max(out["expected_avg_variance"], 0.0))

    def iv(px: float, leg: str, k: float) -> Optional[float]:
        res = _implied_vol(px, leg, S, k, T, r, q)
        return res.get("implied_volatility")

    for leg, px in (("call", call), ("put", put)):
        if kind in (leg, "both"):
            out[leg] = {"price": px, "black_scholes_implied_vol": iv(px, leg, K)}

    if want_greeks:
        h = 1e-3 * S
        cu, pu_, _ = price_pair(S + h, K)
        cd, pd_, _ = price_pair(S - h, K)
        hv = max(1e-5, 1e-3 * v0)
        cv_u = price_pair(S, K, v0 + hv)[0]
        cv_d = price_pair(S, K, max(v0 - hv, 0.0))[0]
        dv0 = (cv_u - cv_d) / (v0 + hv - max(v0 - hv, 0.0))
        g = {"gamma": (cu - 2 * call + cd) / (h * h), "dprice_dv0": dv0,
             "method": "central finite differences on the semi-analytic price"}
        if kind in ("call", "both"):
            g["call_delta"] = (cu - cd) / (2 * h)
        if kind in ("put", "both"):
            g["put_delta"] = (pu_ - pd_) / (2 * h)
        out["greeks"] = g

    if strikes_raw is not None:
        if not isinstance(strikes_raw, list) or not strikes_raw or len(strikes_raw) > 40:
            raise ToolInputError("'strikes' must be a list of 1-40 strikes")
        smile = []
        for i, k_raw in enumerate(strikes_raw):
            k = _num({"k": k_raw}, "k", lo=0, lo_inclusive=False)
            c, p, _ = price_pair(S, k)
            otm = "put" if k < S * math.exp((r - q) * T) else "call"
            px = p if otm == "put" else c
            smile.append({"strike": k, "call": c, "put": p, "otm_leg": otm,
                          "implied_vol": iv(px, otm, k)})
        out["smile"] = smile
        ivs = [s_["implied_vol"] for s_ in smile if s_["implied_vol"] is not None]
        if len(ivs) >= 2:
            out["smile_note"] = ("negative skew (low strikes richer)" if smile[0]["implied_vol"] and
                                 smile[-1]["implied_vol"] and smile[0]["implied_vol"] > smile[-1]["implied_vol"]
                                 and smile[0]["strike"] < smile[-1]["strike"] else "see per-strike implied vols")
    if info["truncated"]:
        out["warnings"] = ["Fourier integral truncated at u=20000; very short maturity or tiny variance — "
                           "treat the price with caution."]
    out["runtime_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return out


HESTON_SCHEMA = {
    "name": "heston_price",
    "description": (
        "Price European options under the Heston stochastic-volatility model (semi-closed form via the "
        "Lewis Fourier integral). Returns call/put prices, Black-Scholes implied vols, Feller-condition "
        "check, Delta/Gamma/dPrice/dv0, and optionally an implied-vol smile across 'strikes'. "
        "Variance parameters are in variance units (v0=0.04 means 20% vol)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "spot": {"type": "number", "description": "Spot S (> 0)."},
            "strike": {"type": "number", "description": "Strike K (> 0)."},
            "time_to_maturity": {"type": "number", "description": "Years to expiry (> 0)."},
            "risk_free_rate": {"type": "number", "description": "r, decimal."},
            "dividend_yield": {"type": "number", "description": "q, decimal. Default 0."},
            "v0": {"type": "number", "description": "Initial variance (e.g. 0.04 = 20% vol)."},
            "kappa": {"type": "number", "description": "Mean-reversion speed of variance (> 0)."},
            "theta": {"type": "number", "description": "Long-run variance level."},
            "vol_of_vol": {"type": "number", "description": "Volatility of variance eta/xi (> 0)."},
            "rho": {"type": "number", "description": "Spot-variance correlation in [-1, 1] (equities typically < 0)."},
            "option_type": {"type": "string", "enum": ["call", "put", "both"], "description": "Default 'both'."},
            "strikes": {"type": "array", "items": {"type": "number"}, "description": "Optional strikes (<= 40) for an implied-vol smile."},
            "greeks": {"type": "boolean", "description": "Compute Greeks (default true)."},
        },
        "required": ["spot", "strike", "time_to_maturity", "risk_free_rate", "v0", "kappa", "theta", "vol_of_vol", "rho"],
    },
}


# ════════════════════════════════════════════════════════════════════════
# 6. Market making — Avellaneda-Stoikov (2008) + GLFT asymptotics
# ════════════════════════════════════════════════════════════════════════

def _avellaneda_stoikov_quotes(args: dict) -> Dict[str, Any]:
    """Inventory-aware optimal quotes."""
    s = _num(args, "mid_price", required=True, lo=0, lo_inclusive=False)
    inv = _num(args, "inventory", 0.0, lo=-1e12, hi=1e12)
    gamma = _num(args, "risk_aversion", required=True, lo=0, lo_inclusive=False, hi=1e6)
    sigma = _num(args, "volatility", required=True, lo=0, hi=1e9)
    tau = _num(args, "time_remaining", 1.0, lo=0, hi=1e6)
    k = _num(args, "order_arrival_decay", required=True, lo=0, lo_inclusive=False, hi=1e9)
    A = _num(args, "order_arrival_rate", None, lo=0, lo_inclusive=False)
    tick = _num(args, "tick_size", None, lo=0, lo_inclusive=False)
    qmax = _num(args, "max_inventory", None, lo=0, lo_inclusive=False)

    var_term = gamma * sigma * sigma * tau
    liq_term = (2.0 / gamma) * math.log1p(gamma / k)
    reservation = s - inv * var_term
    spread = var_term + liq_term
    bid, ask = reservation - spread / 2.0, reservation + spread / 2.0
    warnings: List[str] = []

    out: Dict[str, Any] = {
        "model": "Avellaneda-Stoikov (2008): exponential utility, arithmetic BM mid, fills ~ A*exp(-k*delta)",
        "inputs": {"mid_price": s, "inventory": inv, "risk_aversion": gamma, "volatility": sigma,
                   "time_remaining": tau, "order_arrival_decay": k, "order_arrival_rate": A},
        "reservation_price": reservation,
        "reservation_skew": reservation - s,
        "optimal_spread": spread,
        "spread_decomposition": {"inventory_risk": var_term, "adverse_liquidity": liq_term},
        "bid": bid, "ask": ask,
        "bid_distance_from_mid": s - bid, "ask_distance_from_mid": ask - s,
        "formulas": {"reservation": "r = s - q*gamma*sigma^2*(T-t)",
                     "spread": "delta_a + delta_b = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)"},
    }
    if inv != 0:
        out["interpretation"] = ("long inventory: quotes skewed DOWN to sell" if inv > 0
                                 else "short inventory: quotes skewed UP to buy")
    if bid > s or ask < s:
        warnings.append("A quote crosses the mid: inventory pressure dominates — you are aggressively "
                        "unwinding (expect to pay the spread).")

    if A is not None:
        db, da = s - bid, ask - s
        out["fill_intensity"] = {"bid": A * math.exp(-k * max(db, 0.0)) if db >= 0 else A,
                                 "ask": A * math.exp(-k * max(da, 0.0)) if da >= 0 else A,
                                 "note": "lambda(delta) = A*exp(-k*delta) per unit time (capped at A for crossing quotes)"}
        # Gueant-Lehalle-Fernandez-Tapia closed-form asymptotic (infinite horizon) quotes.
        base = (1.0 / gamma) * math.log1p(gamma / k)
        try:
            c = math.sqrt((sigma * sigma * gamma) / (2.0 * k * A) * (1.0 + gamma / k) ** (1.0 + k / gamma))
        except OverflowError:
            c = None
        if c is not None and math.isfinite(c):
            gb, ga = base + (2 * inv + 1) / 2.0 * c, base - (2 * inv - 1) / 2.0 * c
            out["glft_asymptotic"] = {"bid": s - gb, "ask": s + ga, "bid_distance": gb, "ask_distance": ga,
                                      "spread": gb + ga, "skew_per_unit_inventory": c,
                                      "note": "Gueant-Lehalle-Fernandez-Tapia (2013) large-horizon approximation"}

    if qmax is not None:
        if inv >= qmax:
            out["bid"] = None
            warnings.append(f"inventory {inv:g} >= max_inventory {qmax:g}: bid suppressed (sell-only).")
        if inv <= -qmax:
            out["ask"] = None
            warnings.append(f"inventory {inv:g} <= -max_inventory {qmax:g}: ask suppressed (buy-only).")

    if tick:
        rb = math.floor(out["bid"] / tick + 1e-9) * tick if out["bid"] is not None else None
        ra = math.ceil(out["ask"] / tick - 1e-9) * tick if out["ask"] is not None else None
        if rb is not None and ra is not None and ra - rb < tick - 1e-12:
            ra = rb + tick
        out["quotes_on_tick_grid"] = {"bid": rb, "ask": ra, "tick_size": tick,
                                      "rule": "bid floored, ask ceiled (never tighter than model)"}
    if tau == 0:
        warnings.append("time_remaining = 0: inventory term vanishes; only the liquidity spread remains.")
    if warnings:
        out["warnings"] = warnings
    return out


AS_SCHEMA = {
    "name": "avellaneda_stoikov_quotes",
    "description": (
        "Optimal market-making quotes (Avellaneda-Stoikov 2008): reservation price r = s - q*gamma*sigma^2*(T-t) "
        "and optimal spread gamma*sigma^2*(T-t) + (2/gamma)ln(1+gamma/k), giving inventory-skewed bid/ask. "
        "With order_arrival_rate A also returns fill intensities and the Gueant-Lehalle-Fernandez-Tapia "
        "asymptotic quotes. Supports inventory limits and tick rounding. volatility is ABSOLUTE (price units "
        "per sqrt(time unit), same time unit as time_remaining and k/A) — e.g. 2% daily vol on $100 = 2.0/day."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mid_price": {"type": "number", "description": "Current mid price s."},
            "inventory": {"type": "number", "description": "Signed inventory q (long > 0). Default 0."},
            "risk_aversion": {"type": "number", "description": "CARA risk aversion gamma (> 0)."},
            "volatility": {"type": "number", "description": "Absolute volatility sigma of the mid (price / sqrt(time))."},
            "time_remaining": {"type": "number", "description": "T - t in the same time unit (default 1)."},
            "order_arrival_decay": {"type": "number", "description": "k in lambda(delta)=A*exp(-k*delta) (> 0); larger = thinner fills away from mid."},
            "order_arrival_rate": {"type": "number", "description": "Optional A (fills per time unit at delta=0) for intensities and GLFT quotes."},
            "tick_size": {"type": "number", "description": "Optional tick size for rounded quotes."},
            "max_inventory": {"type": "number", "description": "Optional absolute inventory limit; suppresses the side that would breach it."},
        },
        "required": ["mid_price", "risk_aversion", "volatility", "order_arrival_decay"],
    },
}


# ════════════════════════════════════════════════════════════════════════
# 7. Order-flow toxicity — VPIN (volume-synchronized PIN) variant
# ════════════════════════════════════════════════════════════════════════

def _parse_trades(raw: Any) -> Tuple[List[float], List[float], List[Optional[float]]]:
    if not isinstance(raw, list) or len(raw) < 2:
        raise ToolInputError("'trades' must be a list of at least 2 trades: [price, volume(, side)] or "
                             "{price, volume, side}")
    if len(raw) > 200_000:
        raise ToolInputError("'trades' is limited to 200,000 entries")
    prices, vols, sides = [], [], []
    for i, t in enumerate(raw):
        if isinstance(t, dict):
            p, v, sd = t.get("price"), t.get("volume", t.get("size", t.get("qty"))), t.get("side")
        elif isinstance(t, (list, tuple)) and len(t) >= 2:
            p, v, sd = t[0], t[1], (t[2] if len(t) > 2 else None)
        else:
            raise ToolInputError(f"trades[{i}] must be [price, volume(, side)] or an object")
        try:
            if isinstance(p, bool) or isinstance(v, bool):
                raise TypeError
            pf, vf = float(p), float(v)
        except (TypeError, ValueError):
            raise ToolInputError(f"trades[{i}]: price and volume must be numbers")
        if not (math.isfinite(pf) and math.isfinite(vf)) or pf <= 0 or vf < 0:
            raise ToolInputError(f"trades[{i}]: need price > 0 and volume >= 0")
        side: Optional[float] = None
        if sd is not None:
            if isinstance(sd, str):
                m = sd.strip().lower()
                side = {"buy": 1.0, "b": 1.0, "bid": -1.0, "sell": -1.0, "s": -1.0, "ask": 1.0}.get(m)
                if side is None:
                    raise ToolInputError(f"trades[{i}]: side must be buy/sell or +1/-1")
            elif isinstance(sd, (int, float)) and not isinstance(sd, bool) and sd in (1, -1):
                side = float(sd)
            else:
                raise ToolInputError(f"trades[{i}]: side must be buy/sell or +1/-1")
        prices.append(pf)
        vols.append(vf)
        sides.append(side)
    if sum(vols) <= 0:
        raise ToolInputError("total traded volume must be > 0")
    return prices, vols, sides


def _order_flow_toxicity(args: dict) -> Dict[str, Any]:
    """Volume-bucketed order imbalance, VPIN and price impact."""
    prices, vols, sides = _parse_trades(args.get("trades"))
    n = len(prices)
    total = math.fsum(vols)
    method = _choice(args, "classification", ("auto", "bvc", "tick", "given"), "auto")
    if method == "auto":
        method = "given" if all(s is not None for s in sides) else "bvc"
    if method == "given" and any(s is None for s in sides):
        raise ToolInputError("classification='given' needs a side on every trade")

    dps = [prices[i] - prices[i - 1] for i in range(1, n)]
    warnings: List[str] = []
    sigma_dp = _num(args, "price_change_std", None, lo=0, lo_inclusive=False)
    if sigma_dp is None:
        _, var_dp = _mean_var(dps)
        sigma_dp = math.sqrt(var_dp)

    buy_frac: List[float] = []
    if method == "given":
        buy_frac = [1.0 if s > 0 else 0.0 for s in sides]  # type: ignore[operator]
    elif method == "tick":
        last = 0.0
        for i in range(n):
            dp = prices[i] - prices[i - 1] if i else 0.0
            if dp > 0:
                last = 1.0
            elif dp < 0:
                last = -1.0
            buy_frac.append(0.5 if last == 0 else (1.0 if last > 0 else 0.0))
    else:  # bulk volume classification, trade-level bars
        if sigma_dp == 0:
            warnings.append("price never changes: BVC splits every trade 50/50 (VPIN = 0 by construction).")
        for i in range(n):
            dp = prices[i] - prices[i - 1] if i else 0.0
            buy_frac.append(_ncdf(dp / sigma_dp) if sigma_dp > 0 else 0.5)

    nb_target = _int(args, "num_buckets", 50, lo=1, hi=100_000)
    vb = _num(args, "bucket_volume", None, lo=0, lo_inclusive=False)
    if vb is None:
        vb = total / nb_target
    if total / vb > 100_000:
        raise ToolInputError("bucket_volume too small: more than 100,000 buckets")
    window = _int(args, "window", 50, lo=1, hi=100_000)

    buckets: List[Dict[str, float]] = []
    cb = cs = cv = 0.0
    eps = 1e-9 * vb
    for p, v, f in zip(prices, vols, buy_frac):
        rem = v
        while rem > eps:
            take = min(vb - cv, rem)
            cb += take * f
            cs += take * (1.0 - f)
            cv += take
            rem -= take
            if cv >= vb - eps:
                buckets.append({"buy": cb, "sell": cs, "end_price": p})
                cb = cs = cv = 0.0
    partial = {"buy": cb, "sell": cs, "volume": cv}

    buy_tot = math.fsum(b * v for b, v in zip(buy_frac, vols))
    sell_tot = total - buy_tot
    out: Dict[str, Any] = {
        "classification": method,
        "trades": n,
        "total_volume": total,
        "buy_volume": buy_tot,
        "sell_volume": sell_tot,
        "order_imbalance": (buy_tot - sell_tot) / total,
        "bucket_volume": vb,
        "complete_buckets": len(buckets),
        "partial_bucket": partial,
    }
    if method == "bvc":
        out["bvc_price_change_std"] = sigma_dp

    if not buckets:
        warnings.append("no complete volume bucket: lower bucket_volume or raise num_buckets.")
        out["vpin"] = None
    else:
        imb = [abs(b["buy"] - b["sell"]) / vb for b in buckets]
        w = min(window, len(imb))
        if w < window:
            warnings.append(f"only {len(imb)} buckets < window {window}: VPIN uses all buckets.")
        series = [math.fsum(imb[i - w + 1: i + 1]) / w for i in range(w - 1, len(imb))]
        cur = series[-1]
        pct = sum(1 for x in series if x <= cur) / len(series)
        out.update({
            "vpin": cur,
            "vpin_window_buckets": w,
            "vpin_cdf_percentile": pct,
            "vpin_series_tail": series[-50:],
            "bucket_imbalance_tail": [(b["buy"] - b["sell"]) / vb for b in buckets[-20:]],
            "toxicity_level": ("high" if cur >= 0.4 else "elevated" if cur >= 0.2 else "low"),
            "alert": pct >= 0.9 and len(series) >= 10,
            "definition": "VPIN = sum_{last n buckets} |V_buy - V_sell| / (n * V_bucket)",
            "thresholds_note": "low/elevated/high cut-offs (0.2/0.4) are heuristics; use the CDF percentile "
                               "against your own history for alerts.",
        })
        # Kyle's lambda: price impact per unit signed volume, OLS across buckets.
        if len(buckets) >= 3:
            ends = [prices[0]] + [b["end_price"] for b in buckets]
            dP = [ends[i + 1] - ends[i] for i in range(len(buckets))]
            x = [b["buy"] - b["sell"] for b in buckets]
            mx, vx = _mean_var(x)
            my, vy = _mean_var(dP)
            if vx > 0:
                cov = math.fsum((a - mx) * (c - my) for a, c in zip(x, dP)) / (len(x) - 1)
                lam = cov / vx
                r2 = (cov * cov / (vx * vy)) if vy > 0 else None
                out["kyle_lambda"] = {"lambda": lam, "r_squared": r2,
                                      "interpretation": "price move per unit of signed (buy-sell) volume per bucket"}
    if warnings:
        out["warnings"] = warnings
    return out


TOXICITY_SCHEMA = {
    "name": "order_flow_toxicity",
    "description": (
        "Order-flow toxicity from a time-ordered trade tape: buy/sell classification (given sides, tick rule, "
        "or bulk volume classification), volume-bucketed imbalance, VPIN (current value, rolling series, CDF "
        "percentile, alert flag) and Kyle's lambda price impact."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "trades": {"type": "array", "items": {"type": ["array", "object"]},
                       "description": "Time-ordered trades: [price, volume] or [price, volume, side] or {price, volume, side}; side = buy/sell or +1/-1."},
            "classification": {"type": "string", "enum": ["auto", "bvc", "tick", "given"],
                               "description": "auto = given if every trade has a side, else bvc."},
            "bucket_volume": {"type": "number", "description": "Volume per bucket (default total_volume / num_buckets)."},
            "num_buckets": {"type": "integer", "description": "Target bucket count when bucket_volume is omitted (default 50)."},
            "window": {"type": "integer", "description": "VPIN rolling window in buckets (default 50)."},
            "price_change_std": {"type": "number", "description": "Optional sigma of trade-to-trade price changes for BVC (default: sample std)."},
        },
        "required": ["trades"],
    },
}


# ════════════════════════════════════════════════════════════════════════
# 8. Optimal stopping — secretary + iid-offer threshold rules
# ════════════════════════════════════════════════════════════════════════

def _q(x: Any) -> Any:
    """Uniform numeric shape: exact rationals -> {decimal, exact}; floats -> {decimal}."""
    if x is None or isinstance(x, Fraction):
        return x
    return {"decimal": float(x)}


def _secretary(args: dict) -> Dict[str, Any]:
    n = _int(args, "n", required=True, lo=1, hi=10_000_000)
    exact = n <= 500
    if n == 1:
        return {"problem": "secretary", "n": 1, "skip_first": 0, "success_probability": Fraction(1),
                "rule": "accept the only candidate"}
    # P(r) = (r/n) * sum_{i=r}^{n-1} 1/i  for r >= 1;  P(0) = 1/n.
    best_r, best_p = 0, (Fraction(1, n) if exact else 1.0 / n)
    tail = Fraction(0) if exact else 0.0
    probs: Dict[int, Any] = {}
    for r in range(n - 1, 0, -1):
        tail += Fraction(1, r) if exact else 1.0 / r
        p = Fraction(r, n) * tail if exact else (r / n) * tail
        probs[r] = p
        if p >= best_p:
            best_r, best_p = r, p
    nearby = [{"skip_first": r, "success_probability": _q(probs[r])}
              for r in range(max(1, best_r - 2), min(n - 1, best_r + 2) + 1)]
    return {
        "problem": "secretary",
        "n": n,
        "skip_first": best_r,
        "success_probability": _q(best_p),
        "rule": f"Reject the first {best_r} candidates, then accept the first candidate better than all seen.",
        "cutoff_fraction": best_r / n,
        "asymptotic": {"cutoff_fraction": 1 / math.e, "success_probability": 1 / math.e},
        "nearby_cutoffs": nearby,
    }


def _stopping_dist(args: dict):
    dist = _choice(args, "distribution", ("discrete", "uniform", "normal"), "discrete")
    if dist == "discrete":
        outs = args.get("outcomes")
        if not isinstance(outs, list) or not outs or len(outs) > 10_000:
            raise ToolInputError("discrete: 'outcomes' must be a list of 1-10,000 values")
        xs = [_frac(v, "outcome") for v in outs]
        pr = args.get("probabilities")
        if pr is None:
            ps = [Fraction(1, len(xs))] * len(xs)
        else:
            if not isinstance(pr, list) or len(pr) != len(xs):
                raise ToolInputError("discrete: 'probabilities' must match 'outcomes' in length")
            ps = [_prob(p, "probability") for p in pr]
            tot = sum(ps)
            if abs(float(tot) - 1) > 1e-9:
                raise ToolInputError(f"probabilities sum to {float(tot):.12g}, not 1")
            ps = [p / tot for p in ps]
        pairs = list(zip(xs, ps))

        def emax(t):  # E[max(X, t)]
            return sum(p * (x if x > t else t) for x, p in pairs)

        def ptail(t):  # P(X >= t)
            return sum(p for x, p in pairs if x >= t)

        mean = sum(p * x for x, p in pairs)
        return dist, emax, ptail, mean, min(xs), max(xs), True, {"outcomes": len(xs)}

    if dist == "uniform":
        a = _num(args, "low", 0.0)
        b = _num(args, "high", 1.0)
        if not b > a:
            raise ToolInputError("uniform: need high > low")

        def emax(t):
            if t <= a:
                return (a + b) / 2
            if t >= b:
                return t
            return (t * (t - a) + (b * b - t * t) / 2) / (b - a)

        def ptail(t):
            return 1.0 if t <= a else 0.0 if t >= b else (b - t) / (b - a)

        return dist, emax, ptail, (a + b) / 2, a, b, False, {"low": a, "high": b}

    mu = _num(args, "mean", 0.0)
    sd = _num(args, "std", 1.0, lo=0, lo_inclusive=False)

    def emax(t):
        z = (t - mu) / sd
        return t * _ncdf(z) + mu * (1 - _ncdf(z)) + sd * _npdf(z)

    def ptail(t):
        return 1 - _ncdf((t - mu) / sd)

    return dist, emax, ptail, mu, mu - 40 * sd, mu + 40 * sd, False, {"mean": mu, "std": sd}


def _iid_offers(args: dict) -> Dict[str, Any]:
    dist, emax, ptail, mean, lower, upper, is_exact, dinfo = _stopping_dist(args)
    horizon = _int(args, "horizon", 0, lo=0, hi=100_000)
    c_raw = args.get("cost_per_draw", 0)
    d_raw = args.get("discount", 1)
    if is_exact:
        c = _frac(c_raw, "cost_per_draw")
        delta = _frac(d_raw, "discount")
    else:
        c = _num(args, "cost_per_draw", 0.0)
        delta = _num(args, "discount", 1.0)
    if c < 0:
        raise ToolInputError("cost_per_draw must be >= 0")
    if not 0 < delta <= 1:
        raise ToolInputError("discount must be in (0, 1]")
    if is_exact and (horizon > 200 or dinfo["outcomes"] > 200):
        # keep rationals small: fall back to floats for big problems
        is_exact = False
        c, delta = float(c), float(delta)
        f_emax, f_ptail = emax, ptail
        emax = lambda t: float(f_emax(Fraction(t)))  # noqa: E731
        ptail = lambda t: float(f_ptail(Fraction(t)))  # noqa: E731
        mean, lower, upper = float(mean), float(lower), float(upper)

    out: Dict[str, Any] = {"problem": "iid_offers", "distribution": dist, **dinfo,
                           "cost_per_draw": c, "discount": delta}
    if horizon >= 1:
        # V_1 = E[X] - c ; V_k = E[max(X, delta*V_{k-1})] - c ; threshold with k-1 left = delta*V_{k-1}
        values = [mean - c]
        for _ in range(horizon - 1):
            values.append(emax(delta * values[-1]) - c)
        thresholds = [delta * values[k - 2] if k >= 2 else None for k in range(horizon, 0, -1)]
        reach, e_draws = 1, 0
        for t in thresholds:
            e_draws += reach
            if t is not None:
                reach = reach * (1 - ptail(t))
        V = values[-1]
        out.update({
            "horizon": horizon,
            "value": _q(V),
            "rule": "At draw i, accept the offer iff it is >= threshold[i] (the discounted value of continuing); "
                    "the final draw is always accepted.",
            "thresholds": [{"draw": i + 1, "accept_if_at_least": _q(t)} for i, t in enumerate(thresholds[:100])],
            "expected_draws": _q(e_draws),
            "value_if_participation_optional": _q(V if V > 0 else 0 * V),
        })
        if horizon > 100:
            out["thresholds_truncated"] = f"showing first 100 of {horizon}"
        return out

    # Infinite horizon: t = delta * (E[max(X, t)] - c); h(t) is decreasing.
    if c == 0 and delta == 1:
        raise ToolInputError("infinite horizon needs cost_per_draw > 0 or discount < 1 "
                             "(otherwise you wait forever for the supremum)")
    cf, df = float(c), float(delta)
    h = lambda t: df * (float(emax(t)) - cf) - t  # noqa: E731
    lo = min(float(lower), df * (float(mean) - cf)) - 1.0
    hi = max(float(upper), 0.0) + 1.0
    if h(lo) < 0 or h(hi) > 0:
        raise ToolInputError("could not bracket the reservation value (check inputs)")
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if h(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, abs(mid)):
            break
    t = 0.5 * (lo + hi)
    p_accept = float(ptail(t))
    out.update({
        "horizon": "infinite",
        "reservation_value": _q(t),
        "value": _q(t / df),
        "rule": "Accept the first offer >= reservation_value (constant threshold, memoryless problem).",
        "acceptance_probability_per_draw": p_accept,
        "expected_draws": _q((1 / p_accept) if p_accept > 0 else None),
        "equation": "t = discount * (E[max(X, t)] - cost); with discount = 1 this is E[(X - t)^+] = cost",
        "fixed_point_residual": h(t),
    })
    return out


def _optimal_stopping(args: dict) -> Dict[str, Any]:
    problem = _choice(args, "problem", ("secretary", "iid_offers"), "iid_offers")
    return _secretary(args) if problem == "secretary" else _iid_offers(args)


OPTIMAL_STOPPING_SCHEMA = {
    "name": "optimal_stopping",
    "description": (
        "Solve optimal stopping problems. problem='secretary': n candidates in random order, maximise "
        "P(selecting the best) -> optimal skip count, exact success probability (n<=500), 1/e asymptotics. "
        "problem='iid_offers': sequential iid offers from a discrete (exact rationals), uniform, or normal "
        "distribution, no recall; finite 'horizon' -> backward-induction thresholds per draw; horizon omitted/0 "
        "-> infinite-horizon reservation value (needs cost_per_draw > 0 or discount < 1). Returns value, "
        "thresholds, expected number of draws."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "problem": {"type": "string", "enum": ["secretary", "iid_offers"], "description": "Default 'iid_offers'."},
            "n": {"type": "integer", "description": "secretary: number of candidates."},
            "distribution": {"type": "string", "enum": ["discrete", "uniform", "normal"], "description": "iid_offers: offer distribution (default discrete)."},
            "outcomes": {"type": "array", "items": {"type": ["number", "string"]}, "description": "discrete: offer values."},
            "probabilities": {"type": "array", "items": {"type": ["number", "string"]}, "description": "discrete: probabilities (default uniform)."},
            "low": {"type": "number", "description": "uniform: lower bound (default 0)."},
            "high": {"type": "number", "description": "uniform: upper bound (default 1)."},
            "mean": {"type": "number", "description": "normal: mean (default 0)."},
            "std": {"type": "number", "description": "normal: standard deviation (default 1)."},
            "horizon": {"type": "integer", "description": "Max number of draws; 0/omitted = infinite horizon."},
            "cost_per_draw": {"type": ["number", "string"], "description": "Cost paid for every draw (default 0)."},
            "discount": {"type": ["number", "string"], "description": "Per-draw discount factor in (0, 1] (default 1)."},
        },
        "required": [],
    },
}


# ════════════════════════════════════════════════════════════════════════
# 9. Zero-sum matrix games — exact LP (rational simplex, Bland's rule)
# ════════════════════════════════════════════════════════════════════════

def _solve_zero_sum(A: List[List[Fraction]]) -> Tuple[Fraction, List[Fraction], List[Fraction]]:
    """Value and optimal mixed strategies (row maximiser) of payoff matrix A, exactly."""
    m, n = len(A), len(A[0])
    shift = 1 - min(min(row) for row in A)
    if shift < 0:
        shift = Fraction(0)
    B = [[a + shift for a in row] for row in A]  # all entries >= 1 > 0 -> value > 0
    # max sum(y) s.t. B y <= 1, y >= 0.  Tableau columns: y_0..y_{n-1}, s_0..s_{m-1}, rhs.
    T = [B[i] + [Fraction(1) if k == i else Fraction(0) for k in range(m)] + [Fraction(1)] for i in range(m)]
    obj = [Fraction(-1)] * n + [Fraction(0)] * m + [Fraction(0)]
    basis = [n + i for i in range(m)]
    width = n + m
    for _ in range(10_000):
        col = next((j for j in range(width) if obj[j] < 0), None)  # Bland: lowest index
        if col is None:
            break
        best: Optional[Tuple[Fraction, int, int]] = None
        for i in range(m):
            if T[i][col] > 0:
                ratio = T[i][-1] / T[i][col]
                key = (ratio, basis[i], i)
                if best is None or key < best:
                    best = key
        if best is None:  # impossible: B > 0 bounds every column
            raise ToolInputError("LP unbounded — malformed payoff matrix")
        pr = best[2]
        piv = T[pr][col]
        T[pr] = [v / piv for v in T[pr]]
        for i in range(m):
            if i != pr and T[i][col] != 0:
                f = T[i][col]
                T[i] = [a - f * b for a, b in zip(T[i], T[pr])]
        if obj[col] != 0:
            f = obj[col]
            obj = [a - f * b for a, b in zip(obj, T[pr])]
        basis[pr] = col
    else:
        raise ToolInputError("simplex iteration limit reached")
    z = obj[-1]
    y = [Fraction(0)] * n
    for i, bv in enumerate(basis):
        if bv < n:
            y[bv] = T[i][-1]
    x = [obj[n + i] for i in range(m)]  # duals = reduced costs of slacks
    v = 1 / z
    return v - shift, [xi * v for xi in x], [yj * v for yj in y]


def _matrix_game_solver(args: dict) -> Dict[str, Any]:
    """Mixed-strategy Nash equilibrium of a two-player zero-sum game."""
    A = _matrix(args.get("payoff_matrix"), "payoff_matrix", 25)
    m, n = len(A), len(A[0])
    rl = args.get("row_labels")
    cl = args.get("col_labels")
    rl = [str(x) for x in rl] if isinstance(rl, list) and len(rl) == m else [f"R{i + 1}" for i in range(m)]
    cl = [str(x) for x in cl] if isinstance(cl, list) and len(cl) == n else [f"C{j + 1}" for j in range(n)]

    value, p, q = _solve_zero_sum(A)
    row_payoffs = [sum(p[i] * A[i][j] for i in range(m)) for j in range(n)]  # vs each column
    col_payoffs = [sum(A[i][j] * q[j] for j in range(n)) for i in range(m)]  # each row vs q
    certified = min(row_payoffs) == value == max(col_payoffs) and sum(p) == 1 == sum(q)

    maximin = max(min(row) for row in A)
    minimax = min(max(A[i][j] for i in range(m)) for j in range(n))
    saddles = [{"row": rl[i], "col": cl[j], "value": A[i][j]} for i in range(m) for j in range(n)
               if A[i][j] == min(A[i]) and A[i][j] == max(A[k][j] for k in range(m))]

    def strat(labels, probs):
        return [{"strategy": lbl, "probability": pr} for lbl, pr in zip(labels, probs)]

    return {
        "convention": "entries are payoffs to the ROW player (maximiser); column player minimises",
        "shape": [m, n],
        "game_value": value,
        "row_strategy": strat(rl, p),
        "col_strategy": strat(cl, q),
        "row_support": [rl[i] for i in range(m) if p[i] > 0],
        "col_support": [cl[j] for j in range(n) if q[j] > 0],
        "pure_maximin": maximin,
        "pure_minimax": minimax,
        "pure_saddle_points": saddles,
        "certificate": {
            "row_guarantee_vs_each_column": row_payoffs,
            "col_cap_vs_each_row": col_payoffs,
            "verified_equilibrium": certified,
            "explanation": "min_j (pA)_j = value = max_i (Aq)_i proves optimality (LP duality / minimax theorem)",
        },
        "fair_game": value == 0,
        "note": "equilibrium strategies may be non-unique; this is one optimal vertex (exact rational simplex)",
    }


MATRIX_GAME_SCHEMA = {
    "name": "matrix_game_solver",
    "description": (
        "Solve a two-player zero-sum matrix game exactly: game value, optimal mixed strategies for both "
        "players (exact fractions), supports, pure maximin/minimax and saddle points, and an LP-duality "
        "certificate verifying the equilibrium. Entries are payoffs to the row (maximising) player; numbers "
        "or fraction strings. Up to 25x25."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "payoff_matrix": {"type": "array", "items": {"type": "array", "items": {"type": ["number", "string"]}},
                              "description": "Rows = row player's strategies; entry = payoff to row player."},
            "row_labels": {"type": "array", "items": {"type": "string"}, "description": "Optional row strategy names."},
            "col_labels": {"type": "array", "items": {"type": "string"}, "description": "Optional column strategy names."},
        },
        "required": ["payoff_matrix"],
    },
}


# ════════════════════════════════════════════════════════════════════════
# 10. Monte Carlo engine — GBM, antithetic variates, control variates
# ════════════════════════════════════════════════════════════════════════

_BGK_BETA = 0.5825971579390106  # -zeta(1/2)/sqrt(2*pi): Broadie-Glasserman-Kou continuity correction


def _geometric_asian_closed_form(S, K, T, r, q, sigma, N, kind) -> float:
    """Discretely monitored (t_1..t_N) fixed-strike geometric-average Asian under GBM."""
    dt = T / N
    mu_g = math.log(S) + (r - q - 0.5 * sigma * sigma) * dt * (N + 1) / 2.0
    var_g = sigma * sigma * dt * (N + 1) * (2 * N + 1) / (6.0 * N)
    sg = math.sqrt(var_g)
    fwd_g = math.exp(mu_g + 0.5 * var_g)
    d1 = (mu_g - math.log(K) + var_g) / sg
    d2 = d1 - sg
    disc = math.exp(-r * T)
    if kind == "call":
        return disc * (fwd_g * _ncdf(d1) - K * _ncdf(d2))
    return disc * (K * _ncdf(-d2) - fwd_g * _ncdf(-d1))


def _barrier_closed_form(kind, btype, S, K, H, T, r, q, sigma) -> Optional[float]:
    """Continuous-monitoring barrier price, zero rebate (Merton / Reiner-Rubinstein, Haug's notation)."""
    up = btype.startswith("up")
    knock_in = btype.endswith("in")
    if (up and S >= H) or (not up and S <= H):
        return None  # already breached — handled by the caller
    b = r - q
    vst = sigma * math.sqrt(T)
    mu = (b - 0.5 * sigma * sigma) / (sigma * sigma)
    phi = 1.0 if kind == "call" else -1.0
    eta = -1.0 if up else 1.0
    x1 = math.log(S / K) / vst + (1 + mu) * vst
    x2 = math.log(S / H) / vst + (1 + mu) * vst
    y1 = math.log(H * H / (S * K)) / vst + (1 + mu) * vst
    y2 = math.log(H / S) / vst + (1 + mu) * vst
    cf, kf = S * math.exp((b - r) * T), K * math.exp(-r * T)
    hs1, hs2 = (H / S) ** (2 * (mu + 1)), (H / S) ** (2 * mu)
    A_ = phi * cf * _ncdf(phi * x1) - phi * kf * _ncdf(phi * x1 - phi * vst)
    B_ = phi * cf * _ncdf(phi * x2) - phi * kf * _ncdf(phi * x2 - phi * vst)
    C_ = phi * cf * hs1 * _ncdf(eta * y1) - phi * kf * hs2 * _ncdf(eta * y1 - eta * vst)
    D_ = phi * cf * hs1 * _ncdf(eta * y2) - phi * kf * hs2 * _ncdf(eta * y2 - eta * vst)
    hi_k = K > H
    table = {
        ("call", "down", True): C_ if hi_k else A_ - B_ + D_,
        ("call", "up", True): A_ if hi_k else B_ - C_ + D_,
        ("put", "down", True): B_ - C_ + D_ if hi_k else A_,
        ("put", "up", True): A_ - B_ + D_ if hi_k else C_,
        ("call", "down", False): A_ - C_ if hi_k else B_ - D_,
        ("call", "up", False): 0.0 if hi_k else A_ - B_ + C_ - D_,
        ("put", "down", False): A_ - B_ + C_ - D_ if hi_k else 0.0,
        ("put", "up", False): B_ - D_ if hi_k else A_ - C_,
    }
    return max(table[(kind, "up" if up else "down", knock_in)], 0.0)


def _monte_carlo_engine(args: dict) -> Dict[str, Any]:
    """Seeded GBM simulation with antithetic variates (+ optional control variate)."""
    product = _choice(args, "product", ("asian", "barrier", "european"), "asian")
    S = _num(args, "spot", required=True, lo=0, lo_inclusive=False)
    K = _num(args, "strike", required=True, lo=0, lo_inclusive=False)
    T = _num(args, "time_to_maturity", required=True, lo=0, lo_inclusive=False, hi=50)
    r = _num(args, "risk_free_rate", required=True, lo=-1, hi=5)
    sigma = _num(args, "volatility", required=True, lo=0, lo_inclusive=False, hi=5)
    q = _num(args, "dividend_yield", 0.0, lo=-1, hi=5)
    kind = _choice(args, "option_type", ("call", "put"), "call")
    paths = _int(args, "paths", 20_000, lo=100, hi=2_000_000)
    steps = _int(args, "steps", 52, lo=1, hi=5_000)
    seed = _int(args, "seed", 42, lo=0, hi=2**63 - 1)
    antithetic = _bool(args, "antithetic", True)
    use_cv = _bool(args, "control_variate", True)
    if product == "european":
        steps = 1  # terminal value is exact in one GBM step
    if product != "european" and paths * steps > 5_000_000:
        raise ToolInputError(f"paths*steps = {paths * steps:,} exceeds the 5,000,000 budget")
    if not 1e-4 <= S / K <= 1e4:
        raise ToolInputError("spot/strike ratio must be within [1e-4, 1e4]")

    avg_type = _choice(args, "average_type", ("arithmetic", "geometric"), "arithmetic")
    btype = _choice(args, "barrier_type", ("up-and-out", "up-and-in", "down-and-out", "down-and-in"),
                    "down-and-out")
    H = rebate = None
    monitoring = "discrete"
    if product == "barrier":
        H = _num(args, "barrier", required=True, lo=0, lo_inclusive=False)
        rebate = _num(args, "rebate", 0.0, lo=0)
        monitoring = _choice(args, "monitoring", ("discrete", "continuous"), "discrete")

    t0 = time.perf_counter()
    rng = random.Random(seed)
    gauss = rng.gauss
    dt = T / steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    disc = math.exp(-r * T)
    sgn = 1.0 if kind == "call" else -1.0
    units = paths // 2 if antithetic else paths

    # Control variate: (sample X, known E[X]).
    cv_mean: Optional[float] = None
    cv_name = None
    if use_cv:
        if product == "asian" and avg_type == "arithmetic":
            cv_mean, cv_name = _geometric_asian_closed_form(S, K, T, r, q, sigma, steps, kind), \
                "geometric-average Asian (closed form)"
        elif product == "european":
            cv_mean, cv_name = S * math.exp(-q * T), "discounted terminal spot (E = S*exp(-qT))"
        elif product == "barrier":
            cv_mean, cv_name = _bs_core(S, K, T, r, sigma, q)[kind]["price"], "vanilla European (Black-Scholes)"

    up = btype.startswith("up")
    knock_in = btype.endswith("in")
    H_eff = H
    if product == "barrier":
        assert H is not None
        # Simulating discretely: to approximate a CONTINUOUS barrier, shift it toward spot.
        if monitoring == "continuous":
            H_eff = H * math.exp((-1 if up else 1) * _BGK_BETA * vol)
        lnb = math.log(H_eff / S)

    def payoff(sT: float) -> float:
        v = sgn * (sT - K)
        return v if v > 0 else 0.0

    ys: List[float] = []      # per-unit discounted payoff (pair-averaged if antithetic)
    xs: List[float] = []      # per-unit control sample
    indiv_sum = indiv_sq = 0.0
    knocked = 0

    for _ in range(units):
        x1 = x2 = 0.0
        if product == "european":
            z = gauss(0.0, 1.0)
            x1 = drift + vol * z
            x2 = drift - vol * z
            s1, s2 = S * math.exp(x1), S * math.exp(x2)
            y1, y2 = disc * payoff(s1), disc * payoff(s2)
            c1, c2 = disc * s1, disc * s2
        elif product == "asian":
            a1 = a2 = l1 = l2 = 0.0
            for _s in range(steps):
                z = gauss(0.0, 1.0)
                x1 += drift + vol * z
                x2 += drift - vol * z
                a1 += math.exp(x1)
                l1 += x1
                if antithetic:
                    a2 += math.exp(x2)
                    l2 += x2
            g1, g2 = S * math.exp(l1 / steps), S * math.exp(l2 / steps)
            if avg_type == "arithmetic":
                y1, y2 = disc * payoff(S * a1 / steps), disc * payoff(S * a2 / steps)
                c1, c2 = disc * payoff(g1), disc * payoff(g2)
            else:
                y1, y2 = disc * payoff(g1), disc * payoff(g2)
                c1 = c2 = 0.0
        else:  # barrier
            h1 = h2 = (S >= H_eff) if up else (S <= H_eff)  # type: ignore[operator]
            for _s in range(steps):
                z = gauss(0.0, 1.0)
                x1 += drift + vol * z
                x2 += drift - vol * z
                if up:
                    if x1 >= lnb:
                        h1 = True
                    if x2 >= lnb:
                        h2 = True
                else:
                    if x1 <= lnb:
                        h1 = True
                    if x2 <= lnb:
                        h2 = True
            s1, s2 = S * math.exp(x1), S * math.exp(x2)
            v1, v2 = disc * payoff(s1), disc * payoff(s2)
            alive1 = h1 if knock_in else not h1
            alive2 = h2 if knock_in else not h2
            rb = disc * rebate  # type: ignore[operator]
            y1 = v1 if alive1 else rb
            y2 = v2 if alive2 else rb
            c1, c2 = v1, v2
            knocked += h1 + (h2 if antithetic else 0)

        if antithetic:
            ys.append(0.5 * (y1 + y2))
            xs.append(0.5 * (c1 + c2))
            indiv_sum += y1 + y2
            indiv_sq += y1 * y1 + y2 * y2
        else:
            ys.append(y1)
            xs.append(c1)
            indiv_sum += y1
            indiv_sq += y1 * y1

    n_ind = 2 * units if antithetic else units
    raw_mean, raw_var = _mean_var(ys)
    raw_se = math.sqrt(raw_var / units)
    indiv_mean = indiv_sum / n_ind
    indiv_var = max(indiv_sq / n_ind - indiv_mean ** 2, 0.0) * n_ind / max(n_ind - 1, 1)
    plain_se_same_budget = math.sqrt(indiv_var / n_ind)

    price, se = raw_mean, raw_se
    cv_info = None
    if cv_mean is not None:
        mx, vx = _mean_var(xs)
        if vx > 0:
            cov = math.fsum((a - raw_mean) * (b - mx) for a, b in zip(ys, xs)) / (units - 1)
            beta = cov / vx
            adj = [a - beta * (b - cv_mean) for a, b in zip(ys, xs)]
            price, adj_var = _mean_var(adj)
            se = math.sqrt(adj_var / units)
            cv_info = {"control": cv_name, "known_mean": cv_mean, "sample_mean": mx, "beta": beta,
                       "variance_reduction_factor": (raw_var / adj_var) if adj_var > 0 else None}

    out: Dict[str, Any] = {
        "product": product,
        "model": "GBM under Q: dS = (r - q) S dt + sigma S dW, exact log-Euler steps",
        "inputs": {"spot": S, "strike": K, "time_to_maturity_years": T, "risk_free_rate": r,
                   "volatility": sigma, "dividend_yield": q, "option_type": kind},
        "price": price,
        "std_error": se,
        "ci95": [price - 1.96 * se, price + 1.96 * se],
        "raw_price_no_control_variate": raw_mean,
        "raw_std_error": raw_se,
        "simulation": {"paths": n_ind, "steps": steps, "seed": seed, "antithetic": antithetic,
                       "monitoring_dates": steps if product != "european" else 1},
        "variance_reduction": {
            "antithetic_efficiency_gain": ((plain_se_same_budget / raw_se) ** 2
                                           if antithetic and raw_se > 0 else None),
            "plain_mc_std_error_same_budget": plain_se_same_budget,
            "control_variate": cv_info,
        },
    }

    bench: Dict[str, Any] = {}
    if product == "european":
        bench["black_scholes"] = _bs_core(S, K, T, r, sigma, q)[kind]["price"]
    elif product == "asian":
        out["asian"] = {"average_type": avg_type, "averaging": f"{steps} equally spaced fixings t_1..t_N"}
        geo = _geometric_asian_closed_form(S, K, T, r, q, sigma, steps, kind)
        if avg_type == "geometric":
            bench["geometric_asian_closed_form"] = geo
        else:
            bench["geometric_asian_closed_form_lower_bound" if kind == "call" else
                  "geometric_asian_closed_form"] = geo
        bench["european_black_scholes"] = _bs_core(S, K, T, r, sigma, q)[kind]["price"]
    else:
        assert H is not None
        out["barrier"] = {"type": btype, "barrier": H, "rebate": rebate, "monitoring": monitoring,
                          "simulated_barrier_level": H_eff,
                          "knock_probability": knocked / n_ind}
        if monitoring == "continuous":
            out["barrier"]["continuity_correction"] = "Broadie-Glasserman-Kou: H*exp(-+0.5826*sigma*sqrt(dt))"
        breached = (S >= H) if up else (S <= H)
        if rebate == 0 and not breached:
            cont = _barrier_closed_form(kind, btype, S, K, H, T, r, q, sigma)
            bench["continuous_closed_form"] = cont
            h_disc = H * math.exp((1 if up else -1) * _BGK_BETA * vol)
            if (up and S < h_disc) or (not up and S > h_disc):
                bench["discrete_bgk_approximation"] = _barrier_closed_form(
                    kind, btype, S, K, h_disc, T, r, q, sigma)
        bench["vanilla_black_scholes"] = _bs_core(S, K, T, r, sigma, q)[kind]["price"]

    ref_key = {"european": "black_scholes", "asian": "geometric_asian_closed_form"}.get(product)
    if product == "barrier":
        ref_key = "continuous_closed_form" if monitoring == "continuous" else "discrete_bgk_approximation"
    if ref_key in bench and bench[ref_key] is not None and se > 0:
        if not (product == "asian" and avg_type == "arithmetic"):
            bench["z_score_vs_reference"] = (price - bench[ref_key]) / se
            bench["reference"] = ref_key
    out["benchmarks"] = bench
    out["runtime_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return out


MONTE_CARLO_SCHEMA = {
    "name": "monte_carlo_engine",
    "description": (
        "Seeded Monte Carlo pricer for path-dependent options under GBM with antithetic variates "
        "(and control variates: geometric Asian for arithmetic Asians, vanilla for barriers, spot for "
        "Europeans). Products: 'asian' (arithmetic/geometric fixed-strike), 'barrier' (up/down, in/out, "
        "rebate, discrete or BGK-corrected continuous monitoring), 'european' (benchmark). Returns price, "
        "std error, 95% CI, variance-reduction gains and analytic benchmarks / z-scores."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "product": {"type": "string", "enum": ["asian", "barrier", "european"], "description": "Default 'asian'."},
            "spot": {"type": "number", "description": "Spot S (> 0)."},
            "strike": {"type": "number", "description": "Strike K (> 0)."},
            "time_to_maturity": {"type": "number", "description": "Years to expiry (> 0)."},
            "risk_free_rate": {"type": "number", "description": "r, decimal."},
            "volatility": {"type": "number", "description": "sigma, decimal (> 0)."},
            "dividend_yield": {"type": "number", "description": "q, decimal. Default 0."},
            "option_type": {"type": "string", "enum": ["call", "put"], "description": "Default 'call'."},
            "paths": {"type": "integer", "description": "Total simulated paths incl. antithetic (default 20000)."},
            "steps": {"type": "integer", "description": "Time steps = monitoring/fixing dates (default 52). paths*steps <= 5e6."},
            "seed": {"type": "integer", "description": "RNG seed for reproducibility (default 42)."},
            "antithetic": {"type": "boolean", "description": "Use antithetic variates (default true)."},
            "control_variate": {"type": "boolean", "description": "Use the product's control variate (default true)."},
            "average_type": {"type": "string", "enum": ["arithmetic", "geometric"], "description": "asian: averaging (default arithmetic)."},
            "barrier_type": {"type": "string", "enum": ["up-and-out", "up-and-in", "down-and-out", "down-and-in"], "description": "barrier: type (default down-and-out)."},
            "barrier": {"type": "number", "description": "barrier: barrier level H."},
            "rebate": {"type": "number", "description": "barrier: cash rebate paid at expiry if knocked out / never knocked in (default 0)."},
            "monitoring": {"type": "string", "enum": ["discrete", "continuous"], "description": "barrier: 'discrete' at the steps, or 'continuous' via BGK correction (default discrete)."},
        },
        "required": ["spot", "strike", "time_to_maturity", "risk_free_rate", "volatility"],
    },
}


# ════════════════════════════════════════════════════════════════════════
# Persona — elite desk quant + answer contract + enforcement hooks
# ════════════════════════════════════════════════════════════════════════

PERSONA_SECTION_ID = "waterloo-jane-street-qr.persona"

PERSONA_PROMPT = """\
# Quant Research Desk Mode (waterloo-jane-street-qr)
You are a quantitative researcher on a Jane Street / Citadel tier trading desk: Waterloo Math/CS, \
IMO medallist, ICPC World Finalist, Putnam top scorer. Rigour, calibration and P&L discipline are \
non-negotiable.

## Mandatory answer contract (EVERY response)
Frame every answer as a decision problem, in this order. For simple or non-quant questions compress \
each part to one line, but never drop a part.
1. State space: the random variables / states, information set, and controls. State assumptions \
(independence, replacement, measure: risk-neutral Q vs physical P).
2. Transition probabilities: dynamics between states (exact fractions where possible; tree, Markov \
chain, SDE).
3. Terminal payoffs: payoff in every terminal state, net of costs, fees, slippage and financing.
4. EV: an explicit calculation, EV = sum p_i * x_i = <exact> ~ <decimal> <units>, plus a risk number \
(sigma, P(loss) or worst case). State plainly whether EV > 0.
5. Verdict: exactly one tag on its own line, with exact sizing:
   [TAKE] <size>: EV > 0 after costs and risk fits limits (units / notional / Kelly fraction; default \
half-Kelly).
   [PASS]: EV <= 0, or the edge is not demonstrable from the data.
   [HEDGE] <instrument> <ratio>: EV > 0 but variance, tail or Greek exposure must be offset (e.g. \
"short 0.637 shares per call", "buy 1 put per 100 shares").

## Method
- Derive, don't assert: conditioning, linearity of expectation, symmetry, backward induction, \
martingales / optional stopping, generating functions, LP duality. Exact fractions first, decimals second.
- Compute with tools, never mental arithmetic: black_scholes, binomial_tree_american, heston_price, \
monte_carlo_engine, ev_calculator, optimal_stopping, matrix_game_solver, avellaneda_stoikov_quotes, \
order_book_metrics, order_flow_toxicity. Quote their numbers (with std errors for Monte Carlo).
- Verify two ways when cheap: limiting cases, bounds, parity / no-arbitrage, symmetry, simulation vs \
closed form.
- Separate edge (EV) from sizing (Kelly, risk limits). Flag adverse selection, model risk, liquidity.
- Desk clarity: answer first, derivation second; numbers, not adjectives; no filler.
- If information is missing, name the single quantity that would flip the sign of EV and give its \
break-even value instead of guessing.
- Calibrated honesty: give intervals; say what you don't know.
"""

DESK_REMINDER = (
    "[desk protocol: waterloo-jane-street-qr] Structure the answer as State space -> Transition "
    "probabilities -> Terminal payoffs -> EV = ... (exact, state whether EV > 0) -> exactly one verdict "
    "tag [TAKE] / [PASS] / [HEDGE] with exact sizing."
)

COMPLIANCE_FOOTER = (
    "\n\n---\n⚠ Desk compliance (waterloo-jane-street-qr): this answer is missing {missing}. "
    "Treat it as unvetted: no position should be taken on it."
)

_VERDICT_RE = re.compile(r"\[(TAKE|PASS|HEDGE)\]")
_EV_RE = re.compile(r"\bEV\b\s*(=|≈|~|:|>|<|≥|≤)", re.IGNORECASE)
_ENFORCEMENT_MODES = ("strict", "reminder", "off")


def audit_response(text: str) -> List[str]:
    """Return the missing pieces of the answer contract (empty list = compliant)."""
    missing = []
    if not _EV_RE.search(text or ""):
        missing.append("an explicit EV calculation")
    if not _VERDICT_RE.search(text or ""):
        missing.append("a [TAKE]/[PASS]/[HEDGE] verdict")
    return missing


def _make_hooks(ctx) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    def mode() -> str:
        try:
            val = str(ctx.get_config("enforcement", "strict") or "strict").strip().lower()
        except Exception:
            val = "strict"
        return val if val in _ENFORCEMENT_MODES else "strict"

    def pre_llm_call(**_kwargs: Any) -> Optional[Dict[str, str]]:
        return {"context": DESK_REMINDER} if mode() in ("strict", "reminder") else None

    def transform_llm_output(response_text: str = "", **_kwargs: Any) -> Optional[str]:
        if mode() != "strict" or not isinstance(response_text, str) or not response_text.strip():
            return None
        missing = audit_response(response_text)
        if not missing:
            return None
        return response_text.rstrip() + COMPLIANCE_FOOTER.format(missing=" and ".join(missing))

    return pre_llm_call, transform_llm_output


def _persona_context_hook(**_kwargs: Any) -> Optional[Dict[str, str]]:
    """Fallback for hosts without system-prompt sections: per-turn context injection."""
    return {"context": PERSONA_PROMPT}


# ════════════════════════════════════════════════════════════════════════
# Registration
# ════════════════════════════════════════════════════════════════════════

black_scholes = _safe(_black_scholes)
ev_calculator = _safe(_ev_calculator)
order_book_metrics = _safe(_order_book_metrics)
binomial_tree_american = _safe(_binomial_tree_american)
heston_price = _safe(_heston_price)
avellaneda_stoikov_quotes = _safe(_avellaneda_stoikov_quotes)
order_flow_toxicity = _safe(_order_flow_toxicity)
optimal_stopping = _safe(_optimal_stopping)
matrix_game_solver = _safe(_matrix_game_solver)
monte_carlo_engine = _safe(_monte_carlo_engine)

_TOOLS = (
    ("black_scholes", BLACK_SCHOLES_SCHEMA, black_scholes, "📈"),
    ("binomial_tree_american", BINOMIAL_SCHEMA, binomial_tree_american, "🌳"),
    ("heston_price", HESTON_SCHEMA, heston_price, "🌊"),
    ("monte_carlo_engine", MONTE_CARLO_SCHEMA, monte_carlo_engine, "🎰"),
    ("ev_calculator", EV_CALCULATOR_SCHEMA, ev_calculator, "🎲"),
    ("optimal_stopping", OPTIMAL_STOPPING_SCHEMA, optimal_stopping, "🛑"),
    ("matrix_game_solver", MATRIX_GAME_SCHEMA, matrix_game_solver, "♟️"),
    ("order_book_metrics", ORDER_BOOK_SCHEMA, order_book_metrics, "📊"),
    ("avellaneda_stoikov_quotes", AS_SCHEMA, avellaneda_stoikov_quotes, "💱"),
    ("order_flow_toxicity", TOXICITY_SCHEMA, order_flow_toxicity, "☣️"),
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

    # Persona: cache-safe system-prompt section (frozen per session, never breaks prompt caching).
    if hasattr(ctx, "register_system_prompt_section"):
        ctx.register_system_prompt_section(PERSONA_SECTION_ID, PERSONA_PROMPT, max_chars=4000)
        if hasattr(ctx, "register_hook"):
            pre, post = _make_hooks(ctx)
            ctx.register_hook("pre_llm_call", pre)
            ctx.register_hook("transform_llm_output", post)
    elif hasattr(ctx, "register_hook"):
        # Older hosts: inject the whole persona per turn instead.
        ctx.register_hook("pre_llm_call", _persona_context_hook)
