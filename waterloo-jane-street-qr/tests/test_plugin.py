"""Test suite for waterloo-jane-street-qr.

Runs under pytest (`pytest tests/`) or standalone (`python3 tests/test_plugin.py`).
Every numeric check is against a closed form, a published reference value, an exact
identity (parity, duality, in-out), or an independent simulation.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import random
import re
import sys
import time
import itertools  # noqa: F401  (used by brute-force checks)
import traceback
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("wjsqr", ROOT / "__init__.py")
assert _spec is not None and _spec.loader is not None
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


# ── helpers ────────────────────────────────────────────────────────────────

def _no_nan(token):
    raise ValueError(f"non-strict JSON constant {token}")


def call(fn, **args):
    """Invoke a tool handler; assert strict JSON and success."""
    raw = fn(args)
    assert isinstance(raw, str)
    out = json.loads(raw, parse_constant=_no_nan)
    assert out.get("ok") is True, f"{fn.__name__} failed: {out.get('error')}"
    return out


def call_err(fn, args):
    raw = fn(args)
    out = json.loads(raw, parse_constant=_no_nan)
    assert out["ok"] is False and isinstance(out["error"], str) and out["error"], out
    return out["error"]


def approx(got, want, tol=1e-9, rel=0.0):
    assert got is not None, f"got None, want {want}"
    assert abs(got - want) <= max(tol, rel * abs(want)), f"got {got!r}, want {want!r} (tol {tol}, rel {rel})"


def exact(field, want: str):
    assert field["exact"] == want, f"got {field}, want {want}"


BASE = dict(spot=100, strike=100, time_to_maturity=1, risk_free_rate=0.05, volatility=0.2)

try:  # pytest is optional; the standalone runner understands Skip too
    import pytest as _pytest
    Skip = _pytest.skip.Exception
except ImportError:  # pragma: no cover
    class Skip(Exception):
        pass


def need_numpy():
    if m._np is None:
        raise Skip("NumPy not installed (pure-Python backend tested instead)")


# ════════════════════════════════════════════════════════════════════════
# Registration, manifest, persona & enforcement
# ════════════════════════════════════════════════════════════════════════

class FakeCtx:
    def __init__(self, config=None):
        self.tools, self.hooks, self.sections, self.config = {}, {}, {}, config or {}

    def register_tool(self, name, toolset, schema, handler, **kw):
        assert name not in self.tools
        self.tools[name] = dict(toolset=toolset, schema=schema, handler=handler, **kw)

    def register_hook(self, name, cb, **_kw):
        self.hooks.setdefault(name, []).append(cb)

    def register_system_prompt_section(self, sid, content, position="after_memory", max_chars=4000):
        assert len(content) <= max_chars <= 4000
        self.sections[sid] = content

    def get_config(self, key, default=None):
        return self.config.get(key, default)


EXPECTED_TOOLS = {
    "black_scholes", "binomial_tree_american", "heston_price", "monte_carlo_engine", "ev_calculator",
    "optimal_stopping", "matrix_game_solver", "order_book_metrics", "avellaneda_stoikov_quotes",
    "order_flow_toxicity", "almgren_chriss_liquidation", "longstaff_schwartz_american",
    "black_litterman_optimizer", "jane_street_ev_trainer",
}


def test_register_all_tools_and_schemas():
    ctx = FakeCtx()
    m.register(ctx)
    assert set(ctx.tools) == EXPECTED_TOOLS
    for name, t in ctx.tools.items():
        s = t["schema"]
        assert s["name"] == name and t["toolset"] == m.TOOLSET
        assert len(s["description"]) > 40
        props = s["parameters"]["properties"]
        assert s["parameters"]["type"] == "object"
        assert set(s["parameters"].get("required", [])) <= set(props), name
        for pname, p in props.items():
            assert "description" in p and ("type" in p), f"{name}.{pname}"
        json.dumps(s)  # serialisable
    assert m.PERSONA_SECTION_ID in ctx.sections
    assert set(ctx.hooks) == {"pre_llm_call", "transform_llm_output"}


def test_manifest_matches_registrations():
    text = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
    assert re.search(r'^name:\s*waterloo-jane-street-qr\s*$', text, re.M)
    assert re.search(r'^version:\s*"1\.0\.0"', text, re.M)
    assert re.search(r'^entrypoint:\s*"__init__\.py"', text, re.M)
    block = text.split("provides_tools:")[1].split("provides_hooks:")[0]
    assert set(re.findall(r"-\s*(\w+)", block)) == EXPECTED_TOOLS
    hooks = text.split("provides_hooks:")[1].split("config_schema:")[0]
    assert set(re.findall(r"-\s*(\w+)", hooks)) == {"pre_llm_call", "transform_llm_output"}


def test_persona_contract():
    p = m.PERSONA_PROMPT
    assert len(p) <= 4000
    for needle in ("State space", "Transition probabilities", "Terminal payoffs", "EV", "[TAKE]",
                   "[PASS]", "[HEDGE]", "sizing", "EVERY response"):
        assert needle in p, needle
    for tool in EXPECTED_TOOLS:
        assert tool in p, f"persona does not mention {tool}"


def test_enforcement_hooks_modes():
    good = "State space ... EV = 1/2 > 0\n[TAKE] 0.2 Kelly"
    bad = "The answer is 42."
    for mode, reminder, flags in (("strict", True, True), ("reminder", True, False), ("off", False, False),
                                  ("garbage", True, True)):
        ctx = FakeCtx({"enforcement": mode})
        m.register(ctx)
        pre = ctx.hooks["pre_llm_call"][0]
        post = ctx.hooks["transform_llm_output"][0]
        r = pre(session_id="s", user_message="hi")
        assert (r is not None) == reminder and (not r or "[TAKE]" in r["context"])
        assert post(response_text=good, session_id="s") is None
        out = post(response_text=bad, session_id="s")
        if flags:
            assert out.startswith(bad) and "Desk compliance" in out
            assert "EV" in out and "[TAKE]/[PASS]/[HEDGE]" in out
        else:
            assert out is None
        assert post(response_text="", session_id="s") is None


def test_audit_response():
    assert m.audit_response("EV = 3 > 0 ... [HEDGE] short 0.5 delta") == []
    assert m.audit_response("ev ≈ -0.1\n[PASS]") == []
    assert len(m.audit_response("nothing here")) == 2
    assert m.audit_response("EV = 2") == ["a [TAKE]/[PASS]/[HEDGE] verdict"]
    assert m.audit_response("[TAKE]") == ["an explicit EV calculation"]


def test_legacy_host_fallback():
    class OldCtx:
        def __init__(self):
            self.tools, self.hooks = [], []

        def register_tool(self, **kw):
            self.tools.append(kw["name"])

        def register_hook(self, name, cb):
            self.hooks.append((name, cb))

    ctx = OldCtx()
    m.register(ctx)
    assert len(ctx.tools) == len(EXPECTED_TOOLS) and ctx.hooks[0][0] == "pre_llm_call"
    assert ctx.hooks[0][1]()["context"] == m.PERSONA_PROMPT


# ════════════════════════════════════════════════════════════════════════
# Black-Scholes
# ════════════════════════════════════════════════════════════════════════

def test_black_scholes_reference_values():
    r = call(m.black_scholes, **BASE)
    approx(r["call"]["price"], 10.450584, 1e-5)
    approx(r["put"]["price"], 5.573526, 1e-5)
    approx(r["call"]["delta"], 0.636831, 1e-5)
    approx(r["put"]["delta"], -0.363169, 1e-5)
    approx(r["call"]["gamma"], 0.018762, 1e-5)
    approx(r["call"]["vega"], 37.524035, 1e-4)
    approx(r["call"]["theta"], -6.414028, 1e-4)
    approx(r["put"]["theta"], -1.657880, 1e-4)
    approx(r["put_call_parity_residual"], 0.0, 1e-10)


def test_black_scholes_greeks_vs_finite_differences():
    base = dict(spot=97, strike=103, time_to_maturity=0.37, risk_free_rate=0.03, volatility=0.31, dividend_yield=0.015)
    px = lambda **kw: call(m.black_scholes, **{**base, **kw})["put"]["price"]  # noqa: E731
    g = call(m.black_scholes, **base)["put"]
    h = 1e-4
    approx((px(spot=97 + h) - px(spot=97 - h)) / (2 * h), g["delta"], 1e-6)
    approx((px(spot=97.01) - 2 * px() + px(spot=96.99)) / 1e-4, g["gamma"], 1e-5)
    approx((px(volatility=0.31 + h) - px(volatility=0.31 - h)) / (2 * h), g["vega"], 1e-5)
    approx(-(px(time_to_maturity=0.37 + h) - px(time_to_maturity=0.37 - h)) / (2 * h), g["theta"], 1e-5)
    approx((px(risk_free_rate=0.03 + h) - px(risk_free_rate=0.03 - h)) / (2 * h), g["rho"], 1e-5)


def test_black_scholes_implied_vol_and_edges():
    iv = call(m.black_scholes, **{**BASE, "volatility": 0.5}, option_type="call", market_price=10.450584)
    approx(iv["implied_vol_solve"]["implied_volatility"], 0.2, 1e-6)
    approx(call(m.black_scholes, **{**BASE, "spot": 110, "time_to_maturity": 0})["call"]["price"], 10.0)
    bad = call(m.black_scholes, **BASE, option_type="call", market_price=150)
    assert "error" in bad["implied_vol_solve"]
    assert call(m.black_scholes, **{**BASE, "volatility": 20})["warnings"]


# ════════════════════════════════════════════════════════════════════════
# Binomial tree (American)
# ════════════════════════════════════════════════════════════════════════

def test_binomial_hull_five_step_example():
    # Hull, Options Futures & Other Derivatives: S=K=50, r=10%, sigma=40%, T=5 months, 5 steps -> 4.49
    r = call(m.binomial_tree_american, spot=50, strike=50, time_to_maturity=5 / 12, risk_free_rate=0.1,
             volatility=0.4, steps=5)
    approx(r["american_price"], 4.4885, 5e-4)
    assert r["american_price"] > r["european_price_tree"]


def test_binomial_converges_to_reference_and_black_scholes():
    r = call(m.binomial_tree_american, **BASE, option_type="put", steps=1000)
    approx(r["american_price"], 6.0904, 2e-3)                 # literature value ~6.0903-6.0905
    approx(r["american_price_control_variate"], 6.0904, 2e-3)
    approx(r["european_price_tree"], r["european_price_black_scholes"], 3e-3)
    assert r["early_exercise_premium"] > 0.4
    assert r["american_price"] >= max(r["intrinsic_value"], r["european_price_tree"])


def test_binomial_call_without_dividends_never_exercised_early():
    r = call(m.binomial_tree_american, **BASE, option_type="call", steps=400)
    approx(r["early_exercise_premium"], 0.0, 1e-10)
    assert r["early_exercise_boundary"]["points"] == []
    # With a big dividend yield early exercise becomes valuable.
    rd = call(m.binomial_tree_american, **BASE, option_type="call", dividend_yield=0.10, steps=400)
    assert rd["early_exercise_premium"] > 0.05
    pts = [p["critical_spot"] for p in rd["early_exercise_boundary"]["points"] if p["critical_spot"]]
    assert all(p >= 100 for p in pts)


def test_binomial_put_boundary_shape():
    r = call(m.binomial_tree_american, **BASE, option_type="put", steps=600, boundary_points=30)
    pts = [p for p in r["early_exercise_boundary"]["points"] if p["critical_spot"] is not None]
    crit = [p["critical_spot"] for p in pts]
    assert len(crit) >= 20 and all(c <= 100 + 1e-9 for c in crit)
    # Exercise boundary rises toward the strike as expiry approaches (tolerate lattice jitter).
    assert crit[-1] == 100 and crit[0] < crit[len(crit) // 2] < crit[-1]
    assert crit[0] > 70


def test_binomial_greeks_vs_black_scholes_scale():
    am = call(m.binomial_tree_american, **BASE, option_type="put", steps=800)["greeks"]
    bs = call(m.black_scholes, **BASE)["put"]
    assert -1 < am["delta"] < bs["delta"] < 0          # American put is more negative-delta
    approx(am["gamma"], bs["gamma"], 0.01)
    approx(am["vega"], bs["vega"], 2.0)
    assert am["rho"] < 0 and am["theta"] < 0


def test_binomial_deep_itm_put_exercise_now():
    r = call(m.binomial_tree_american, spot=50, strike=100, time_to_maturity=1, risk_free_rate=0.05,
             volatility=0.2, option_type="put", steps=200)
    assert r["exercise_now_optimal"] is True
    approx(r["american_price"], 50.0, 1e-9)


def test_binomial_input_errors():
    call_err(m.binomial_tree_american, {**BASE, "steps": 1})
    call_err(m.binomial_tree_american, {**BASE, "steps": 100000})
    call_err(m.binomial_tree_american, {**BASE, "time_to_maturity": 0})
    call_err(m.binomial_tree_american, {**BASE, "option_type": "straddle"})
    # p outside (0,1): huge rate, tiny vol, few steps
    err = call_err(m.binomial_tree_american, {**BASE, "risk_free_rate": 2.0, "volatility": 0.01, "steps": 2})
    assert "steps" in err


# ════════════════════════════════════════════════════════════════════════
# Heston
# ════════════════════════════════════════════════════════════════════════

HESTON_COS = dict(spot=100, strike=100, time_to_maturity=1, risk_free_rate=0.0, v0=0.0175, kappa=1.5768,
                  theta=0.0398, vol_of_vol=0.5751, rho=-0.5711)


def test_heston_reference_value_fang_oosterlee():
    # Fang & Oosterlee (2008) COS paper benchmark: 5.785155450
    r = call(m.heston_price, **HESTON_COS)
    approx(r["call"]["price"], 5.785155450, 1e-6)
    assert r["feller_condition"]["satisfied"] is False
    assert r["numerics"]["truncated"] is False


def test_heston_black_scholes_limit():
    args = dict(spot=100, strike=110, time_to_maturity=0.5, risk_free_rate=0.03, dividend_yield=0.01,
                v0=0.04, kappa=3, theta=0.04, rho=0.0, greeks=False)
    bs = call(m.black_scholes, spot=100, strike=110, time_to_maturity=0.5, risk_free_rate=0.03,
              dividend_yield=0.01, volatility=0.2)
    errs = []
    for eta in (0.1, 0.01, 0.001):
        h = call(m.heston_price, **args, vol_of_vol=eta)
        errs.append(abs(h["call"]["price"] - bs["call"]["price"]))
        approx(h["put"]["price"] - h["call"]["price"], bs["put"]["price"] - bs["call"]["price"], 1e-9)
    assert errs[1] < 1e-4 and errs[2] < 1e-6 and errs[0] > errs[1] > errs[2]  # O(eta^2) when rho = 0
    approx(h["call"]["black_scholes_implied_vol"], 0.2, 1e-5)


def test_heston_put_call_parity_and_greeks():
    args = {**HESTON_COS, "risk_free_rate": 0.03, "dividend_yield": 0.01, "strike": 95}
    r = call(m.heston_price, **args)
    S, K, T = 100, 95, 1
    approx(r["call"]["price"] - r["put"]["price"], S * math.exp(-0.01 * T) - K * math.exp(-0.03 * T), 1e-9)
    g = r["greeks"]
    approx(g["call_delta"] - g["put_delta"], math.exp(-0.01), 1e-6)
    assert 0 < g["call_delta"] < 1 and g["gamma"] > 0 and g["dprice_dv0"] > 0


def test_heston_skew_from_negative_correlation():
    r = call(m.heston_price, **{**HESTON_COS, "v0": 0.04, "theta": 0.04, "rho": -0.7},
             strikes=[80, 90, 100, 110, 120], greeks=False)
    ivs = [s["implied_vol"] for s in r["smile"]]
    assert all(v is not None for v in ivs)
    assert all(a > b for a, b in zip(ivs, ivs[1:])), ivs     # downward-sloping skew
    rp = call(m.heston_price, **{**HESTON_COS, "v0": 0.04, "theta": 0.04, "rho": 0.7},
              strikes=[80, 120], greeks=False)
    assert rp["smile"][0]["implied_vol"] < rp["smile"][1]["implied_vol"]


def test_heston_vs_monte_carlo():
    """Independent check: full-truncation Euler simulation of the Heston SDE."""
    S0, K, T, r, v0, kap, th, eta, rho = 100.0, 105.0, 0.5, 0.02, 0.05, 2.0, 0.05, 0.4, -0.6
    hp = call(m.heston_price, spot=S0, strike=K, time_to_maturity=T, risk_free_rate=r, v0=v0, kappa=kap,
              theta=th, vol_of_vol=eta, rho=rho, greeks=False)["call"]["price"]
    rng = random.Random(3)
    n, steps = 12000, 100
    dt = T / steps
    sq = math.sqrt(dt)
    c = math.sqrt(1 - rho * rho)
    pay = []
    for _ in range(n // 2):
        zs = [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(steps)]
        for sgn in (1, -1):
            x, v = math.log(S0), v0
            for z1, z2 in zs:
                z1, z2 = sgn * z1, sgn * z2
                vp = max(v, 0.0)
                x += (r - 0.5 * vp) * dt + math.sqrt(vp) * sq * z1
                v += kap * (th - vp) * dt + eta * math.sqrt(vp) * sq * (rho * z1 + c * z2)
            pay.append(math.exp(-r * T) * max(math.exp(x) - K, 0.0))
    mean = sum(pay) / len(pay)
    se = math.sqrt(sum((p - mean) ** 2 for p in pay) / (len(pay) - 1) / len(pay))
    assert abs(mean - hp) < 4 * se + 0.03, (mean, hp, se)


def test_heston_input_errors():
    call_err(m.heston_price, {**HESTON_COS, "rho": 1.5})
    call_err(m.heston_price, {**HESTON_COS, "kappa": 0})
    call_err(m.heston_price, {**HESTON_COS, "vol_of_vol": 0})
    call_err(m.heston_price, {**HESTON_COS, "strikes": "100"})
    call_err(m.heston_price, {k: v for k, v in HESTON_COS.items() if k != "v0"})


# ════════════════════════════════════════════════════════════════════════
# Avellaneda-Stoikov
# ════════════════════════════════════════════════════════════════════════

AS = dict(mid_price=100, risk_aversion=0.1, volatility=2, time_remaining=1, order_arrival_decay=1.5)


def test_as_paper_parameters():
    r = call(m.avellaneda_stoikov_quotes, **AS, inventory=0)
    spread = 0.1 * 4 * 1 + (2 / 0.1) * math.log(1 + 0.1 / 1.5)
    approx(r["optimal_spread"], spread, 1e-12)
    approx(spread, 1.6907704, 1e-6)
    approx(r["reservation_price"], 100, 1e-12)
    approx(r["bid"] + r["ask"], 200, 1e-12)


def test_as_inventory_skew():
    long_ = call(m.avellaneda_stoikov_quotes, **AS, inventory=3)
    short = call(m.avellaneda_stoikov_quotes, **AS, inventory=-3)
    approx(long_["reservation_price"], 100 - 3 * 0.1 * 4 * 1, 1e-12)
    assert long_["reservation_skew"] < 0 < short["reservation_skew"]
    approx(long_["optimal_spread"], short["optimal_spread"], 1e-12)  # spread independent of q
    assert long_["ask"] < 100 + long_["optimal_spread"] / 2
    assert "DOWN" in long_["interpretation"]
    # inventory term vanishes at the horizon
    end = call(m.avellaneda_stoikov_quotes, **{**AS, "time_remaining": 0}, inventory=5)
    approx(end["reservation_price"], 100, 1e-12)


def test_as_glft_limits_ticks_and_intensity():
    r = call(m.avellaneda_stoikov_quotes, **AS, inventory=2, order_arrival_rate=140, tick_size=0.05,
             max_inventory=2)
    assert r["bid"] is None and r["ask"] is not None and r["warnings"]
    g = r["glft_asymptotic"]
    approx(g["ask_distance"] - g["bid_distance"], -2 * 2 * g["skew_per_unit_inventory"], 1e-9)
    t = r["quotes_on_tick_grid"]
    assert t["bid"] is None and abs(t["ask"] / 0.05 - round(t["ask"] / 0.05)) < 1e-9
    r0 = call(m.avellaneda_stoikov_quotes, **AS, order_arrival_rate=140)
    fi = r0["fill_intensity"]
    approx(fi["bid"], 140 * math.exp(-1.5 * r0["bid_distance_from_mid"]), 1e-9)
    call_err(m.avellaneda_stoikov_quotes, {**AS, "risk_aversion": 0})
    call_err(m.avellaneda_stoikov_quotes, {**AS, "order_arrival_decay": -1})


# ════════════════════════════════════════════════════════════════════════
# Order-flow toxicity (VPIN)
# ════════════════════════════════════════════════════════════════════════

def test_vpin_extremes_given_sides():
    buys = [[100 + i * 0.01, 10, "buy"] for i in range(100)]
    r = call(m.order_flow_toxicity, trades=buys, num_buckets=10)
    assert r["classification"] == "given"
    approx(r["vpin"], 1.0, 1e-12)
    assert r["toxicity_level"] == "high"
    alt = [[100, 5, "buy" if i % 2 == 0 else "sell"] for i in range(200)]
    r = call(m.order_flow_toxicity, trades=alt, bucket_volume=100)
    approx(r["vpin"], 0.0, 1e-12)
    approx(r["order_imbalance"], 0.0, 1e-12)


def test_vpin_bucket_volume_conservation_and_splitting():
    rng = random.Random(11)
    trades = [[100 + rng.gauss(0, 0.1), rng.uniform(1, 37)] for _ in range(777)]
    r = call(m.order_flow_toxicity, trades=trades, bucket_volume=123.0, window=5)
    total = sum(t[1] for t in trades)
    approx(r["complete_buckets"] * 123.0 + r["partial_bucket"]["volume"], total, 1e-6)
    approx(r["buy_volume"] + r["sell_volume"], total, 1e-6)
    assert 0 <= r["vpin"] <= 1 and 0 < r["vpin_cdf_percentile"] <= 1
    assert r["vpin_window_buckets"] == 5


def test_vpin_bvc_and_tick_rule_direction():
    up = [[100 + 0.05 * i, 10] for i in range(300)]
    for method in ("bvc", "tick"):
        r = call(m.order_flow_toxicity, trades=up, classification=method)
        assert r["buy_volume"] > r["sell_volume"], method
    tick = call(m.order_flow_toxicity, trades=up, classification="tick")
    approx(tick["sell_volume"], 5.0, 1e-9)  # only the first (unclassifiable) trade is split 50/50


def test_vpin_kyle_lambda_recovers_impact():
    rng = random.Random(5)
    lam, p, trades = 0.002, 100.0, []
    for _ in range(4000):
        side = 1 if rng.random() < 0.5 else -1
        vol = rng.randint(1, 50)
        p += lam * side * vol + rng.gauss(0, 0.005)
        trades.append([p, vol, side])
    r = call(m.order_flow_toxicity, trades=trades, num_buckets=100)
    approx(r["kyle_lambda"]["lambda"], lam, 4e-4)
    assert r["kyle_lambda"]["r_squared"] > 0.8


def test_vpin_errors():
    call_err(m.order_flow_toxicity, {"trades": [[100, 1]]})
    call_err(m.order_flow_toxicity, {"trades": [[100, 1, "maybe"], [100, 1, "buy"]]})
    call_err(m.order_flow_toxicity, {"trades": [[100, 1], [100, 1]], "classification": "given"})
    call_err(m.order_flow_toxicity, {"trades": [[-1, 1], [100, 1]]})
    call_err(m.order_flow_toxicity, {"trades": [[100, 1], [100, 1]], "bucket_volume": 1e-9})


# ════════════════════════════════════════════════════════════════════════
# Optimal stopping
# ════════════════════════════════════════════════════════════════════════

def test_secretary_exact_and_asymptotic():
    r = call(m.optimal_stopping, problem="secretary", n=10)
    assert r["skip_first"] == 3
    want = Fraction(3, 10) * sum(Fraction(1, i) for i in range(3, 10))
    exact(r["success_probability"], str(want))
    big = call(m.optimal_stopping, problem="secretary", n=100_000)
    approx(big["cutoff_fraction"], 1 / math.e, 1e-4)
    approx(big["success_probability"]["decimal"], 1 / math.e, 1e-4)
    assert call(m.optimal_stopping, problem="secretary", n=1)["skip_first"] == 0
    assert call(m.optimal_stopping, problem="secretary", n=2)["success_probability"]["exact"] == "1/2"
    assert set(big["success_probability"]) == {"decimal"}                  # same shape, float path


def test_iid_uniform_finite_horizon_recursion():
    # V_{k+1} = (1 + V_k^2) / 2 on U[0,1]: 1/2, 5/8, 89/128
    for h, want in ((1, 0.5), (2, 5 / 8), (3, 89 / 128)):
        r = call(m.optimal_stopping, distribution="uniform", horizon=h)
        approx(r["value"]["decimal"], want, 1e-12)
    r = call(m.optimal_stopping, distribution="uniform", horizon=3)
    ts = [t["accept_if_at_least"] and t["accept_if_at_least"]["decimal"] for t in r["thresholds"]]
    approx(ts[0], 5 / 8, 1e-12)
    approx(ts[1], 1 / 2, 1e-12)
    assert ts[2] is None
    approx(r["expected_draws"]["decimal"], 1 + 5 / 8 + 5 / 8 * 1 / 2, 1e-12)


def test_iid_discrete_die_exact():
    r = call(m.optimal_stopping, distribution="discrete", outcomes=[1, 2, 3, 4, 5, 6], horizon=3)
    exact(r["value"], "14/3")
    r = call(m.optimal_stopping, outcomes=[0, 10], probabilities=["9/10", "1/10"], horizon=2, cost_per_draw="1/2")
    # V1 = 1 - 1/2 = 1/2; V2 = E[max(X, 1/2)] - 1/2 = 9/20 + 1 - 1/2
    exact(r["value"], "19/20")


def test_iid_infinite_horizon_closed_forms():
    r = call(m.optimal_stopping, distribution="uniform", cost_per_draw=0.02)
    approx(r["reservation_value"]["decimal"], 1 - math.sqrt(2 * 0.02), 1e-10)   # (1 - t)^2 / 2 = c
    approx(r["expected_draws"]["decimal"], 5.0, 1e-8)
    r = call(m.optimal_stopping, distribution="uniform", discount=0.9)
    approx(r["reservation_value"]["decimal"], (1 - math.sqrt(1 - 0.81)) / 0.9, 1e-10)  # t = 0.9 (1 + t^2) / 2
    r = call(m.optimal_stopping, distribution="normal", mean=10, std=2, cost_per_draw=0.1)
    t = r["reservation_value"]["decimal"]
    z = (t - 10) / 2
    e_excess = 2 * (math.exp(-z * z / 2) / math.sqrt(2 * math.pi) - z * 0.5 * math.erfc(z / math.sqrt(2)))
    approx(e_excess, 0.1, 1e-9)                                        # E[(X - t)^+] = c
    assert abs(r["fixed_point_residual"]) < 1e-9


def test_optimal_stopping_errors():
    call_err(m.optimal_stopping, {"distribution": "uniform"})          # infinite, no cost/discount
    call_err(m.optimal_stopping, {"distribution": "uniform", "low": 1, "high": 0, "horizon": 2})
    call_err(m.optimal_stopping, {"problem": "secretary"})
    call_err(m.optimal_stopping, {"outcomes": [1, 2], "probabilities": [0.5, 0.6], "horizon": 2})
    call_err(m.optimal_stopping, {"distribution": "uniform", "horizon": 2, "discount": 1.5})


# ════════════════════════════════════════════════════════════════════════
# Matrix games
# ════════════════════════════════════════════════════════════════════════

def _probs(side):
    return [Fraction(s["probability"]["exact"]) for s in side]


def test_matrix_classic_games():
    r = call(m.matrix_game_solver, payoff_matrix=[[1, -1], [-1, 1]])
    exact(r["game_value"], "0")
    assert _probs(r["row_strategy"]) == [Fraction(1, 2)] * 2
    r = call(m.matrix_game_solver, payoff_matrix=[[0, -1, 1], [1, 0, -1], [-1, 1, 0]],
             row_labels=["R", "P", "S"], col_labels=["R", "P", "S"])
    assert _probs(r["row_strategy"]) == [Fraction(1, 3)] * 3 == _probs(r["col_strategy"])
    assert r["fair_game"] and r["certificate"]["verified_equilibrium"]
    # 2x2 closed form: p = (d - c) / (a - b - c + d), v = (ad - bc) / (a - b - c + d)
    r = call(m.matrix_game_solver, payoff_matrix=[[2, -1], [-1, 1]])
    exact(r["game_value"], "1/5")
    assert _probs(r["row_strategy"]) == [Fraction(2, 5), Fraction(3, 5)]


def test_matrix_saddle_point_and_fractions():
    r = call(m.matrix_game_solver, payoff_matrix=[[3, 1], [4, 2]])
    exact(r["game_value"], "2")
    assert r["pure_saddle_points"] == [{"row": "R2", "col": "C2", "value": {"decimal": 2.0, "exact": "2"}}]
    assert _probs(r["row_strategy"]) == [0, 1]
    r = call(m.matrix_game_solver, payoff_matrix=[["1/2", "-1/3"], ["-1/4", "1/5"]])
    assert r["certificate"]["verified_equilibrium"]


def test_matrix_random_games_certified():
    rng = random.Random(2024)
    for _ in range(40):
        rows, cols = rng.randint(1, 7), rng.randint(1, 7)
        A = [[rng.randint(-9, 9) for _ in range(cols)] for _ in range(rows)]
        r = call(m.matrix_game_solver, payoff_matrix=A)
        v = Fraction(r["game_value"]["exact"])
        p, q = _probs(r["row_strategy"]), _probs(r["col_strategy"])
        assert sum(p) == 1 == sum(q) and min(p) >= 0 and min(q) >= 0
        assert min(sum(p[i] * A[i][j] for i in range(rows)) for j in range(cols)) == v
        assert max(sum(A[i][j] * q[j] for j in range(cols)) for i in range(rows)) == v
        assert r["pure_maximin"]["decimal"] <= float(v) <= r["pure_minimax"]["decimal"]


def test_matrix_errors():
    call_err(m.matrix_game_solver, {"payoff_matrix": []})
    call_err(m.matrix_game_solver, {"payoff_matrix": [[1, 2], [3]]})
    call_err(m.matrix_game_solver, {"payoff_matrix": [[1] * 26]})
    call_err(m.matrix_game_solver, {"payoff_matrix": [["x", 1]]})


# ════════════════════════════════════════════════════════════════════════
# Monte Carlo engine
# ════════════════════════════════════════════════════════════════════════

def test_mc_european_matches_black_scholes():
    r = call(m.monte_carlo_engine, product="european", **BASE, paths=40000)
    bs = r["benchmarks"]["black_scholes"]
    assert abs(r["price"] - bs) < 4 * r["std_error"]
    assert r["variance_reduction"]["antithetic_efficiency_gain"] > 1.5
    assert r["variance_reduction"]["control_variate"]["variance_reduction_factor"] > 5
    put = call(m.monte_carlo_engine, product="european", **BASE, option_type="put", paths=40000)
    assert abs(put["price"] - put["benchmarks"]["black_scholes"]) < 4 * put["std_error"]


def test_mc_geometric_asian_matches_closed_form():
    for kind in ("call", "put"):
        r = call(m.monte_carlo_engine, product="asian", average_type="geometric", option_type=kind, **BASE,
                 paths=20000, steps=24)
        ref = r["benchmarks"]["geometric_asian_closed_form"]
        assert abs(r["price"] - ref) < 4 * r["std_error"], (kind, r["price"], ref)


def test_mc_arithmetic_asian_control_variate():
    r = call(m.monte_carlo_engine, product="asian", **BASE, paths=20000, steps=24)
    geo = r["benchmarks"]["geometric_asian_closed_form_lower_bound"]
    assert r["price"] > geo                                  # AM >= GM
    assert r["price"] < r["benchmarks"]["european_black_scholes"]
    assert r["variance_reduction"]["control_variate"]["variance_reduction_factor"] > 50
    assert r["std_error"] < 0.01
    # Without the control variate the estimate agrees within noise.
    raw = call(m.monte_carlo_engine, product="asian", **BASE, paths=20000, steps=24, control_variate=False)
    assert abs(raw["price"] - r["price"]) < 4 * raw["std_error"]
    assert raw["variance_reduction"]["control_variate"] is None


def test_mc_barrier_in_out_parity_exact():
    common = dict(product="barrier", **BASE, barrier=90, paths=4000, steps=50, seed=7, control_variate=False)
    out = call(m.monte_carlo_engine, **common, barrier_type="down-and-out")
    inn = call(m.monte_carlo_engine, **common, barrier_type="down-and-in")
    vanilla = call(m.monte_carlo_engine, **{**common, "control_variate": True},
                   barrier_type="down-and-out")["variance_reduction"]["control_variate"]["sample_mean"]
    approx(out["price"] + inn["price"], vanilla, 1e-9)
    approx(out["barrier"]["knock_probability"], inn["barrier"]["knock_probability"], 0)


def test_barrier_closed_form_in_out_parity_all_types():
    for kind in ("call", "put"):
        vanilla = call(m.black_scholes, **BASE, dividend_yield=0.02)[kind]["price"]
        for H, K in ((90, 100), (90, 85), (110, 100), (110, 115)):
            up = H > 100
            pre = "up" if up else "down"
            i = m._barrier_closed_form(kind, f"{pre}-and-in", 100, K, H, 1, 0.05, 0.02, 0.2)
            o = m._barrier_closed_form(kind, f"{pre}-and-out", 100, K, H, 1, 0.05, 0.02, 0.2)
            v = vanilla if K == 100 else call(m.black_scholes, **{**BASE, "strike": K}, dividend_yield=0.02)[kind]["price"]
            approx(i + o, v, 1e-9)


def test_mc_barrier_continuous_matches_closed_form_all_types():
    for kind in ("call", "put"):
        for btype, H, K in (("down-and-out", 90, 100), ("down-and-in", 90, 100), ("down-and-out", 95, 90),
                            ("up-and-out", 120, 100), ("up-and-in", 120, 100), ("up-and-out", 110, 115)):
            r = call(m.monte_carlo_engine, product="barrier", spot=100, strike=K, time_to_maturity=1,
                     risk_free_rate=0.05, volatility=0.25, option_type=kind, barrier=H, barrier_type=btype,
                     monitoring="continuous", paths=12000, steps=100, seed=13)
            ref = r["benchmarks"]["continuous_closed_form"]
            assert abs(r["price"] - ref) < 4 * r["std_error"] + 0.03, (kind, btype, H, K, r["price"], ref)


def test_mc_barrier_already_breached_and_rebate():
    r = call(m.monte_carlo_engine, product="barrier", **BASE, barrier=105, barrier_type="down-and-out",
             rebate=3, paths=2000, steps=10, control_variate=False)
    approx(r["price"], 3 * math.exp(-0.05), 1e-9)          # knocked at t0 -> discounted rebate
    approx(r["barrier"]["knock_probability"], 1.0, 0)
    r = call(m.monte_carlo_engine, product="barrier", **BASE, barrier=105, barrier_type="down-and-in",
             paths=2000, steps=10)
    assert abs(r["price"] - r["benchmarks"]["vanilla_black_scholes"]) < 4 * r["std_error"] + 1e-9


def test_mc_determinism_and_antithetic_toggle():
    a = call(m.monte_carlo_engine, **BASE, paths=2000, steps=12, seed=99)
    b = call(m.monte_carlo_engine, **BASE, paths=2000, steps=12, seed=99)
    assert a["price"] == b["price"]
    c = call(m.monte_carlo_engine, **BASE, paths=2000, steps=12, seed=100)
    assert a["price"] != c["price"]
    na = call(m.monte_carlo_engine, **BASE, paths=2000, steps=12, antithetic=False)
    assert na["variance_reduction"]["antithetic_efficiency_gain"] is None
    assert na["simulation"]["paths"] == 2000


def test_mc_errors():
    call_err(m.monte_carlo_engine, {**BASE, "paths": 2_000_000, "steps": 100})
    call_err(m.monte_carlo_engine, {**BASE, "product": "barrier"})                 # missing barrier
    call_err(m.monte_carlo_engine, {**BASE, "product": "lookback"})
    call_err(m.monte_carlo_engine, {**BASE, "antithetic": "maybe"})
    call_err(m.monte_carlo_engine, {**BASE, "paths": 10})


# ════════════════════════════════════════════════════════════════════════
# EV calculator & order book (original modules)
# ════════════════════════════════════════════════════════════════════════

def test_ev_puzzles():
    ev = lambda **kw: call(m.ev_calculator, **kw)  # noqa: E731
    exact(ev(puzzle="dice_reroll", faces=6, rolls=2)["expected_value"], "17/4")
    exact(ev(puzzle="dice_reroll", faces=6, rolls=3)["expected_value"], "14/3")
    exact(ev(puzzle="dice_max", faces=6, dice=2)["expected_max"], "161/36")
    approx(ev(puzzle="card_red_black")["expected_value"]["decimal"], 2.6244755489939, 1e-9)
    exact(ev(puzzle="card_red_black", red=2, black=2)["expected_value"], "2/3")
    for pat, want in (("HH", "6"), ("HT", "4"), ("HTH", "10"), ("HHH", "14"), ("HHTHH", "38")):
        exact(ev(puzzle="coin_pattern", pattern=pat)["expected_flips"], want)
    exact(ev(puzzle="coin_pattern", pattern="HH", p_heads="1/3")["expected_flips"], "12")
    exact(ev(puzzle="coin_race", pattern_a="THH", pattern_b="HHT")["p_a_first"], "3/4")
    exact(ev(puzzle="coin_race", pattern_a="HHT", pattern_b="HTH")["p_a_first"], "2/3")
    exact(ev(puzzle="coin_race", pattern_a="HH", pattern_b="TT")["expected_flips_until_either"], "3")
    b = ev(puzzle="bayes", priors=["999/1000", "1/1000"], likelihoods=["1/2", 1], num_observations=10)
    exact(b["posteriors"][1]["posterior"], "1024/2023")
    exact(ev(puzzle="coupon_collector", coupons=6)["expected_draws"], "147/10")
    exact(ev(puzzle="kelly", p_win=0.6, odds=1)["kelly_fraction"], "1/5")
    gr = ev(puzzle="gamblers_ruin", start=5, target=10, p_win=0.5)
    approx(gr["p_reach_target"], 0.5)
    approx(gr["expected_steps"], 25)
    gr = ev(puzzle="gamblers_ruin", start=3, target=7, p_win=0.6)
    qp = 0.4 / 0.6
    approx(gr["p_reach_target"], (1 - qp ** 3) / (1 - qp ** 7))
    c = ev(puzzle="custom", outcomes=[1, 2, 3, 4, 5, 6], cost=3)
    exact(c["expected_value"], "1/2")
    assert c["verdict"].startswith("EV > 0")


def test_ev_hth_simulation():
    rng = random.Random(7)
    tot = 0
    for _ in range(20000):
        s, n = "", 0
        while not s.endswith("HTH"):
            s += rng.choice("HT")
            n += 1
        tot += n
    approx(tot / 20000, 10, 0.25)


def test_order_book_metrics():
    ob = call(m.order_book_metrics, bids=[[99.98, 300], [99.99, 100], [99.97, 500]],
              asks=[[100.01, 300], [100.02, 400]], impact_quantity=500, tick_size=0.01)
    approx(ob["mid_price"], 100.0)
    approx(ob["spread"], 0.02)
    approx(ob["spread_bps"], 2.0, 1e-6)
    approx(ob["spread_ticks"], 2.0, 1e-6)
    approx(ob["microprice"], (99.99 * 300 + 100.01 * 100) / 400)
    approx(ob["top_of_book_imbalance"], -0.5)
    approx(ob["market_impact"]["buy_market_order"]["avg_price"], (300 * 100.01 + 200 * 100.02) / 500)
    crossed = call(m.order_book_metrics, bids=[[101, 1]], asks=[[100, 1]])
    assert crossed["book_state"] == "crossed"


# ════════════════════════════════════════════════════════════════════════
# Almgren-Chriss
# ════════════════════════════════════════════════════════════════════════

AC = dict(shares=1e6, horizon=5, periods=5, volatility=0.3 * 50 / math.sqrt(250), permanent_impact=2.5e-7,
          temporary_impact=2.5e-6, fixed_cost=0.0625, risk_aversion=1e-6, spot=50)


def _ac_utility(x, a):
    tau = a["horizon"] / a["periods"]
    eta_t = a["temporary_impact"] - 0.5 * a["permanent_impact"] * tau
    n = [x[k - 1] - x[k] for k in range(1, len(x))]
    E = 0.5 * a["permanent_impact"] * x[0] ** 2 + a["fixed_cost"] * sum(abs(v) for v in n) + eta_t / tau * sum(v * v for v in n)
    V = a["volatility"] ** 2 * tau * sum(v * v for v in x[1:])
    return E, V, E + a["risk_aversion"] * V


def test_ac_paper_example_closed_form():
    r = call(m.almgren_chriss_liquidation, **AC)
    tau, eta_t = 1.0, 2.5e-6 - 0.5 * 2.5e-7
    kt2 = 1e-6 * AC["volatility"] ** 2 / eta_t
    kappa = math.acosh(1 + 0.5 * kt2 * tau * tau) / tau
    approx(r["derived"]["kappa"], kappa, 1e-12)
    approx(kappa, 0.6063, 1e-4)                                   # AC (2000): kappa ~ 0.6 / day
    x = [s["holdings"] for s in r["schedule"]]
    for k, xk in enumerate(x):                                    # independent sinh formula
        approx(xk, 1e6 * math.sinh(kappa * (5 - k)) / math.sinh(kappa * 5), 1e-6)
    E, V, U = _ac_utility(x, AC)
    approx(r["optimal"]["expected_cost"], E, 1e-6)
    approx(r["optimal"]["variance"], V, 1e-3)
    approx(r["benchmarks"]["twap"]["expected_cost"], 662_500.0, 1e-6)   # 125000 + 62500 + 475000
    assert r["optimality_certificate"]["euler_lagrange_max_rel_residual"] < 1e-12
    assert r["optimality_certificate"]["beats_twap"] and r["optimality_certificate"]["beats_immediate"]
    approx(r["optimal"]["expected_cost_bps"], E / (1e6 * 50) * 1e4, 1e-9)


def test_ac_risk_neutral_twap_and_urgency_monotone():
    r0 = call(m.almgren_chriss_liquidation, **{**AC, "risk_aversion": 0})
    assert [round(s["holdings"]) for s in r0["schedule"]] == [1_000_000, 800_000, 600_000, 400_000, 200_000, 0]
    prev = None
    for lam in (1e-8, 1e-7, 1e-6, 1e-5):
        r = call(m.almgren_chriss_liquidation, **{**AC, "risk_aversion": lam})
        cur = (r["pct_executed_first_half"], r["optimal"]["expected_cost"], r["optimal"]["variance"])
        if prev:
            assert cur[0] > prev[0] and cur[1] > prev[1] and cur[2] < prev[2]   # more urgency: E up, V down
        prev = cur
    f = r["efficient_frontier"]
    assert all(a["expected_cost"] >= b["expected_cost"] - 1e-6 for a, b in zip(f, f[1:]))  # sorted by std


def test_ac_optimum_beats_random_perturbations():
    a = {**AC, "periods": 20, "horizon": 5}
    r = call(m.almgren_chriss_liquidation, **a, max_schedule_rows=1000)
    x = [s["holdings"] for s in r["schedule"]]
    U0 = _ac_utility(x, a)[2]
    rng = random.Random(9)
    for _ in range(200):
        y = x[:]
        for k in range(1, len(y) - 1):
            y[k] = min(max(y[k] + rng.gauss(0, 5_000), 0), 1e6)
        assert _ac_utility(y, a)[2] > U0


def test_ac_monte_carlo_shortfall_matches_mean_variance():
    """Simulate the AC price process directly; shortfall mean/variance must match E and V."""
    r = call(m.almgren_chriss_liquidation, **AC)
    x = [s["holdings"] for s in r["schedule"]]
    n = [x[k - 1] - x[k] for k in range(1, len(x))]
    rng = random.Random(4)
    S0, sig, gam, eta, eps = 50.0, AC["volatility"], 2.5e-7, 2.5e-6, 0.0625
    sf = []
    for _ in range(20000):
        S, cash = S0, 0.0
        for nk in n:
            cash += nk * (S - eps - eta * nk)                    # sale at S_{k-1} - h(n_k / tau), tau = 1
            S += sig * rng.gauss(0, 1) - gam * nk                # S_k = S_{k-1} + sigma xi - gamma n_k
        sf.append(1e6 * S0 - cash)
    mean = sum(sf) / len(sf)
    var = sum((v - mean) ** 2 for v in sf) / (len(sf) - 1)
    assert abs(mean - r["optimal"]["expected_cost"]) < 4 * math.sqrt(var / len(sf))
    approx(var, r["optimal"]["variance"], rel=0.05)


def test_ac_buy_side_alias_and_errors():
    sell = call(m.almgren_chriss_liquidation, **AC)
    buy = call(m.almgren_chriss_liquidation, **{**AC, "shares": -1e6})
    approx(buy["optimal"]["expected_cost"], sell["optimal"]["expected_cost"], 1e-6)
    assert buy["side"].startswith("buy")
    alias = {k: v for k, v in AC.items() if k != "permanent_impact"}
    kl = call(m.almgren_chriss_liquidation, **alias, kyle_lambda=2.5e-7)
    approx(kl["optimal"]["utility"], sell["optimal"]["utility"], 1e-6)
    call_err(m.almgren_chriss_liquidation, {**AC, "kyle_lambda": 1e-7})              # both aliases
    call_err(m.almgren_chriss_liquidation, {**AC, "shares": 0})
    err = call_err(m.almgren_chriss_liquidation, {**AC, "temporary_impact": 1e-7})   # eta_tilde <= 0
    assert "eta_tilde" in err


# ════════════════════════════════════════════════════════════════════════
# Longstaff-Schwartz
# ════════════════════════════════════════════════════════════════════════

LS = dict(spot=36, strike=40, time_to_maturity=1, risk_free_rate=0.06, volatility=0.2, exercise_dates=50)
LS_BERMUDAN = 4.478015  # CRR, 2000 steps, exercise every 40th step (converged: 4.4792 / 4.4784 / 4.4780)


def test_lsm_python_backend_longstaff_schwartz_table1():
    r = call(m.longstaff_schwartz_american, **LS, backend="python", paths=20000, boundary_points=0)
    assert r["backend"] == "python"
    approx(r["benchmarks"]["bermudan_crr"], LS_BERMUDAN, 2e-3)
    assert abs(r["price"] - LS_BERMUDAN) < 4 * r["std_error"] + 0.02
    ins = r["estimators"]["in_sample"]["price"]
    assert abs(ins - LS_BERMUDAN) < 0.05                       # LS (2001) report 4.472 (s.e. 0.010)
    assert r["price"] > r["benchmarks"]["european_black_scholes"] + 0.5
    assert 0.5 < r["early_exercise"]["probability_before_maturity"] < 0.9
    # European CV helps little for an ITM American put (early exercise decorrelates the payoffs): ~1.1x here
    assert r["estimators"]["out_of_sample"]["control_variate"]["variance_reduction_factor"] >= 1.0


def test_lsm_numpy_backend_and_low_bias():
    need_numpy()
    oos = []
    for seed in range(6):
        r = call(m.longstaff_schwartz_american, **LS, paths=50000, seed=seed, benchmark=False, boundary_points=0)
        assert r["backend"].startswith("numpy")
        oos.append(r["estimators"]["out_of_sample"]["price"])
    mean = sum(oos) / 6
    sd = math.sqrt(sum((v - mean) ** 2 for v in oos) / 5)
    assert abs(mean - LS_BERMUDAN) < 0.015
    assert mean < LS_BERMUDAN + 3 * sd / math.sqrt(6)          # out-of-sample policy value is low-biased


def test_lsm_backends_agree():
    need_numpy()
    py = call(m.longstaff_schwartz_american, **LS, backend="python", paths=20000, benchmark=False, boundary_points=0)
    nv = call(m.longstaff_schwartz_american, **LS, backend="numpy", paths=100000, benchmark=False, boundary_points=0)
    assert abs(py["price"] - nv["price"]) < 4 * math.hypot(py["std_error"], nv["std_error"]) + 0.02


def test_lsm_boundary_tracks_crr():
    r = call(m.longstaff_schwartz_american, **LS, backend="python", paths=20000, boundary_points=10)
    crr = m._crr_tree(36, 40, 1, 0.06, 0, 0.2, 1000, "put")["boundary"]
    pts = [p for p in r["exercise_boundary"]["points"] if p["critical_spot"] is not None and p["time"] > 0.2]
    assert len(pts) >= 6
    for p in pts:
        ref = crr[round(p["time"] * 1000)]
        assert p["critical_spot"] < 40 and abs(p["critical_spot"] - ref) < 1.5, (p, ref)


def test_lsm_call_without_dividends_is_european():
    r = call(m.longstaff_schwartz_american, **{**BASE, "option_type": "call", "exercise_dates": 25},
             backend="python", paths=20000, boundary_points=0)
    bs = r["benchmarks"]["european_black_scholes"]
    assert abs(r["price"] - bs) < 4 * r["std_error"] + 0.03
    assert r["early_exercise"]["probability_before_maturity"] < 0.02


def test_lsm_bases_and_determinism():
    kw = {**LS, "exercise_dates": 20, "backend": "python", "paths": 8000, "benchmark": False, "boundary_points": 0}
    a = call(m.longstaff_schwartz_american, **kw)
    assert a["price"] == call(m.longstaff_schwartz_american, **kw)["price"]
    for basis in ("monomial", "laguerre_unweighted"):
        b = call(m.longstaff_schwartz_american, **{**kw, "basis": basis})
        assert abs(b["price"] - a["price"]) < 0.05, basis


def test_lsm_errors_and_backend_fallback():
    call_err(m.longstaff_schwartz_american, {**LS, "paths": 5_000_000, "exercise_dates": 500})
    call_err(m.longstaff_schwartz_american, {**LS, "degree": 9})
    call_err(m.longstaff_schwartz_american, {**LS, "basis": "hermite"})
    if m._np is None:
        r = call(m.longstaff_schwartz_american, **{**LS, "exercise_dates": 10}, backend="numpy", paths=2000,
                 benchmark=False, boundary_points=0)
        assert r["backend"] == "python" and "NumPy" in r["warnings"][0]


# ════════════════════════════════════════════════════════════════════════
# Black-Litterman
# ════════════════════════════════════════════════════════════════════════

BL_ASSETS = ["US", "EU", "JP", "EM"]
BL_VOLS = [0.16, 0.18, 0.20, 0.25]
BL_CORR = [[1, .7, .5, .6], [.7, 1, .5, .6], [.5, .5, 1, .5], [.6, .6, .5, 1]]
BL_W = [0.55, 0.25, 0.10, 0.10]
BL = dict(assets=BL_ASSETS, volatilities=BL_VOLS, correlation=BL_CORR, market_weights=BL_W)


def _inv(A):
    n = len(A)
    M_ = [list(map(float, row)) + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r_: abs(M_[r_][c]))
        M_[c], M_[p] = M_[p], M_[c]
        d = M_[c][c]
        M_[c] = [x / d for x in M_[c]]
        for r_ in range(n):
            if r_ != c:
                f = M_[r_][c]
                M_[r_] = [a - f * b for a, b in zip(M_[r_], M_[c])]
    return [row[n:] for row in M_]


def _mv(A, x):
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def test_bl_matches_precision_form_posterior():
    """Independent check: Theil mixed estimation [(tS)^-1 + P'O^-1P]^-1 [(tS)^-1 pi + P'O^-1 q]."""
    views = [{"asset": "EM", "return": 0.10, "confidence": 0.6},
             {"assets": {"US": 1, "EU": -1}, "return": 0.01, "omega": 0.0009}]
    r = call(m.black_litterman_optimizer, **BL, views=views)
    n, tau = 4, 0.05
    Sig = [[BL_CORR[i][j] * BL_VOLS[i] * BL_VOLS[j] for j in range(n)] for i in range(n)]
    pi = [2.5 * v for v in _mv(Sig, BL_W)]
    P, q = [[0, 0, 0, 1], [1, -1, 0, 0]], [0.10, 0.01]
    om = [0.4 / 0.6 * tau * Sig[3][3], 0.0009]
    tSi = _inv([[tau * x for x in row] for row in Sig])
    prec = [[tSi[i][j] + sum(P[k][i] * P[k][j] / om[k] for k in range(2)) for j in range(n)] for i in range(n)]
    rhs = [a + sum(P[k][i] * q[k] / om[k] for k in range(2)) for i, a in enumerate(_mv(tSi, pi))]
    mu = _mv(_inv(prec), rhs)
    for a, x in zip(BL_ASSETS, mu):
        approx(r["posterior_excess_returns"][a], x, 1e-12)
    for a, x in zip(BL_ASSETS, pi):
        approx(r["equilibrium_excess_returns"][a], x, 1e-14)
    # posterior covariance of the mean = [(tS)^-1 + P'O^-1P]^-1  ->  Sigma_post = Sigma + that
    Minv = _inv(prec)
    for i, a in enumerate(BL_ASSETS):
        approx(r["posterior_volatility"][a], math.sqrt(Sig[i][i] + Minv[i][i]), 1e-12)


def test_bl_no_view_identities():
    r = call(m.black_litterman_optimizer, **BL)
    pf = r["portfolios"]
    for a, w in zip(BL_ASSETS, BL_W):
        approx(r["posterior_excess_returns"][a], r["equilibrium_excess_returns"][a], 1e-15)
        approx(pf["unconstrained_prior_cov"]["weights"][a], w, 1e-12)            # reverse optimisation
        approx(pf["unconstrained_posterior_cov"]["weights"][a], w / 1.05, 1e-12)  # He-Litterman: w/(1+tau)
    prior = call(m.black_litterman_optimizer, **BL, use_posterior_covariance=False)["portfolios"]
    for a, w in zip(BL_ASSETS, BL_W):  # with the prior covariance, w_mkt is already fully invested -> optimal
        approx(prior["fully_invested"]["weights"][a], w, 1e-9)
        approx(prior["long_only"]["weights"][a], w, 1e-9)


def test_bl_confidence_monotone_and_certain_view():
    fr = []
    for c in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
        r = call(m.black_litterman_optimizer, **BL, views=[{"asset": "EM", "return": 0.10, "confidence": c}])
        fr.append(r["views"][0]["fraction_of_gap_closed"])
    assert all(a < b for a, b in zip(fr, fr[1:]))
    approx(fr[-1], 1.0, 1e-12)                                     # c = 1: posterior satisfies the view exactly
    approx(fr[2], 0.5, 1e-12)                                      # c = 0.5: omega = tau p S p' -> halfway


def test_bl_tilt_lies_in_view_span():
    views = [{"asset": "EM", "return": 0.10, "confidence": 0.6},
             {"assets": {"US": 1, "EU": -1}, "return": 0.01, "confidence": 0.4}]
    r = call(m.black_litterman_optimizer, **BL, views=views)
    lam = [v["tilt_lambda"] for v in r["views"]]
    P = [[0, 0, 0, 1], [1, -1, 0, 0]]
    w = r["portfolios"]["unconstrained_prior_cov"]["weights"]
    for i, a in enumerate(BL_ASSETS):
        approx(w[a] - BL_W[i], sum(P[k][i] * lam[k] for k in range(2)), 1e-12)   # w* = w_mkt + P' Lambda


def test_bl_long_only_qp_optimal():
    views = [{"asset": "EM", "return": 0.15, "confidence": 0.9}, {"asset": "JP", "return": -0.05, "confidence": 0.8}]
    for use_post in (True, False):
        r = call(m.black_litterman_optimizer, **BL, views=views, max_weight=0.4, use_posterior_covariance=use_post)
        lo = r["portfolios"]["long_only"]
        w = [lo["weights"][a] for a in BL_ASSETS]
        approx(sum(w), 1.0, 1e-12)
        assert all(-1e-12 <= x <= 0.4 + 1e-12 for x in w) and lo["solver"]["kkt_max_violation"] < 1e-9
    # With the prior covariance the objective is fully reconstructible here: brute-force dominance check.
    n = 4
    C = [[BL_CORR[i][j] * BL_VOLS[i] * BL_VOLS[j] for j in range(n)] for i in range(n)]
    mu = [r["posterior_excess_returns"][a] for a in BL_ASSETS]
    obj = lambda v: sum(a * b for a, b in zip(v, mu)) - 1.25 * sum(v[i] * v[j] * C[i][j]  # noqa: E731
                                                                   for i in range(n) for j in range(n))
    best = obj(w)
    rng = random.Random(3)
    for _ in range(3000):
        v = m._proj_capped_simplex([rng.random() * 2 for _ in range(n)], 0.4)
        assert obj(v) <= best + 1e-12


def test_bl_market_excess_return_and_errors():
    r = call(m.black_litterman_optimizer, **BL, market_excess_return=0.06)
    approx(r["market"]["implied_excess_return"], 0.06, 1e-12)
    call_err(m.black_litterman_optimizer, {**BL, "views": [{"asset": "XX", "return": 0.1}]})
    call_err(m.black_litterman_optimizer, {**BL, "max_weight": 0.2})                     # 4 * 0.2 < 1
    call_err(m.black_litterman_optimizer, {**BL, "correlation": [[1, 2, 0, 0], [2, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]})
    call_err(m.black_litterman_optimizer, {"assets": ["A", "B"], "covariance": [[1, 2], [2, 1]], "market_weights": [.5, .5]})
    dup = [{"asset": "EM", "return": 0.1, "confidence": 1}, {"asset": "EM", "return": 0.12, "confidence": 1}]
    assert "linearly dependent" in call_err(m.black_litterman_optimizer, {**BL, "views": dup})


# ════════════════════════════════════════════════════════════════════════
# Jane Street EV trainer
# ════════════════════════════════════════════════════════════════════════

TR_CATS = ("conditional_expectation", "dice_stopping", "card_stopping", "auction")


def _gen(**kw):
    return call(m.jane_street_ev_trainer, action="generate", **kw)


def _solve(pid):
    return call(m.jane_street_ev_trainer, action="solve", puzzle_id=pid)["solution"]


def _find(cat, template, diff, pred=lambda pz: True, limit=400):
    for seed in range(limit):
        pz = m._tr_build(cat, template, diff, seed)
        if pred(pz):
            return pz
    raise AssertionError(f"no {template} puzzle found")


def _val(x):
    return Fraction(x["exact"]) if isinstance(x, dict) and "exact" in x else (x["decimal"] if isinstance(x, dict) else x)


def test_trainer_generate_deterministic_and_answer_hidden():
    a = _gen(category="auction", difficulty=2, seed=11)
    b = _gen(category="auction", difficulty=2, seed=11)
    assert a == b and a["puzzle_id"].startswith("JSQR-AU-")
    assert not {"ev", "solution", "verdict", "dist"} & set(a)
    assert "[TAKE]" in a["answer_contract"] and "half-Kelly" in a["answer_contract"]
    assert _gen(difficulty=1, seed=5)["category"] in TR_CATS


def test_trainer_sweep_measures_and_verdict_rule():
    seen = set()
    for cat in TR_CATS:
        for d in (1, 2, 3):
            for seed in range(25):
                g = _gen(category=cat, difficulty=d, seed=seed)
                pz = m._tr_build(*m._tr_parse_id(g["puzzle_id"]))
                assert sum(pz["dist"].values()) == 1 and all(p > 0 for p in pz["dist"].values())
                sol = m._tr_solution(pz)
                seen.add((cat, sol["verdict"]))
                if sol["ev"] <= 0:
                    assert sol["verdict"] == "PASS"
                elif sol["verdict"] == "HEDGE":
                    h = pz["hedge"]
                    assert h["var_after"] < h["cap"] < h["var_before"]
                    approx(float(h["var_after"]), float(h["var_before"] - h["ratio"] ** 2 * h["var_hedge"]), 1e-12)
                elif sol["size"] is not None:                         # TAKE: Kelly first-order condition
                    net = {x - pz["price"]: p for x, p in pz["dist"].items()}
                    n_star, B = 2 * sol["size"], float(pz["bankroll"])
                    foc = sum(float(p) * float(x) / (B + n_star * float(x)) for x, p in net.items())
                    assert abs(foc) < 1e-9 * max(1.0, sum(abs(float(x)) for x in net) / B)
    assert {v for _, v in seen} == {"TAKE", "PASS", "HEDGE"}


def test_trainer_reroll_matches_ev_calculator():
    for seed in range(15):
        pz = m._tr_build("dice_stopping", "reroll", 2, seed)
        n = int(re.search(r"fair (\d+)-sided", pz["prompt"]).group(1))
        k = int(re.search(r"up to (\d+) rolls", pz["prompt"]).group(1))
        c = re.search(r"re-roll costs \$(\d+(?:\.\d+)?)", pz["prompt"])
        c = Fraction(c.group(1)) if c else Fraction(0)
        ref = call(m.ev_calculator, puzzle="dice_reroll", faces=n, rolls=k, cost_per_roll=str(c))
        fair, _ = m._moments(pz["dist"])
        assert fair == Fraction(ref["expected_value"]["exact"]) + c    # trainer's first roll is free


def test_trainer_red_black_matches_ev_calculator():
    pz = _find("card_stopping", "red_black", 1)  # difficulty 1 uses +1 red / -1 black
    R, B = map(int, re.search(r"deck of (\d+) red and (\d+) black", pz["prompt"]).groups())
    ref = call(m.ev_calculator, puzzle="card_red_black", red=R, black=B)
    assert m._moments(pz["dist"])[0] == Fraction(ref["expected_value"]["exact"])


def test_trainer_pig_hold_at_20():
    pz = _find("dice_stopping", "pig", 2, lambda p: "6-sided" in p["prompt"])
    fair = m._moments(pz["dist"])[0]
    assert fair == Fraction(492303203, 60466176)                  # 8.14179..., the classic hold-at-20 value
    rng = random.Random(1)
    tot, n = 0, 60000
    for _ in range(n):
        s = 0
        while s < 20:
            f = rng.randint(1, 6)
            if f == 1:
                s = 0
                break
            s += f
        tot += s
    assert abs(tot / n - float(fair)) < 0.12


def test_trainer_cards_bruteforce():
    names = {"an ace": lambda c: c[0] == 1, "a heart": lambda c: c[1] == 0, "a red card": lambda c: c[1] in (0, 1),
             "a face card (J, Q, K)": lambda c: c[0] >= 11}
    deck = [(r_, s_) for r_ in range(1, 14) for s_ in range(4)]
    checked = 0
    for seed in range(12):
        pz = m._tr_build("conditional_expectation", "cards", 2, seed)
        mt = re.search(r"(\d) cards are dealt.*at least one of them is (.+?)\. You may pay .+? to receive \$(\d+) "
                       r"for every card among the \d that is (.+?)\.$", pz["prompt"])
        d, a1, mult, a2 = int(mt.group(1)), mt.group(2), int(mt.group(3)), mt.group(4)
        num = den = 0
        for hand in itertools.combinations(deck, d):
            if any(names[a1](c) for c in hand):
                den += 1
                num += mult * sum(names[a2](c) for c in hand)
        assert m._moments(pz["dist"])[0] == Fraction(num, den)
        checked += 1
    assert checked == 12


def test_trainer_auctions_bruteforce():
    for seed in range(10):
        pz = m._tr_build("auction", "acquire", 2, seed)
        N = int(re.search(r"integers 0\.\.(\d+)", pz["prompt"]).group(1))
        k = Fraction(re.search(r"worth (\d+(?:/\d+)?)\*V", pz["prompt"]).group(1))
        b0 = int(re.search(r"offer b = \$(\d+)", pz["prompt"]).group(1))
        ev = sum((k * v - b0) for v in range(N + 1) if v <= b0) / Fraction(N + 1)
        assert m._tr_solution(pz)["ev"] == ev
        assert pz["extra"]["optimal_bid"] == (0 if k < 2 else N)
    for seed in range(10):
        pz = m._tr_build("auction", "first_price", 2, seed)
        v = int(re.search(r"value an item at \$(\d+)", pz["prompt"]).group(1))
        H = int(re.search(r"uniform on 0\.\.(\d+)", pz["prompt"]).group(1))
        b0 = int(re.search(r"bid \$(\d+)\?", pz["prompt"]).group(1))
        win = sum(Fraction(1) if b0 > x else (Fraction(1, 2) if b0 == x else 0) for x in range(H + 1)) / (H + 1)
        assert m._tr_solution(pz)["ev"] == (v - b0) * win - pz["price"]
        evs = {b: (v - b) * sum(Fraction(1) if b > x else (Fraction(1, 2) if b == x else 0)
                                for x in range(H + 1)) / (H + 1) for b in range(0, min(v, H + 1) + 1)}
        assert evs[pz["extra"]["optimal_bid"]] == max(evs.values())


def test_trainer_next_red_martingale():
    pz = _find("card_stopping", "next_red", 1)
    R, B = map(int, re.search(r"deck of (\d+) red and (\d+) black", pz["prompt"]).groups())
    rng = random.Random(8)
    wins, n = 0, 20000
    for _ in range(n):  # a 'greedy' strategy: call when reds are over-represented, else at the last card
        deck = [1] * R + [0] * B
        rng.shuffle(deck)
        r_, b_ = R, B
        for i, c in enumerate(deck):
            if r_ > b_ or i == len(deck) - 1:
                wins += c
                break
            r_, b_ = r_ - c, b_ - (1 - c)
    assert abs(wins / n - R / (R + B)) < 4 * math.sqrt(0.25 / n)


def test_trainer_check_scoring_and_strictness():
    g = _gen(category="dice_stopping", difficulty=1, seed=3)
    sol = _solve(g["puzzle_id"])
    ev_exact = sol["ev"]["exact"] if isinstance(sol["ev"], dict) else str(sol["ev"])
    size = _val(sol["size"]) if sol["size"] is not None else None
    proof = "State space: ... Transition probabilities: ... Terminal payoffs: ... EV = x. " + sol["verdict_line"]
    perfect = call(m.jane_street_ev_trainer, action="check", puzzle_id=g["puzzle_id"], verdict=sol["verdict_line"],
                   ev=ev_exact, size=float(size) if size else None, proof=proof)
    assert perfect["score"] == 100 and perfect["grade"] == "desk-ready"
    wrong_tag = "[PASS]" if sol["verdict"] != "PASS" else "[TAKE] 5"
    bad = call(m.jane_street_ev_trainer, action="check", puzzle_id=g["puzzle_id"], verdict=wrong_tag, ev=ev_exact)
    assert bad["breakdown"]["verdict"]["points"] == 0 and bad["score"] <= 60
    dec = call(m.jane_street_ev_trainer, action="check", puzzle_id=g["puzzle_id"], verdict=sol["verdict_line"],
               ev=round(float(Fraction(ev_exact)), 6))
    assert dec["breakdown"]["ev"]["points"] in (27, 30)
    for v in ("TAKE", "[TAKE] [PASS]", "[take]"):
        call_err(m.jane_street_ev_trainer, {"action": "check", "puzzle_id": g["puzzle_id"], "verdict": v, "ev": 1})
    call_err(m.jane_street_ev_trainer, {"action": "check", "puzzle_id": "JSQR-XX-dice-1-1", "verdict": "[PASS]", "ev": 0})
    call_err(m.jane_street_ev_trainer, {"action": "solve", "puzzle_id": "JSQR-DS-pig-1-1"})   # pig needs diff >= 2


def test_trainer_hedge_and_full_kelly_scoring():
    pz = None
    for seed in range(300):
        for d in (2, 3):
            cand = m._tr_build("conditional_expectation", "dice", d, seed)
            if m._tr_solution(cand)["verdict"] == "HEDGE":
                pz = cand
                break
        if pz:
            break
    assert pz is not None
    sol = m._tr_solution(pz)
    ok = call(m.jane_street_ev_trainer, action="check", puzzle_id=pz["puzzle_id"], verdict=sol["verdict_line"],
              ev=str(sol["ev"]), size=float(sol["size"]))
    assert ok["breakdown"]["size"]["points"] == 20 and ok["breakdown"]["verdict"]["points"] == 40
    take = None
    for seed in range(300):
        cand = m._tr_build("auction", "acquire", 2, seed)
        s = m._tr_solution(cand)
        if s["verdict"] == "TAKE" and s["size"]:
            take = (cand, s)
            break
    cand, s = take
    full = call(m.jane_street_ev_trainer, action="check", puzzle_id=cand["puzzle_id"], verdict="[TAKE]",
                ev=str(s["ev"]), size=2 * s["size"])
    assert full["breakdown"]["size"]["points"] == 10                   # full Kelly: half credit


# ════════════════════════════════════════════════════════════════════════
# Robustness: zero-crash fuzzing of every tool
# ════════════════════════════════════════════════════════════════════════

_JUNK = [None, True, False, 0, -1, 1, 2, 0.5, -0.5, 1e308, -1e308, float("nan"), float("inf"), "", "abc",
         "1/3", "1/0", "1e999999", "-7", [], [1], [[1, 2]], [["x"]], {}, {"a": 1}, [[1, 2], [3, 4]],
         [[100, 1], [101, 2], [99, 3]], "call", "put", "barrier", "secretary", "bvc", 10**30]


def test_fuzz_every_tool_never_raises_and_returns_strict_json():
    rng = random.Random(1234)
    ctx = FakeCtx()
    m.register(ctx)
    t_start = time.time()
    for name, t in ctx.tools.items():
        props = list(t["schema"]["parameters"]["properties"])
        handler = t["handler"]
        for args in (None, "string", 42, [], {}):
            out = json.loads(handler(args), parse_constant=_no_nan)
            assert "ok" in out
        for _ in range(250):
            args = {p: rng.choice(_JUNK) for p in props if rng.random() < 0.7}
            raw = handler(args)
            out = json.loads(raw, parse_constant=_no_nan)
            assert isinstance(out, dict) and "ok" in out, (name, args)
            if not out["ok"]:
                assert "internal error" not in out["error"], (name, args, out["error"])
    assert time.time() - t_start < 60


def test_handlers_accept_extra_kwargs():
    out = json.loads(m.black_scholes(dict(BASE), task_id="t1", session_id="s"))
    assert out["ok"]


# ── standalone runner ──────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    t0 = time.time()
    for name, fn in tests:
        t = time.time()
        try:
            fn()
            print(f"PASS  {name}  ({time.time() - t:.2f}s)")
        except Skip as e:
            skipped += 1
            print(f"SKIP  {name}  ({e})")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed - skipped} passed, {failed} failed, {skipped} skipped in {time.time() - t0:.1f}s")
    sys.exit(1 if failed else 0)
