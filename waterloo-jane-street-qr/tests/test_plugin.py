"""Known-answer tests for waterloo-jane-street-qr. Run: python3 tests/test_plugin.py"""
import importlib.util
import json
import math
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("wjsqr", ROOT / "__init__.py")
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

passed = failed = 0


def call(fn, **args):
    raw = fn(args)
    json.loads(raw)  # strict JSON
    return json.loads(raw)


def check(name, got, want, tol=1e-6):
    global passed, failed
    ok = (got == want) if isinstance(want, str) else (got is not None and abs(got - want) <= tol)
    passed += ok
    failed += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r}, want {want!r}")


# ── Black-Scholes (Hull, S=K=100, T=1, r=5%, vol=20%) ──
r = call(m.black_scholes, spot=100, strike=100, time_to_maturity=1, risk_free_rate=0.05, volatility=0.2)
check("bs call", r["call"]["price"], 10.450584, 1e-5)
check("bs put", r["put"]["price"], 5.573526, 1e-5)
check("bs call delta", r["call"]["delta"], 0.636831, 1e-5)
check("bs put delta", r["put"]["delta"], -0.363169, 1e-5)
check("bs gamma", r["call"]["gamma"], 0.018762, 1e-5)
check("bs vega", r["call"]["vega"], 37.524035, 1e-4)
check("bs call theta", r["call"]["theta"], -6.414028, 1e-4)
check("bs put theta", r["put"]["theta"], -1.657880, 1e-4)
check("bs parity", r["put_call_parity_residual"], 0.0, 1e-10)
# finite-difference Greeks vs analytic, with dividends
base = dict(spot=97, strike=103, time_to_maturity=0.37, risk_free_rate=0.03, volatility=0.31, dividend_yield=0.015)
px = lambda **kw: call(m.black_scholes, **{**base, **kw})["put"]["price"]
g = call(m.black_scholes, **base)["put"]
h = 1e-4
check("fd put delta", (px(spot=97 + h) - px(spot=97 - h)) / (2 * h), g["delta"], 1e-6)
check("fd put gamma", (px(spot=97 + 1e-2) - 2 * px() + px(spot=97 - 1e-2)) / 1e-4, g["gamma"], 1e-5)
check("fd put vega", (px(volatility=0.31 + h) - px(volatility=0.31 - h)) / (2 * h), g["vega"], 1e-5)
check("fd put theta", -(px(time_to_maturity=0.37 + h) - px(time_to_maturity=0.37 - h)) / (2 * h), g["theta"], 1e-5)
iv = call(m.black_scholes, spot=100, strike=100, time_to_maturity=1, risk_free_rate=0.05, volatility=0.5,
          option_type="call", market_price=10.450584)
check("implied vol", iv["implied_vol_solve"]["implied_volatility"], 0.2, 1e-6)
check("bs expiry intrinsic", call(m.black_scholes, spot=110, strike=100, time_to_maturity=0,
                                  risk_free_rate=0.05, volatility=0.2)["call"]["price"], 10.0)

# ── EV puzzles ──
ev = lambda **kw: call(m.ev_calculator, **kw)
check("dice reroll 2", ev(puzzle="dice_reroll", faces=6, rolls=2)["expected_value"]["exact"], "17/4")
check("dice reroll 3", ev(puzzle="dice_reroll", faces=6, rolls=3)["expected_value"]["exact"], "14/3")
check("dice max 2d6", ev(puzzle="dice_max", faces=6, dice=2)["expected_max"]["exact"], "161/36")
check("red-black 26/26", ev(puzzle="card_red_black")["expected_value"]["decimal"], 2.6244755489939, 1e-9)
check("red-black 2/2", ev(puzzle="card_red_black", red=2, black=2)["expected_value"]["exact"], "2/3")
for pat, want in [("HH", "6"), ("HT", "4"), ("HTH", "10"), ("HHH", "14"), ("HHTHH", "38")]:
    check(f"pattern {pat}", ev(puzzle="coin_pattern", pattern=pat)["expected_flips"]["exact"], want)
check("biased HH p=1/3", ev(puzzle="coin_pattern", pattern="HH", p_heads="1/3")["expected_flips"]["exact"], "12")
check("penney THH>HHT", ev(puzzle="coin_race", pattern_a="THH", pattern_b="HHT")["p_a_first"]["exact"], "3/4")
check("penney HHT>HTH", ev(puzzle="coin_race", pattern_a="HHT", pattern_b="HTH")["p_a_first"]["exact"], "2/3")
check("race HH vs TT length", ev(puzzle="coin_race", pattern_a="HH", pattern_b="TT")["expected_flips_until_either"]["exact"], "3")
b = ev(puzzle="bayes", priors=["999/1000", "1/1000"], likelihoods=["1/2", 1], num_observations=10)
check("bayes double-headed", b["posteriors"][1]["posterior"]["exact"], "1024/2023")
check("coupon 6", ev(puzzle="coupon_collector", coupons=6)["expected_draws"]["exact"], "147/10")
check("kelly", ev(puzzle="kelly", p_win=0.6, odds=1)["kelly_fraction"]["exact"], "1/5")
gr = ev(puzzle="gamblers_ruin", start=5, target=10, p_win=0.5)
check("ruin fair p", gr["p_reach_target"], 0.5)
check("ruin fair dur", gr["expected_steps"], 25)
gr = ev(puzzle="gamblers_ruin", start=3, target=7, p_win=0.6)
q_p = 0.4 / 0.6
check("ruin biased", gr["p_reach_target"], (1 - q_p ** 3) / (1 - q_p ** 7))
c = ev(puzzle="custom", outcomes=[1, 2, 3, 4, 5, 6], cost=3)
check("custom die minus 3", c["expected_value"]["exact"], "1/2")
check("custom verdict", c["verdict"], "EV > 0 — take it (size it)")
# Monte-Carlo cross-check of HTH waiting time
random.seed(7)
tot = 0
for _ in range(20000):
    s, n = "", 0
    while not s.endswith("HTH"):
        s += random.choice("HT"); n += 1
    tot += n
check("MC HTH ~ 10", tot / 20000, 10, 0.25)

# ── Order book ──
ob = call(m.order_book_metrics, bids=[[99.98, 300], [99.99, 100], [99.97, 500]],
          asks=[[100.01, 300], [100.02, 400]], impact_quantity=500, tick_size=0.01)
check("ob mid", ob["mid_price"], 100.0)
check("ob spread", ob["spread"], 0.02, 1e-9)
check("ob spread bps", ob["spread_bps"], 2.0, 1e-6)
check("ob microprice", ob["microprice"], (99.99 * 300 + 100.01 * 100) / 400, 1e-9)
check("ob imbalance", ob["top_of_book_imbalance"], -0.5)
check("ob buy 500 avg", ob["market_impact"]["buy_market_order"]["avg_price"], (300 * 100.01 + 200 * 100.02) / 500, 1e-9)

# ── Error handling never raises ──
for fn, args in [(m.black_scholes, {"spot": -1}), (m.ev_calculator, {"puzzle": "nope"}),
                 (m.order_book_metrics, {"bids": "x"}), (m.black_scholes, None),
                 (m.ev_calculator, {"puzzle": "coin_pattern", "pattern": "HH", "p_heads": 0})]:
    out = json.loads(fn(args))
    check(f"error path {fn.__name__}", str(out["ok"]), "False")

check("persona fits 4000 chars", float(len(m.PERSONA_PROMPT) <= 4000), 1.0)
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
