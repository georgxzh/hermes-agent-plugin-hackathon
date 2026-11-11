# Hermes Agent Plugin Hackathon by Nous Research

<p align="left">
  <img src="assets/hermes.jpg" alt="Hermes Agent" height="100" style="margin-right: 10px;">
  <img src="assets/janestreet.png" alt="Jane Street" height="100" style="margin-right: 10px;">
  <img src="assets/waterloo.png" alt="Waterloo" height="100">
</p>

The plugin below is built for the [Nous Research Hermes Agent Plugin Hackathon](https://x.com/NousResearch/status/2118094256438936062) 
hosted by [Nous Research](https://nousresearch.com/).

# Waterloo-Jane-Street-QR

**A math-first quantitative research and trading-interview assistant, packaged as a [Hermes Agent](https://github.com/NousResearch/hermes-agent) v0.22.7 plugin.**

Fourteen deterministic, validated pricing / execution / portfolio / probability / microstructure engines exposed as LLM tools, plus a system-prompt persona and two lifecycle hooks that force every answer into a desk-grade decision format: *state space → transitions → payoffs → EV → `[TAKE]` / `[PASS]` / `[HEDGE]`*.

Built for the Hermes Plugin Hackathon to the standard you’d expect from a Waterloo Math/CS intern on a Jane Street–style trading desk, with an ICPC / Putnam background: That means every answer is derived, computed, verified two ways, and sized.

| | |
|---|---|
| Plugin id | `waterloo-jane-street-qr` · v1.0.0 |
| Host | Hermes Agent v0.22.7 (plugin manifest v2) |
| Runtime deps | **None required.** Python standard library only. **Optional:** NumPy vectorises `longstaff_schwartz_american` (5× path budget, ~10× faster); SciPy is not used. |
| Python | 3.11 (Hermes venv) and 3.14 tested |
| Tools | 14 · Hooks: `pre_llm_call`, `transform_llm_output` · 1 system-prompt section |
| Tests | 81 (known-answer, cross-model, simulation, brute-force, property and fuzz tests), ~15–20 s |
| License | MIT |

---

## Table of contents

1. [Executive overview](#1-executive-overview)
2. [Architecture](#2-architecture)
3. [Installation](#3-installation)
4. [Tool inventory & mathematical specifications](#4-tool-inventory--mathematical-specifications)
   - [4.1 `black_scholes`](#41-black_scholes)
   - [4.2 `binomial_tree_american`](#42-binomial_tree_american)
   - [4.3 `heston_price`](#43-heston_price)
   - [4.4 `monte_carlo_engine`](#44-monte_carlo_engine)
   - [4.5 `avellaneda_stoikov_quotes`](#45-avellaneda_stoikov_quotes)
   - [4.6 `order_flow_toxicity`](#46-order_flow_toxicity)
   - [4.7 `optimal_stopping`](#47-optimal_stopping)
   - [4.8 `matrix_game_solver`](#48-matrix_game_solver)
   - [4.9 `ev_calculator`](#49-ev_calculator)
   - [4.10 `order_book_metrics`](#410-order_book_metrics)
   - [4.11 `almgren_chriss_liquidation`](#411-almgren_chriss_liquidation)
   - [4.12 `longstaff_schwartz_american`](#412-longstaff_schwartz_american)
   - [4.13 `black_litterman_optimizer`](#413-black_litterman_optimizer)
   - [4.14 `jane_street_ev_trainer`](#414-jane_street_ev_trainer)
5. [System prompt & persona engineering](#5-system-prompt--persona-engineering)
6. [Programmatic usage & API contract](#6-programmatic-usage--api-contract)
7. [Test suite](#7-test-suite)
8. [Numerical validation summary](#8-numerical-validation-summary)
9. [Limitations & model risk](#9-limitations--model-risk)
10. [Repository layout](#10-repository-layout)

---

## 1. Executive overview

LLMs are bad at arithmetic and worse at being calibrated. On a trading desk or in a quant interview, the failure modes are always the same: a number nobody derived, an answer with no EV, a "good trade" with no size and no hedge.

`waterloo-jane-street-qr` addresses this on two fronts.

**1. Computation is moved out of the model.** Every number that matters comes from a deterministic tool:

- closed forms where they exist (Black–Scholes–Merton, geometric Asian, Reiner–Rubinstein barriers, Avellaneda–Stoikov);
- exact rational arithmetic for discrete problems (`fractions.Fraction`: Bayes, Markov chains, backward induction, the simplex method);
- convergent numerics where they don't (CRR lattices, Fourier integration, variance-reduced Monte Carlo).

Every tool reports its own diagnostics, such as standard errors, put–call parity residuals, LP duality certificates, quadrature truncation flags and analytic benchmarks with z-scores. That lets the model check its own numbers instead of trusting them.

**2. The reasoning format is enforced.** A system-prompt section sets the persona and a mandatory answer contract. A per-turn hook restates the contract, and an output hook audits the final answer, flagging any reply that lacks an explicit EV or a verdict tag.

Intended uses:

- quant research scratchpad: pricing, Greeks, implied vol, smile shape, early exercise;
- market-making and microstructure analysis: quoting, inventory skew, toxicity, price impact;
- quant / trading interview preparation: EV puzzles, optimal stopping, game theory, Kelly sizing.

It is a research and teaching tool, **not** an execution system (see [§9](#9-limitations--model-risk)).

---

## 2. Architecture

```
                         ┌─────────────────────────────── Hermes Agent v0.22.7 ───────────────────────────────┐
 user ── message ──────▶ │  system prompt  ◀── register_system_prompt_section("waterloo-jane-street-qr.persona") │
                         │       │                    (frozen per session → prompt-cache safe)                  │
                         │       ▼                                                                             │
                         │  pre_llm_call ──▶ appends DESK_REMINDER to the turn's user message (ephemeral)      │
                         │       │                                                                             │
                         │       ▼                                                                             │
                         │   LLM  ⇄  tool calls  ──▶  handler(args: dict, **kw) -> str (JSON)                  │
                         │       │                    │                                                        │
                         │       │                    └─ _safe(): validate → compute → _clean() → json.dumps   │
                         │       ▼                                                                             │
                         │  transform_llm_output ──▶ audit_response(): EV present? verdict tag present?        │
                         │       │                    → append compliance footer if not (mode = strict)         │
                         └───────┼─────────────────────────────────────────────────────────────────────────────┘
                                 ▼
                              user
```

### Layering inside `__init__.py`

| Layer | Responsibility |
|---|---|
| **Input validation** (`_num`, `_int`, `_frac`, `_prob`, `_choice`, `_bool`, `_matrix`) | Type coercion with explicit domains. Rejects `bool`-as-number, NaN/∞, out-of-range values and pathological fraction strings (>64 chars or >4096-bit numerators). Raises `ToolInputError` with a message the model can act on. |
| **Pure numerical kernels** (`_bs_core`, `_crr_tree`, `_heston_call`, `_solve_zero_sum`, `_solve_linear`, `_barrier_closed_form`, `_geometric_asian_closed_form`, …) | Side-effect-free math. Kernels call each other for cross-checks; for example, the binomial tree uses `_bs_core` as a control variate and the Monte Carlo engine uses it as a benchmark. |
| **Tool functions** (`_black_scholes`, …) | `dict -> dict`. Orchestrate kernels, attach diagnostics, warnings and benchmarks. |
| **Safety wrapper** (`_safe`) | Turns a `dict -> dict` function into a Hermes handler `(args, **kwargs) -> str` that **never raises**. |
| **Serialisation** (`_clean`, `_frac_out`) | Strict JSON. NaN/∞ become `null`. A `Fraction` becomes `{"decimal": float, "exact": "p/q"}`; `exact` is omitted when it would exceed 40 digits. |
| **Persona & enforcement** (`PERSONA_PROMPT`, `DESK_REMINDER`, `audit_response`, `_make_hooks`) | Prompt engineering and the per-turn and output hooks. |
| **Registration** (`register(ctx)`) | Registers 14 tools under toolset `waterloo_qr`, the prompt section and both hooks, with a fallback for older hosts. |

### Design rules

- **Determinism.** Monte Carlo is seeded (`seed`, default 42). The same inputs always produce byte-identical outputs.
- **Exact before approximate.** Discrete problems are solved over ℚ, with the decimal rendered alongside the exact fraction.
- **Bounded cost.** Every tool has hard input caps (lattice steps ≤ 2000, `paths × steps` ≤ 5·10⁶, matrix ≤ 25×25, trades ≤ 200k, …), so a single tool call cannot stall the agent loop.
- **Self-verifying outputs.** Each result carries the evidence needed to trust it: parity residuals, z-scores against closed forms, duality certificates, `truncated` flags, warnings when inputs look like the wrong units.

---

## 3. Installation

Prerequisite: a working Hermes Agent v0.22.7 install (`hermes --version`).

```bash
git clone https://github.com/georgxzh/hermes-agent-plugin-hackathon.git
cd hermes-agent-plugin-hackathon

# Link the plugin into your Hermes home (respects profiles via $HERMES_HOME)
ln -sfn "$PWD/waterloo-jane-street-qr" "${HERMES_HOME:-$HOME/.hermes}/plugins/waterloo-jane-street-qr"
hermes plugins enable waterloo-jane-street-qr

# Validate with the real runtime contracts and the catalog gate
hermes plugins doctor --ci waterloo-jane-street-qr
hermes plugins validate waterloo-jane-street-qr
```

Expected output: `registrations: 14 tool(s), 2 hook(s)` and `security scan — safe`.

Tools and the system-prompt section take effect from the **next new session**.

**Hot-reload after editing the code.** A running gateway keeps the old module loaded. Symptom: the model claims a newly added tool "doesn't exist". Fix:

```bash
hermes plugins disable waterloo-jane-street-qr && hermes plugins enable waterloo-jane-street-qr
```

### Configuration

| Key | Values | Default | Effect |
|---|---|---|---|
| `plugins.entries.waterloo-jane-street-qr.settings.enforcement` | `strict` · `reminder` · `off` | `strict` | See [§5.3](#53-enforcement-modes) |

```bash
hermes config set plugins.entries.waterloo-jane-street-qr.settings.enforcement reminder
```

---

## 4. Tool inventory & mathematical specifications

Conventions used by every tool:

- **Units.** Rates, yields and volatilities are **decimals** (5% → `0.05`, 20% vol → `0.20`). Times are in **years**. Tools warn if inputs look like percentages or days.
- **Measure.** Pricing is under the risk-neutral measure ℚ, with continuous compounding and a continuous dividend / borrow yield $q$.
- **Return shape.** Every tool returns `{"ok": true, ...}` or `{"ok": false, "error": "..."}`. Exact quantities are returned as `{"decimal": 0.25, "exact": "1/4"}`.

| Tool | Domain | Method | Complexity |
|---|---|---|---|
| `black_scholes` | European options | Closed form + bisection IV | $O(1)$ |
| `binomial_tree_american` | American options | CRR lattice, backward induction | $O(N^2)$ time, $O(N)$ memory |
| `heston_price` | Stochastic vol | Lewis Fourier integral, Gauss–Legendre | ~10³ CF evaluations per price (16 per panel) |
| `monte_carlo_engine` | Path-dependent | GBM + antithetic + control variates | $O(\text{paths}\cdot\text{steps})$ |
| `avellaneda_stoikov_quotes` | Market making | AS (2008) closed form + GLFT asymptotics | $O(1)$ |
| `order_flow_toxicity` | Microstructure | Volume-bucket VPIN, BVC, Kyle λ | $O(\text{trades} + \text{buckets}\cdot w)$ |
| `optimal_stopping` | Stopping | Backward induction / fixed point | $O(n)$ or $O(H\cdot|X|)$ |
| `matrix_game_solver` | Game theory | Exact rational simplex, LP duality | Bland's rule, ≤ 25×25 |
| `ev_calculator` | Probability puzzles | Exact Markov chains, DP over ℚ | Puzzle-dependent |
| `order_book_metrics` | Microstructure | Top-of-book / depth analytics | $O(L \log L)$ |
| `almgren_chriss_liquidation` | Optimal execution | Closed-form discrete AC trajectory + frontier | $O(N)$ |
| `longstaff_schwartz_american` | American / Bermudan | LSM, Laguerre regression, in/out-of-sample | $O(M N d^2)$, NumPy-vectorised |
| `black_litterman_optimizer` | Portfolio construction | Bayesian posterior + FISTA/active-set QP | $O(n^3 + k^2 n)$ |
| `jane_street_ev_trainer` | Interview drills | Exact DP / enumeration, Kelly, min-var hedge | Puzzle-dependent |

---

### 4.1 `black_scholes`

Closed-form Black–Scholes–Merton pricing with continuous dividend yield, all first-order Greeks plus Gamma, and an implied-volatility solver.

```math
d_{1}=\frac{\ln(S/K)+\left(r-q+\tfrac12\sigma^{2}\right)T}{\sigma\sqrt T},\qquad d_{2}=d_{1}-\sigma\sqrt T
```

```math
C=S e^{-qT}N(d_1)-K e^{-rT}N(d_2),\qquad P=K e^{-rT}N(-d_2)-S e^{-qT}N(-d_1)
```

| Greek | Call | Put |
|---|---|---|
| Δ | $e^{-qT}N(d_1)$ | $-e^{-qT}N(-d_1)$ |
| Γ | $\dfrac{e^{-qT}\varphi(d_1)}{S\sigma\sqrt T}$ | same |
| Vega | $S e^{-qT}\varphi(d_1)\sqrt T$ | same |
| Θ | $-\dfrac{S e^{-qT}\varphi(d_1)\sigma}{2\sqrt T}-rKe^{-rT}N(d_2)+qSe^{-qT}N(d_1)$ | $-\dfrac{S e^{-qT}\varphi(d_1)\sigma}{2\sqrt T}+rKe^{-rT}N(-d_2)-qSe^{-qT}N(-d_1)$ |
| ρ | $KTe^{-rT}N(d_2)$ | $-KTe^{-rT}N(-d_2)$ |

Greeks are reported per unit change and also in desk units: `vega_per_vol_point` (÷100), `theta_per_day` (÷365) and `rho_per_bp` (÷10⁴). Other outputs: risk-neutral $\mathbb{P}(\text{ITM}) = N(\pm d_2)$, forward $F = Se^{(r-q)T}$, log-moneyness $\ln(F/K)$, and the put–call parity residual $(C-P)-(Se^{-qT}-Ke^{-rT})$.

**Implied volatility.** The price is strictly increasing in σ, so the solver uses **bisection** on $[10^{-9}, 5]$, doubling the upper end up to 100 if needed. It runs at most 200 iterations to a tolerance of $10^{-12}$.

Bisection is chosen over Newton because convergence is guaranteed, including deep OTM and near expiry where vega → 0 makes Newton diverge.

Before solving, the market price is checked against the no-arbitrage band $[\text{price}(\sigma{=}0),\ Se^{-qT})$ for calls, or $[\text{price}(\sigma{=}0),\ Ke^{-rT})$ for puts. A quote outside that band returns a structured error instead of a fake vol.

**Degenerate limits.** When $T=0$ or $\sigma=0$, the tool returns the deterministic limit: the discounted payoff on the forward, with Γ = Vega = 0 and Θ undefined at expiry.

| Parameter | Type | Default | Domain |
|---|---|---|---|
| `spot`, `strike` | number | required | > 0 |
| `time_to_maturity` | number | required | ≥ 0 (years) |
| `risk_free_rate` | number | required | [−1, 5] |
| `volatility` | number | required | [0, 20] |
| `dividend_yield` | number | 0 | [−1, 5] |
| `option_type` | `call`/`put`/`both` | `both` | |
| `market_price` | number | none | Triggers the IV solve |

---

### 4.2 `binomial_tree_american`

Cox–Ross–Rubinstein lattice for American calls and puts, giving the price, the early-exercise premium, the **early-exercise boundary**, and Greeks.

```math
\Delta t=\frac{T}{N},\quad u=e^{\sigma\sqrt{\Delta t}},\quad d=\frac1u,\quad p=\frac{e^{(r-q)\Delta t}-d}{u-d}
```

```math
V_{i,j}=\max\Big(\underbrace{\phi\,(S_{i,j}-K)}_{\text{exercise}},\ \underbrace{e^{-r\Delta t}\big[pV_{i+1,j+1}+(1-p)V_{i+1,j}\big]}_{\text{continuation}}\Big),\qquad \phi=\pm1
```

**Implementation notes**

- **Single backward pass.** American and European values are rolled back together on the same lattice. Spot levels come from a precomputed table $S\,e^{k\sigma\sqrt{\Delta t}}$, $k\in[-N,N]$. Memory is $O(N)$.
- **Arbitrage guard.** If $p\notin(0,1)$, meaning $|r-q|$ is too large relative to $\sigma\sqrt{\Delta t}$, the tool returns an error telling the caller to increase `steps`. It never silently prices with negative probabilities.
- **Control-variate correction (Hull).** The lattice error of the American and European claims is highly correlated, so the tool also reports
  $V^{\text{Am}}_{\text{CV}} = V^{\text{Am}}_{\text{tree}} + \big(V^{\text{Eu}}_{\text{BS}} - V^{\text{Eu}}_{\text{tree}}\big)$.
- **Exercise boundary.** At each time slice it records the critical spot $S^*(t)$. For puts this is the highest node where immediate exercise strictly beats continuation; for calls, the lowest. Resolution is one tree step. Up to `boundary_points` points are returned, sampled uniformly in time.
- **Greeks.** Δ and Γ use the step-1 and step-2 nodes, and $\Theta = (V_{2,1}-V_{0,0})/(2\Delta t)$, so they cost no extra trees. Vega and ρ use central bump-and-reprice (σ bump $\max(10^{-4}, 0.01\sigma)$, r bump $10^{-4}$), but only for $N \le 1000$.
- **Sanity flags.** `exercise_now_optimal` is set when intrinsic value ≥ lattice value. A call on a non-dividend asset correctly reports *no* early-exercise region (Merton's theorem).

Example: ATM put with S=K=100, T=1, r=5%, σ=20%, N=1000.

| Output | Value |
|---|---|
| `american_price` | 6.0896 |
| `american_price_control_variate` | 6.0916 |
| `early_exercise_premium` | 0.5181 |
| Boundary $S^*(t)$ at t = 0.25 / 0.5 / 0.75 / 1.0 | 81.68 / 83.77 / 87.01 / 100.00 |
| Δ | −0.4111 (vs −0.3632 for the European) |

| Parameter | Type | Default | Domain |
|---|---|---|---|
| `spot`, `strike` | number | required | > 0, with $S/K\in[10^{-6},10^6]$ |
| `time_to_maturity` | number | required | (0, 100] |
| `risk_free_rate` | number | required | [−1, 5] |
| `volatility` | number | required | (0, 20] |
| `dividend_yield` | number | 0 | [−1, 5] |
| `option_type` | `call`/`put` | `put` | |
| `steps` | int | 500 | [2, 2000] |
| `boundary_points` | int | 20 | [0, 200] |
| `greeks` | bool | true | |

---

### 4.3 `heston_price`

European options under Heston (1993) stochastic volatility.

```math
\begin{aligned}
dS_t &= (r-q)S_t\,dt+\sqrt{v_t}\,S_t\,dW^{(1)}_t\\
dv_t &= \kappa(\theta-v_t)\,dt+\eta\sqrt{v_t}\,dW^{(2)}_t,\qquad d\langle W^{(1)},W^{(2)}\rangle_t=\rho\,dt
\end{aligned}
```

**Characteristic function.** The CF of $\ln S_T$ is evaluated in the *"little Heston trap"* form (Albrecher, Mayer, Schoutens & Tistaert, 2007). This form has no branch-cut discontinuity in the complex logarithm, which is what breaks the original 1993 formula for long maturities.

```math
\begin{aligned}
b&=\kappa-\rho\eta\, iz,\quad D=\sqrt{b^2+\eta^2(iz+z^2)},\quad g=\frac{b-D}{b+D}\\
C&=\frac{\kappa\theta}{\eta^2}\Big[(b-D)T-2\ln\frac{1-ge^{-DT}}{1-g}\Big],\quad
\mathcal D=\frac{b-D}{\eta^2}\cdot\frac{1-e^{-DT}}{1-ge^{-DT}}\\
\phi(z)&=\exp\!\big(iz\,[\ln S+(r-q)T]+C+\mathcal D\,v_0\big)
\end{aligned}
```

**Pricing integral.** The call uses Lewis (2001)'s single integral, which needs only one CF evaluation per node and has a $1/(u^2+\tfrac14)$ damping factor:

```math
C = S e^{-qT} - \frac{\sqrt{K}\,e^{-rT}}{\pi}\int_0^\infty \mathrm{Re}\Big[e^{-iu\ln K}\,\phi\!\big(u-\tfrac{i}{2}\big)\Big]\frac{du}{u^2+\tfrac14}
```

**Quadrature**

- Adaptive panels of width 2, each integrated with 16-point Gauss–Legendre. Nodes are computed at import by Newton iteration on $P_{16}$.
- Integration stops once $u \ge 20$ and $\max|f|\cdot u < 10^{-13}(\sqrt{SK}+1)$, a tail bound from the integrand's $O(u^{-2})$ decay. The hard cap is $u = 20\,000$; hitting it sets `numerics.truncated = true` and emits a warning.
- The result is clipped to the no-arbitrage band $[\max(Se^{-qT}-Ke^{-rT},0),\ Se^{-qT}]$, and the put comes from parity, so parity holds to machine precision.

**Outputs**

- call and put prices, with Black–Scholes implied vols;
- **Feller check** $2\kappa\theta-\eta^2>0$ (reported, not enforced: prices remain valid when it fails);
- expected average variance $\bar v = \theta+(v_0-\theta)\frac{1-e^{-\kappa T}}{\kappa T}$;
- Δ, Γ and $\partial V/\partial v_0$ by central differences;
- optionally an **implied-vol smile** over up to 40 `strikes`, each priced off the OTM leg.

Reference check (Fang & Oosterlee 2008 COS benchmark: S=K=100, T=1, r=0, $v_0$=0.0175, κ=1.5768, θ=0.0398, η=0.5751, ρ=−0.5711):

| | Value |
|---|---|
| Published | 5.785155450 |
| `heston_price` | **5.785155431** (93 panels, $u_{\max}$ = 186) |
| Smile IV at K = 80 / 100 / 120 | 20.26% / 14.51% / 12.78% (negative skew from ρ < 0) |

| Parameter | Type | Default | Domain |
|---|---|---|---|
| `spot`, `strike` | number | required | > 0, with $K/S\in[10^{-4},10^4]$ |
| `time_to_maturity` | number | required | (0, 50] |
| `risk_free_rate` | number | required | [−1, 5] |
| `dividend_yield` | number | 0 | [−1, 5] |
| `v0`, `theta` | number | required | [0, 25] (variance units: 0.04 = 20% vol) |
| `kappa` | number | required | (0, 100] |
| `vol_of_vol` | number | required | [10⁻⁴, 10] |
| `rho` | number | required | [−1, 1] (clamped to ±0.999999 internally) |
| `option_type` | `call`/`put`/`both` | `both` | |
| `strikes` | number[] | none | ≤ 40 strikes, for the smile |
| `greeks` | bool | true | |

---

### 4.4 `monte_carlo_engine`

Seeded Monte Carlo under GBM for **Asian**, **barrier** and **European** options, with **antithetic variates** and product-specific **control variates**.

**Path generation.** Exact log-Euler steps (no discretisation bias in the marginals):

```math
\ln S_{t+\Delta t}=\ln S_t+\Big(r-q-\tfrac12\sigma^2\Big)\Delta t+\sigma\sqrt{\Delta t}\,Z,\qquad Z\sim\mathcal N(0,1)
```

**Antithetic variates.** Each draw $Z_{1:N}$ also drives the mirrored path $-Z_{1:N}$. The estimator averages each pair, $\hat Y_k=\tfrac12\big(Y(Z^{(k)})+Y(-Z^{(k)})\big)$, and the standard error is computed over pair means, which is the correct variance for correlated pairs. The reported efficiency gain is

```math
\text{gain}_{\text{AV}}=\left(\frac{\widehat{\mathrm{SE}}_{\text{plain, same budget}}}{\widehat{\mathrm{SE}}_{\text{antithetic}}}\right)^2
```

**Control variates.** $\hat Y_{\text{CV}}=\bar Y-\hat\beta(\bar X-\mathbb E[X])$, with $\hat\beta=\widehat{\mathrm{Cov}}(Y,X)/\widehat{\mathrm{Var}}(X)$ estimated in-sample:

| Product | Control $X$ | Known $\mathbb E[X]$ |
|---|---|---|
| Arithmetic Asian | Discrete geometric-average Asian | Closed form (below) |
| Barrier | Vanilla payoff on the same paths | Black–Scholes |
| European | Discounted terminal spot $e^{-rT}S_T$ | $Se^{-qT}$ |

The geometric Asian closed form uses fixings $t_1,\dots,t_N$ with $\Delta t = T/N$:

```math
\mu_G=\ln S+\Big(r-q-\tfrac{\sigma^2}{2}\Big)\Delta t\,\frac{N+1}{2},\qquad
\sigma_G^2=\sigma^2\Delta t\,\frac{(N+1)(2N+1)}{6N}
```

**Barriers**

- Supports up/down × in/out, plus a cash rebate paid at expiry.
- `monitoring="discrete"` checks the barrier at the simulation dates.
- `monitoring="continuous"` approximates continuous monitoring with the **Broadie–Glasserman–Kou** shift $H\,e^{\mp\beta\sigma\sqrt{\Delta t}}$, where $\beta=-\zeta(\tfrac12)/\sqrt{2\pi}\approx0.5826$.
- Benchmarks: continuous-monitoring closed forms (Merton / Reiner–Rubinstein), the BGK discrete approximation, and the vanilla price.
- Knock-in + knock-out = vanilla holds **exactly** path-by-path, and the tests assert it.
- A spot already through the barrier is handled: knock-out pays the rebate, knock-in pays the vanilla.

Example: arithmetic Asian call with S=K=100, T=1, r=5%, σ=20%, 52 weekly fixings, 20 000 paths, seed 42.

| Output | Value |
|---|---|
| `price` ± `std_error` | **5.85514 ± 0.00168** (95% CI [5.85185, 5.85843]) |
| Antithetic efficiency gain | 2.06× |
| Control-variate variance reduction | **575.8×** |
| Geometric Asian (lower bound, AM ≥ GM) | 5.63743 |
| Runtime | ~230 ms (pure Python) |

Example: down-and-out call with H=90 and continuous (BGK) monitoring, 100 steps.

| Output | Value |
|---|---|
| `price` | 8.6268 ± 0.0400 |
| Closed form | 8.6655 (z = −0.97) |
| Knock probability | 55.4% |

| Parameter | Type | Default | Domain |
|---|---|---|---|
| `product` | `asian`/`barrier`/`european` | `asian` | |
| `spot`, `strike` | number | required | > 0, with $S/K\in[10^{-4},10^4]$ |
| `time_to_maturity` | number | required | (0, 50] |
| `risk_free_rate` | number | required | [−1, 5] |
| `volatility` | number | required | (0, 5] |
| `dividend_yield` | number | 0 | [−1, 5] |
| `option_type` | `call`/`put` | `call` | |
| `paths` | int | 20 000 | [100, 2·10⁶], including antithetic paths |
| `steps` | int | 52 | [1, 5000], with paths × steps ≤ 5·10⁶ (forced to 1 for `european`) |
| `seed` | int | 42 | [0, 2⁶³−1] |
| `antithetic`, `control_variate` | bool | true | |
| `average_type` | `arithmetic`/`geometric` | `arithmetic` | Asian only |
| `barrier_type` | `up-and-out`/`up-and-in`/`down-and-out`/`down-and-in` | `down-and-out` | Barrier only |
| `barrier` | number | required for barrier | > 0 |
| `rebate` | number | 0 | ≥ 0 |
| `monitoring` | `discrete`/`continuous` | `discrete` | |

---

### 4.5 `avellaneda_stoikov_quotes`

Inventory-aware optimal quoting from Avellaneda & Stoikov (2008). The setup:

- a CARA market maker with risk aversion γ;
- the mid follows arithmetic Brownian motion, $dS = \sigma\,dW$;
- fills arrive as Poisson processes with intensity $\lambda(\delta) = A e^{-k\delta}$ at distance δ from the mid.

```math
r(s,q,t)=s-q\,\gamma\sigma^2(T-t)
\qquad
\delta^a+\delta^b=\underbrace{\gamma\sigma^2(T-t)}_{\text{inventory risk}}+\underbrace{\frac{2}{\gamma}\ln\!\Big(1+\frac{\gamma}{k}\Big)}_{\text{adverse liquidity}}
```

Quotes are $p^{b,a} = r \mp \tfrac12(\delta^a+\delta^b)$. The spread does not depend on inventory; inventory only **skews** the reservation price.

**Guéant–Lehalle–Fernandez-Tapia (2013) asymptotics.** When `order_arrival_rate` $A$ is supplied, the tool also returns the infinite-horizon closed-form quotes. These are what production market makers typically use, since they avoid the AS terminal-time artefact:

```math
\delta^{b}_{\infty}=\frac1\gamma\ln\!\Big(1+\frac\gamma k\Big)+\frac{2q+1}{2}c,\quad
\delta^{a}_{\infty}=\frac1\gamma\ln\!\Big(1+\frac\gamma k\Big)-\frac{2q-1}{2}c,\quad
c=\sqrt{\frac{\sigma^2\gamma}{2kA}\Big(1+\frac\gamma k\Big)^{1+\frac k\gamma}}
```

**Risk controls**

- `max_inventory` suppresses the side that would breach the limit, so a long position becomes sell-only.
- `tick_size` snaps quotes to the grid: bid floored, ask ceiled, never tighter than the model.
- A warning fires when inventory pressure pushes a quote through the mid, i.e. you are paying the spread to unwind.

Example: s=100, q=+3, γ=0.1, σ=2, T−t=1, k=1.5, A=140, tick 0.05.

| Output | Value |
|---|---|
| Reservation price | **98.80** (skewed down 1.20 to sell inventory) |
| Optimal spread | 1.69077 |
| AS bid / ask | 97.9546 / 99.6454 → on the tick grid: **97.95 / 99.65** |
| GLFT bid / ask | 99.1736 / 100.5161 |

> **Units.** `volatility` is **absolute** (price units per √time) and must use the same time unit as `time_remaining`, `k` and `A`. For example, 2% daily vol on a $100 stock is `volatility=2.0` with time in days.

| Parameter | Type | Default | Domain |
|---|---|---|---|
| `mid_price` | number | required | > 0 |
| `inventory` | number | 0 | Signed; long > 0 |
| `risk_aversion` | number | required | γ ∈ (0, 10⁶] |
| `volatility` | number | required | ≥ 0 (absolute) |
| `time_remaining` | number | 1 | ≥ 0 |
| `order_arrival_decay` | number | required | k > 0 |
| `order_arrival_rate` | number | none | A > 0; enables fill intensities and GLFT |
| `tick_size`, `max_inventory` | number | none | > 0 |

---

### 4.6 `order_flow_toxicity`

Volume-synchronised probability of informed trading (VPIN; Easley, López de Prado & O'Hara 2012), plus Kyle's λ, computed from a time-ordered trade tape.

**1. Trade classification** (`classification`)

| Mode | Rule for the buy fraction $f_i$ |
|---|---|
| `given` | $f_i = \mathbb 1[\text{side}_i = \text{buy}]$ |
| `tick` | Lee–Ready tick rule. Sign of the last non-zero price change; unclassifiable opening trades are split 50/50. |
| `bvc` | Bulk Volume Classification, $f_i = \Phi(\Delta P_i/\sigma_{\Delta P})$ |
| `auto` (default) | `given` if every trade carries a side, otherwise `bvc` |

> **Variant note.** Canonical VPIN applies BVC to time bars. This implementation applies it **per trade**, with $\sigma_{\Delta P}$ the sample standard deviation of trade-to-trade price changes, overridable with `price_change_std`. This is the "VPIN variant" referenced in the spec.

**2. Volume bucketing.** Volume is poured into equal-volume buckets of size $V$ (default total volume ÷ `num_buckets`). A trade that straddles a boundary is **split pro rata**, so volume is conserved exactly. The tests check that buckets × V + the partial bucket equals the total.

**3. VPIN** over a rolling window of $n$ buckets:

```math
\mathrm{VPIN}=\frac{1}{n}\sum_{\tau=1}^{n}\frac{\lvert V^{B}_{\tau}-V^{S}_{\tau}\rvert}{V}\in[0,1]
```

The tool returns the current value, the last 50 values of the rolling series, the **empirical CDF percentile** of the current value, an `alert` flag (percentile ≥ 0.9 with ≥ 10 points), and a heuristic level: low < 0.2 ≤ elevated < 0.4 ≤ high.

**4. Kyle's λ.** OLS slope of bucket price change on bucket signed volume, $\Delta P_\tau = \lambda (V^B_\tau - V^S_\tau) + \varepsilon_\tau$, with $R^2$. Requires ≥ 3 buckets.

Example: a synthetic tape of 4 000 trades with true impact λ = 0.002 per share plus noise.

| Output | Value |
|---|---|
| Estimated λ | **0.002008** ($R^2$ = 0.975) |
| VPIN | 0.174, at the 94th percentile of its own history |

| Parameter | Type | Default | Domain |
|---|---|---|---|
| `trades` | array | required | 2–200 000 entries of `[price, volume]`, `[price, volume, side]` or `{price, volume, side}`; side = `buy`/`sell`/`+1`/`−1` |
| `classification` | `auto`/`bvc`/`tick`/`given` | `auto` | |
| `bucket_volume` | number | total ÷ `num_buckets` | ≤ 10⁵ buckets |
| `num_buckets` | int | 50 | |
| `window` | int | 50 | VPIN window in buckets |
| `price_change_std` | number | sample std | BVC σ |

---

### 4.7 `optimal_stopping`

**Secretary problem** (`problem="secretary"`). With $n$ candidates in random order, reject the first $r$, then accept the first candidate who beats everyone seen so far. The success probability is

```math
P_n(r)=\frac{r}{n}\sum_{i=r}^{n-1}\frac1i,\qquad P_n(0)=\frac1n,\qquad r^*=\arg\max_r P_n(r)\ \xrightarrow{n\to\infty}\ n/e,\ \ P_n(r^*)\to 1/e
```

The result is exact over ℚ for $n\le500$ (n = 10 gives $r^*=3$ and $P = 3349/8400 \approx 0.39869$) and a float sum up to $n = 10^7$. Nearby cutoffs are returned so you can see how flat the optimum is.

**IID offers without recall** (`problem="iid_offers"`). Offers $X_1, X_2, \dots$ are iid from a discrete, uniform or normal distribution. Each draw costs $c$, and continuing is discounted by δ.

*Finite horizon $H$* (backward induction):

```math
V_1=\mathbb E[X]-c,\qquad V_k=\mathbb E\big[\max(X,\ \delta V_{k-1})\big]-c,\qquad \text{accept at a draw with } k-1 \text{ left iff } X\ge \delta V_{k-1}
```

The tool returns per-draw thresholds, the expected number of draws, and the value with optional participation, $\max(V_H, 0)$. Discrete distributions are solved **exactly**: a die with 3 rolls gives $V = 14/3$ and thresholds $17/4$, $7/2$, then accept anything. When `horizon` > 200 or there are more than 200 outcomes, the solver switches to floats to keep rational denominators bounded.

*Infinite horizon* (requires $c>0$ or $\delta<1$): the reservation value is the unique fixed point

```math
t=\delta\big(\mathbb E[\max(X,t)]-c\big)\quad\Longleftrightarrow\quad\mathbb E[(X-t)^+]=c\ \ (\delta=1)
```

It is solved by bisection, since $h(t)=\delta(\mathbb E\max(X,t)-c)-t$ is strictly decreasing. The normal case uses the closed form $\mathbb E[\max(X,t)] = t\Phi(z)+\mu(1-\Phi(z))+\sigma\varphi(z)$. The tool also returns $\mathbb P(\text{accept})$, the expected number of draws and the fixed-point residual.

Example: U[0,1] with c = 0.02 gives $t = 1-\sqrt{2c} = 0.8$ and an expected 5 draws.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `problem` | `secretary`/`iid_offers` | `iid_offers` | |
| `n` | int | none | Secretary: [1, 10⁷] |
| `distribution` | `discrete`/`uniform`/`normal` | `discrete` | |
| `outcomes`, `probabilities` | array | none, uniform | Discrete: ≤ 10 000 values; numbers or `"p/q"` |
| `low`, `high` | number | 0, 1 | Uniform |
| `mean`, `std` | number | 0, 1 | Normal |
| `horizon` | int | 0 | 0 = infinite; ≤ 100 000 |
| `cost_per_draw` | number/str | 0 | ≥ 0 |
| `discount` | number/str | 1 | (0, 1] |

---

### 4.8 `matrix_game_solver`

Exact mixed-strategy Nash equilibrium of a two-player zero-sum game with payoff matrix $A\in\mathbb Q^{m\times n}$, where entries are payoffs to the maximising row player.

**LP formulation.** Shift the matrix to be strictly positive, $B = A + s\mathbf 1\mathbf 1^\top$ with $s = \max(0, 1-\min A)$, then solve

```math
\max_{y\ge0}\ \mathbf 1^\top y\quad\text{s.t.}\quad By\le\mathbf 1
```

with a **dense tableau simplex over `fractions.Fraction`** and **Bland's anti-cycling rule**. The column strategy is $q = y/\mathbf 1^\top y$. The row strategy comes from the optimal duals, read off the slack reduced costs. The game value is $v = 1/\mathbf 1^\top y - s$.

**Certificate.** By the minimax theorem / LP duality, optimality holds iff

```math
\min_j\,(p^\top A)_j \;=\; v \;=\; \max_i\,(Aq)_i ,
```

and the tool checks this **exactly in rational arithmetic**, returning `verified_equilibrium: true` together with both payoff vectors. It also reports pure maximin and minimax values, all pure saddle points, supports, and whether the game is fair ($v = 0$).

Example: $A = \begin{pmatrix}2&-1\\-1&1\end{pmatrix}$ gives $v = 1/5$ and $p = q = (2/5,\ 3/5)$, matching the 2×2 closed form $p_1=\frac{d-c}{a-b-c+d}$.

> When the equilibrium is not unique, the tool returns one optimal vertex.

| Parameter | Type | Default | Domain |
|---|---|---|---|
| `payoff_matrix` | array of rows | required | ≤ 25×25; numbers or `"p/q"` strings |
| `row_labels`, `col_labels` | string[] | `R1…`, `C1…` | |

---

### 4.9 `ev_calculator`

Exact solvers for the discrete-probability puzzles that come up in trading interviews. Select one with `puzzle`:

| `puzzle` | Model | Example (exact) |
|---|---|---|
| `custom` | Finite lottery with a cost. Returns EV, variance, σ, Sharpe per play, P(win/loss/push) and a verdict. | Fair die minus $3 → EV = 1/2 |
| `bayes` | Posterior over hypotheses after $k$ iid observations; predictive probability that the next observation repeats | 1/1000 coins double-headed, 10 heads → 1024/2023 |
| `dice_reroll` | Up to $k$ rolls, keep or reroll, optional fee per roll; backward induction | 2 rolls → 17/4; 3 → 14/3 |
| `dice_max` | $\mathbb E[\max]$ and $\mathbb E[\min]$ of $k$ dice via tail sums | 2d6: E[max] = 161/36 |
| `card_red_black` | Draw without replacement, +1 red / −1 black, stop any time; DP $V(r,b)$ | 2R/2B → 2/3; full deck ≈ 2.62448 |
| `coin_pattern` | Expected waiting time for an H/T pattern (prefix-automaton Markov chain, exact Gaussian elimination over ℚ, Conway leading-number cross-check), biased coins supported | HTH → 10, HHTHH → 38 |
| `coin_race` | Penney's game: P(A first), P(B first), expected length | THH beats HHT with 3/4 |
| `gamblers_ruin` | Hit-target probability and expected duration, numerically stable in log space | — |
| `coupon_collector` | $\mathbb E$ and variance of draws to collect $m$ of $n$ coupons | n = 6 → 147/10 |
| `kelly` | Kelly fraction $f^*=p/a-q/b$ for net odds $b$ and loss fraction $a$, half-Kelly, log-growth | p = 0.6 at evens → 1/5 |

---

### 4.10 `order_book_metrics`

Snapshot analytics on L2 ladders. Input order does not matter, duplicate price levels are aggregated, and zero-size levels are dropped.

- mid, spread (absolute, bps, ticks);
- **microprice** $\dfrac{P^b Q^a + P^a Q^b}{Q^a + Q^b}$ and its deviation from mid, with a direction signal;
- top-of-book imbalance $\dfrac{Q^b-Q^a}{Q^b+Q^a}$;
- depth volumes, VWAPs and a depth-weighted mid over $N$ levels;
- crossed / locked book detection;
- optional walk-the-book execution for `impact_quantity`: average fill price, worst price, slippage in price and bps, unfilled size.

---

### 4.11 `almgren_chriss_liquidation`

Optimal execution of a large block over $N$ discrete intervals (Almgren & Chriss, 2000). The trader minimises expected implementation shortfall plus $\lambda\cdot$variance.

**Model**

- The price follows an arithmetic random walk with linear **permanent** impact $g(v)=\gamma v$ (Kyle's λ; `kyle_lambda` is an alias).
- Linear **temporary** impact is $h(v)=\varepsilon\,\mathrm{sgn}(v)+\eta v$, where $v = n_k/\tau$ is the trading rate.

```math
S_k=S_{k-1}+\sigma\sqrt\tau\,\xi_k-\gamma n_k,\qquad \tilde S_k=S_{k-1}-\varepsilon-\eta\,\frac{n_k}{\tau},\qquad \tau=T/N
```

```math
\mathbb E[\text{IS}]=\tfrac12\gamma X^2+\varepsilon\sum_k|n_k|+\frac{\tilde\eta}{\tau}\sum_k n_k^2,\qquad
\mathrm{Var}[\text{IS}]=\sigma^2\tau\sum_{k=1}^{N}x_k^2,\qquad \tilde\eta=\eta-\tfrac12\gamma\tau
```

**Solution.** Minimising $U=\mathbb E+\lambda\mathrm{Var}$ over the holdings $x_1,\dots,x_{N-1}$ gives the discrete Euler–Lagrange equation $(x_{k-1}-2x_k+x_{k+1})/\tau^2=\tilde\kappa^2x_k$, where $\tilde\kappa^2=\lambda\sigma^2/\tilde\eta$. Its solution is

```math
x_k=X\,\frac{\sinh\big(\kappa(T-t_k)\big)}{\sinh(\kappa T)},\qquad \cosh(\kappa\tau)=1+\tfrac12\tilde\kappa^2\tau^2 .
```

**Numerics**

- $\kappa$ uses the cancellation-free $\mathrm{acosh}(1+x)=\log1p\!\big(x+\sqrt{x(x+2)}\big)$.
- The sinh ratio is evaluated as $e^{-\kappa t}\,\mathrm{expm1}(-2\kappa(T-t))/\mathrm{expm1}(-2\kappa T)$, so it cannot overflow for urgent trades.
- $\lambda=0$ reduces exactly to TWAP.
- Inputs with $\tilde\eta\le0$ are rejected: the problem is ill-posed there and the "optimal" schedule would be arbitrage.

**Outputs**

- the schedule (holdings, trades, rates, % executed), expected cost, std and utility, also in bps of arrival notional if `spot` is given;
- the cost split into permanent, fixed and temporary;
- **TWAP** and **immediate-execution** benchmarks;
- a 14-point **efficient frontier**;
- an **optimality certificate**: Euler–Lagrange residual, plus a check that the optimum beats TWAP and immediate execution.

Example from Almgren & Chriss (2000): X = 10⁶ shares, S₀ = $50, 30% annual vol (σ = 0.95 \$/√day), T = 5 days, N = 5, ε = 0.0625, η = 2.5·10⁻⁶, γ = 2.5·10⁻⁷, λ = 10⁻⁶.

| Output | Value |
|---|---|
| κ | **0.6063 / day** (paper: ≈ 0.6) |
| Holdings $x_0..x_5$ | 1,000,000 · 542,381 · 290,295 · 148,216 · 62,303 · 0 |
| E[IS] ± sd (optimal) | $910,478 ± $603,215 |
| E[IS] ± sd (TWAP) | $662,500 ± $1,039,230 |
| Euler–Lagrange residual | 1.4·10⁻¹⁶ |

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `shares` | number | required | X; > 0 sell, < 0 buy |
| `horizon` | number | required | T in time units (e.g. days) |
| `periods` | int | 10 | N ∈ [1, 10 000] |
| `volatility` | number | required | **absolute** σ (\$/share/√time) |
| `permanent_impact` / `kyle_lambda` | number | 0 | γ ($/share per share) |
| `temporary_impact` | number | required | η > γτ/2 |
| `fixed_cost` | number | 0 | ε (half-spread + fees) |
| `risk_aversion` | number | required | λ ≥ 0 (1/$) |
| `spot` | number | none | Enables bps reporting |
| `max_schedule_rows` | int | 60 | Schedule rows returned |

---

### 4.12 `longstaff_schwartz_american`

Least-squares Monte Carlo (Longstaff & Schwartz, 2001) for American and Bermudan options on GBM.

**Algorithm**

1. Simulate GBM paths with antithetic pairing.
2. Set the cash flow at maturity to the payoff.
3. Step backwards. At each exercise date $t_i$, regress the discounted realised cash flows of **in-the-money** paths on basis functions of $x = S/K$:

```math
\hat C(t_i,x)=\sum_{j=0}^{d}\beta_j\,\psi_j(x),\qquad \psi_0=1,\ \ \psi_{j+1}(x)=e^{-x/2}L_j(x)\ \text{(weighted Laguerre)}
```

   The Laguerre polynomials come from the recurrence $L_{n+1}=\big((2n+1-x)L_n-nL_{n-1}\big)/(n+1)$. `basis` can also be `laguerre_unweighted` or `monomial`, with `degree` 1–6.

4. Exercise where intrinsic value > $\hat C$, and replace that path's cash flow.

**Two estimators** (both are reported, so the user can see the bias bracket):

| Estimator | Construction | Bias |
|---|---|---|
| In-sample | Regression and valuation on the same paths | High (the regression sees the future) |
| **Out-of-sample** (default) | Fitted policy $\{\beta_{t_i}\}$ applied to an **independent** path set | **Low**: a feasible stopping rule cannot beat the optimal one |

**Variance reduction and error bars**

- A European control variate (Black–Scholes mean) is applied to both estimators.
- Its measured gain is modest for American puts (≈1.1× ITM, 1.3× ATM, 2.2× OTM) because early exercise decorrelates the payoffs. It is reported, not assumed.
- `std_error` is **conditional on the fitted policy**. Across seeds, regression noise adds roughly as much again (measured: per-run SE 0.0058 vs seed-to-seed sd 0.0081). The output states this explicitly.

**Benchmarks**

- An apples-to-apples **Bermudan CRR**: a lattice with exercise allowed only every `⌈400/N⌉` steps, via `exercise_every` on the shared `_crr_tree` kernel.
- The continuous-exercise American CRR.
- A z-score against the Bermudan reference.

The exercise boundary comes from scanning, starting at the strike, for the first crossing of intrinsic over $\hat C$, where the regression is best populated.

**Backends.** With NumPy installed (`backend="auto"`), paths are generated as one $(N{+}1)\times M$ matrix (cumulative sum and `exp` in place) and the regressions use `numpy.linalg.lstsq`. The budget is 5·10⁶ path-dates. Without NumPy, a pure-Python implementation (column-wise `array('d')` storage, column-scaled normal equations solved by Cholesky with a ridge fallback) gives identical semantics, with a budget of 10⁶ path-dates.

Validation (Longstaff–Schwartz Table 1: S = 36, K = 40, r = 6%, σ = 20%, T = 1, 50 dates; the paper reports LSM 4.472):

| | Value |
|---|---|
| Bermudan CRR (400 / 1000 / 2000 steps) | 4.4792 / 4.4784 / **4.4780** |
| Python backend, 20k paths | 4.4940 ± 0.0131 (z = +1.1), in-sample 4.4713, 1.6 s |
| NumPy backend, 100k paths | 4.4859 ± 0.0058 (z = +1.1), 0.7 s |
| **12 seeds × 100k (out-of-sample mean)** | **4.4748 ± 0.0024**, low-biased by 0.003 as theory predicts |
| Boundary $S^*(t)$ at t ≈ 0.34 / 0.56 / 0.76 / 0.98 | LSM 33.95 / 34.53 / 35.21 / 37.16 vs CRR 33.37 / 34.22 / 35.10 / 37.87 |

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `spot`, `strike`, `time_to_maturity`, `risk_free_rate`, `volatility`, `dividend_yield` | number | required (q = 0) | As in `black_scholes` |
| `option_type` | `call`/`put` | `put` | |
| `exercise_dates` | int | 50 | [1, 500] |
| `paths` | int | 50 000 (NumPy) / 20 000 (Python) | Includes antithetic paths |
| `basis` | `laguerre`/`laguerre_unweighted`/`monomial` | `laguerre` | |
| `degree` | int | 3 | [1, 6] |
| `seed` | int | 42 | Out-of-sample paths use an independent stream |
| `antithetic`, `out_of_sample`, `control_variate`, `benchmark` | bool | true | |
| `boundary_points` | int | 10 | |
| `backend` | `auto`/`numpy`/`python` | `auto` | `numpy` without NumPy falls back with a warning |

---

### 4.13 `black_litterman_optimizer`

Black & Litterman (1992), in the He & Litterman (1999) formulation.

**Prior.** Reverse optimisation of market weights gives equilibrium excess returns $\pi=\delta\Sigma w_{\text{mkt}}$. The prior is $\mu\sim\mathcal N(\pi,\tau\Sigma)$. Use $\delta$ = `risk_aversion` (default 2.5), or derive it from `market_excess_return` / $w^\top\Sigma w$.

**Views.** $P\mu=q+\epsilon$ with $\epsilon\sim\mathcal N(0,\Omega)$. Views can be absolute (`{"asset": "EM", "return": 0.10}`) or relative (`{"assets": {"US": 1, "EU": -1}, ...}`). Confidence $c\in(0,1]$ maps to $\omega_k=\frac{1-c}{c}\,\tau\,p_k\Sigma p_k^\top$, so $c = 1$ is an exact view and $c = \tfrac12$ is He–Litterman's $\omega=\tau p\Sigma p^\top$. An explicit `omega` overrides the confidence.

**Posterior.** Computed in the form that stays valid as $\Omega\to0$ (no $\Omega^{-1}$), using Cholesky solves throughout:

```math
\mu_{BL}=\pi+\tau\Sigma P^\top\big(P\tau\Sigma P^\top+\Omega\big)^{-1}(q-P\pi),\qquad
\Sigma_{\text{post}}=\Sigma+\tau\Sigma-\tau\Sigma P^\top\big(P\tau\Sigma P^\top+\Omega\big)^{-1}P\tau\Sigma
```

**Portfolios returned**

| Portfolio | Definition |
|---|---|
| Unconstrained (prior cov) | $(\delta\Sigma)^{-1}\mu_{BL}=w_{\text{mkt}}+P^\top\Lambda$. Tilts lie only along the view portfolios; $\Lambda$ is reported per view. |
| Unconstrained (posterior cov) | $(\delta\Sigma_{\text{post}})^{-1}\mu_{BL}$; with no views this equals $w_{\text{mkt}}/(1+\tau)$ |
| Fully invested | Closed form with a budget multiplier; shorts allowed |
| **Long-only** with `max_weight` | FISTA projected gradient onto the capped simplex (exact bisection projection), then an **exact active-set KKT solve** on the identified free set. KKT violation is reported (~10⁻¹⁷ in tests). |

Per-view diagnostics: equilibrium-implied value $p\pi$, posterior value $p\mu$, and the **fraction of the gap closed**. That fraction is exactly 0.5 at $c = \tfrac12$ and exactly 1 at $c = 1$ (tested).

Example (4 regions, 2 views, `max_weight` 0.5): the EM view at 10% with c = 0.6 moves EM from π = 7.18% to μ = 8.84% (58.8% of the gap). The long-only solution is US 0.50 / EU 0.168 / JP 0.096 / EM 0.237 after 233 iterations, with KKT violation 9·10⁻¹⁸. The posterior matches the independent precision form $[(\tau\Sigma)^{-1}+P^\top\Omega^{-1}P]^{-1}[(\tau\Sigma)^{-1}\pi+P^\top\Omega^{-1}q]$ to **1.4·10⁻¹⁷**.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `assets` | string[] | required | 1–60 distinct names |
| `covariance` **or** `volatilities` + `correlation` | matrix / arrays | required | Annualised; must be symmetric positive definite |
| `market_weights` | number[] | required | Normalised (with a warning) if they don't sum to 1 |
| `risk_aversion` / `market_excess_return` | number | 2.5 / none | Mutually exclusive |
| `tau` | number | 0.05 | |
| `views` | object[] | [] | `asset` or `assets`, plus `return`, `confidence` or `omega`, and optional `label` |
| `risk_free_rate` | number | 0 | For reporting total returns |
| `max_weight` | number | 1 | Long-only cap; must satisfy max_weight × n ≥ 1 |
| `use_posterior_covariance` | bool | true | Covariance used by the constrained optimisers |

---

### 4.14 `jane_street_ev_trainer`

An interactive, **stateless** interview simulator. A puzzle id such as `JSQR-AU-first_price-3-2026` deterministically regenerates the puzzle from (category, template, difficulty, seed). Generation uses a SHA-512-seeded `random.Random`, so it does not depend on `PYTHONHASHSEED` and needs no server-side storage.

| `action` | Behaviour |
|---|---|
| `generate` | Returns the prompt (priced as a trade, with a bankroll), the puzzle id and the answer contract. **No solution.** |
| `check` | Grades `verdict` (**must** contain exactly one `[TAKE]`/`[PASS]`/`[HEDGE]` tag), `ev`, `size` and the `proof` text, out of 100, then reveals the full solution. |
| `solve` | Reveals the exact solution: the 5-part proof, verdict line, risk numbers and hedge details. |

**Puzzle families** (all solved in exact rational arithmetic, including the full payoff distribution under the optimal policy):

| Category | Templates | Solution method |
|---|---|---|
| `conditional_expectation` | 2–3 dice (d6/d8), n coin flips, 2–3 cards from a 52-card deck, each under random conditioning events | Enumeration over the conditional space; multivariate hypergeometric for cards |
| `dice_stopping` | Re-roll game (k rolls, re-roll fee, payoff f or f²); Pig ("roll a 1, lose it all") | Backward induction; one-step lookahead proved optimal by closedness of the stopping set |
| `card_stopping` | Red/black deck with asymmetric payoffs; "call the next red card" | DP over (r, b); martingale / optional stopping (no strategy beats R/(R+B)) |
| `auction` | Acquiring-a-company (winner's curse, value multiplier k); first-price sealed bid vs a uniform rival with entry fee | Closed-form EV(b) and an integer argmax for the optimal bid |

**Verdict rule** (the same contract the persona enforces):

| Condition | Verdict | Size |
|---|---|---|
| EV ≤ 0 | `[PASS]` | 0 |
| EV > 0 and the stated variance limit is breached | `[HEDGE]` | Minimum-variance ratio β = Cov(Y,H)/Var(H) in the offered fair-priced hedge instrument |
| Otherwise | `[TAKE]` | **Half-Kelly** contracts: n* solves $\sum_i p_i x_i/(B+n x_i)=0$ on the stated bankroll |

**Scoring**

| Component | Points | Rule |
|---|---|---|
| Verdict | 40 | Correct tag |
| EV | 30 | Exact fraction 30 · correct to rounding 27 · within 1% 15 |
| Size | 20 | Half-Kelly within 10% (full Kelly gets 10) · hedge ratio within 1% |
| Proof structure | 10 | State space, transitions, payoffs, an EV equation and a verdict tag |

Independent checks in the test suite:

- re-roll values equal `ev_calculator` exactly;
- red/black values equal `ev_calculator` exactly;
- Pig d6 gives V(0) = 492303203/60466176 ≈ 8.14179, the classic hold-at-20 value, confirmed by simulation;
- card puzzles are brute-forced over every C(52, d) hand;
- auctions are brute-forced over every rival bid;
- "next red" is checked by simulating a non-trivial stopping strategy;
- every TAKE size satisfies the Kelly first-order condition.

| Parameter | Used by | Notes |
|---|---|---|
| `action` | all | `generate` (default) / `check` / `solve` |
| `category` | generate | `random` (default), `conditional_expectation`, `dice_stopping`, `card_stopping`, `auction` |
| `template` | generate | Optional: `dice`, `coins`, `cards`, `reroll`, `pig`, `red_black`, `next_red`, `acquire`, `first_price` (gated by difficulty) |
| `difficulty` | generate | 1–3 (default 2) |
| `seed` | generate | Reproducible puzzle (default random; echoed in `puzzle_id`) |
| `puzzle_id` | check, solve | Id returned by `generate` |
| `verdict`, `ev`, `size`, `proof` | check | Tag string, EV per contract (an exact fraction scores highest), contracts or hedge ratio, derivation text |

Live example: in a Hermes session the agent generated `JSQR-AU-first_price-3-2026` ($100 value, rival uniform on 0..50, $45 fee, bid $31). It derived EV = (21/34)·69 − 45 = −81/34, answered `[PASS]`, and the grader returned **100/100**.

---

## 5. System prompt & persona engineering

Prompting a model to "be rigorous" does not reliably work. The plugin instead enforces a *format*, because a format can be checked. It works on three layers, from always-on context to post-hoc audit.

### 5.1 Layer 1 — the system-prompt section (persona + answer contract)

`register_system_prompt_section("waterloo-jane-street-qr.persona", PERSONA_PROMPT, max_chars=4000)` injects 2,406 characters (Hermes caps a section at 4,000) into the system prompt at position `after_memory`.

Hermes freezes the rendered section **once per session**, so it never invalidates the provider's prompt cache. This is the main reason it is a system-prompt section rather than per-turn injection.

The prompt establishes:

1. **Persona.** A quantitative researcher on a Jane Street / Citadel tier desk with a Waterloo Math/CS, IMO, ICPC and Putnam background. Rigour, calibration and P&L discipline are non-negotiable.
2. **A mandatory answer contract for every response**, compressible to one line per part for non-quant questions but never dropped:

   | # | Section | Required content |
   |---|---|---|
   | 1 | **State space** | Random variables / states, information set, controls; explicit assumptions (independence, replacement, ℚ vs ℙ) |
   | 2 | **Transition probabilities** | Dynamics between states: tree, Markov chain or SDE, with exact fractions where possible |
   | 3 | **Terminal payoffs** | Payoff in every terminal state, net of costs, fees, slippage and financing |
   | 4 | **EV** | $EV=\sum_i p_i x_i$ = exact ≈ decimal (units), plus a risk number (σ, P(loss) or worst case); explicitly whether $EV>0$ |
   | 5 | **Verdict** | Exactly one tag with exact sizing: `[TAKE] <size>` (EV > 0 and risk fits, default half-Kelly), `[PASS]` (EV ≤ 0 or edge not demonstrable), or `[HEDGE] <instrument> <ratio>` (EV > 0 but variance, tail or Greek exposure must be offset) |

3. **Method rules**
   - Derive with conditioning, linearity, symmetry, backward induction, martingales / optional stopping, generating functions or LP duality.
   - Never use mental arithmetic for anything a tool computes. All ten tools are named, and Monte Carlo results must be quoted with standard errors.
   - Verify two ways: limiting cases, bounds, parity, simulation vs closed form.
   - Keep edge separate from sizing.
   - When information is missing, name the single quantity that flips the sign of EV and give its break-even value instead of guessing.

### 5.2 Layer 2 & 3 — lifecycle hooks

| Hook | Behaviour |
|---|---|
| `pre_llm_call` | Returns `{"context": DESK_REMINDER}`. Hermes appends this one-line protocol restatement to the current user message, as ephemeral turn context that never touches the system prompt. It counters long-context drift: the contract is restated next to the question it applies to. |
| `transform_llm_output` | Runs `audit_response(text)` on the final answer. If it lacks an explicit EV (regex `\bEV\b\s*(=\|≈\|~\|:\|>\|<\|≥\|≤)`) or a verdict tag (`\[(TAKE\|PASS\|HEDGE)\]`), it appends a footer: *"⚠ Desk compliance (waterloo-jane-street-qr): this answer is missing … Treat it as unvetted: no position should be taken on it."* Compliant answers are returned unchanged (`None`). The footer is applied before the reply is persisted, so the user sees exactly what is stored. |

Condensed excerpt of a compliant answer from a live Hermes session. The question asked whether to buy an American put offered at $5.90:

```
4. EV (per share): American (tree) 6.0896, CV-corrected 6.0916, European (BS) 5.5735 → early-exercise premium 0.518
   EV = 6.0916 − 5.90 = +0.1916 per share (+$19.16 per 100-share contract). EV > 0.
   Risk: vega 0.375 per vol point → $5.90 implies ≈19.5% vol; delta −0.411.
5. Verdict: [HEDGE] long 0.411 shares per put (411 shares per 10 contracts), rebalanced as delta changes.
```

### 5.3 Enforcement modes

| `enforcement` | System prompt | Per-turn reminder | Output audit / footer |
|---|---|---|---|
| `strict` (default; also used for unknown values) | ✓ | ✓ | ✓ |
| `reminder` | ✓ | ✓ | ✗ |
| `off` | ✓ | ✗ | ✗ |

The mode is read through `ctx.get_config("enforcement")` on every hook call, so changing it takes effect without reloading the plugin.

**Backwards compatibility.** On hosts without `register_system_prompt_section`, the plugin falls back to injecting the full persona through `pre_llm_call` on every turn.

**Known trade-off.** The audit is **syntactic**. It proves an EV line and a verdict tag exist, not that they are correct; correctness comes from the tools. In `strict` mode the contract also applies to non-quant replies (by design: "every response"). Use `reminder` for general-purpose sessions.

---

## 6. Programmatic usage & API contract

### 6.1 Handler signature

Every tool is registered as

```python
handler(args: dict | None, **kwargs) -> str   # strict JSON; never raises
```

`kwargs` receives Hermes runtime metadata (`task_id`, `session_id`, …), which the handlers ignore.

Registration, in `register(ctx)`:

```python
ctx.register_tool(name=..., toolset="waterloo_qr", schema=<JSON Schema>, handler=<fn>,
                  description=<first line of schema description>, emoji=...)
ctx.register_system_prompt_section("waterloo-jane-street-qr.persona", PERSONA_PROMPT, max_chars=4000)
ctx.register_hook("pre_llm_call", pre_llm_call)
ctx.register_hook("transform_llm_output", transform_llm_output)
```

### 6.2 Response envelope

```jsonc
// success
{"ok": true, "...tool-specific fields...": "...", "warnings": ["optional, unit/model caveats"]}

// any failure: validation, numerics, or an unexpected exception
{"ok": false, "error": "'spot' must be > 0, got -1.0"}
```

| Failure class | Error prefix | Logged? |
|---|---|---|
| `ToolInputError` (bad / out-of-domain input) | Message only | No |
| `ZeroDivisionError`, `OverflowError` | `numerical error: …` | No |
| Anything else | `internal error: <Type>: …` | Yes (`logger.exception`) |

The fuzz suite asserts that the third class **never** occurs.

Exact rationals are serialised as `{"decimal": 0.39869, "exact": "3349/8400"}`, and fraction strings like `"1/6"` are accepted wherever a probability or payoff is expected.

### 6.3 Calling the engines from Python (outside Hermes)

The directory name contains hyphens, so load it by path:

```python
import importlib.util, json, pathlib

path = pathlib.Path("waterloo-jane-street-qr/__init__.py")
spec = importlib.util.spec_from_file_location("wjsqr", path)
qr = importlib.util.module_from_spec(spec); spec.loader.exec_module(qr)

call = lambda tool, **kw: json.loads(getattr(qr, tool)(kw))

# European pricing + implied vol
bs = call("black_scholes", spot=100, strike=100, time_to_maturity=1,
          risk_free_rate=0.05, volatility=0.2, option_type="call", market_price=12.0)
bs["call"]["price"], bs["call"]["delta"]              # 10.450584, 0.636831
bs["implied_vol_solve"]["implied_volatility"]         # 0.241117

# American put + exercise boundary
am = call("binomial_tree_american", spot=100, strike=100, time_to_maturity=1,
          risk_free_rate=0.05, volatility=0.2, option_type="put", steps=1000)
am["american_price_control_variate"], am["early_exercise_premium"]   # 6.0916, 0.5181

# Heston smile
h = call("heston_price", spot=100, strike=100, time_to_maturity=1, risk_free_rate=0.0,
         v0=0.0175, kappa=1.5768, theta=0.0398, vol_of_vol=0.5751, rho=-0.5711,
         strikes=[80, 100, 120])
[(s["strike"], s["implied_vol"]) for s in h["smile"]]  # [(80, .2026), (100, .1451), (120, .1278)]

# Variance-reduced Asian
mc = call("monte_carlo_engine", product="asian", spot=100, strike=100, time_to_maturity=1,
          risk_free_rate=0.05, volatility=0.2, paths=20_000, steps=52, seed=42)
mc["price"], mc["std_error"]                           # 5.85514, 0.00168

# Market-making quotes with inventory
mm = call("avellaneda_stoikov_quotes", mid_price=100, inventory=3, risk_aversion=0.1,
          volatility=2, order_arrival_decay=1.5, order_arrival_rate=140, tick_size=0.05)
mm["reservation_price"], mm["quotes_on_tick_grid"]    # 98.8, {'bid': 97.95, 'ask': 99.65, ...}

# Exact game theory and stopping
call("matrix_game_solver", payoff_matrix=[[2, -1], [-1, 1]])["game_value"]   # {'exact': '1/5', ...}
call("optimal_stopping", problem="secretary", n=10)["success_probability"]   # {'exact': '3349/8400', ...}
call("ev_calculator", puzzle="coin_race", pattern_a="THH", pattern_b="HHT")["p_a_first"]  # 3/4
```

### 6.4 Through Hermes

```bash
hermes chat -q "Price an arithmetic Asian call S=K=100, T=1, r=5%, vol=20%, weekly fixings, \
with monte_carlo_engine. Quote the std error. Fair at 5.70?"
```

The model calls `monte_carlo_engine`, answers in the five-part contract and ends with a verdict tag. If it doesn't, the compliance footer says so.

---

## 7. Test suite

`waterloo-jane-street-qr/tests/test_plugin.py` has **81 tests**. It is self-contained: it loads the plugin by path and needs no Hermes runtime. It runs under pytest or as a plain script.

```bash
cd waterloo-jane-street-qr

# pytest (installed in the Hermes venv)
~/.hermes/hermes-agent/venv/bin/python -m pytest -q tests/        # 81 passed (79 + 2 skipped if NumPy is absent)

# standalone, no third-party deps
python3 tests/test_plugin.py                                       # 79 passed, 2 skipped (NumPy-only tests)

# with NumPy available, the NumPy-backend tests run too
/path/to/venv-with-numpy/bin/python tests/test_plugin.py          # 81 passed, 0 skipped

# host-contract checks
hermes plugins doctor --ci .
hermes plugins validate .
```

| Category | What is asserted |
|---|---|
| **Registration & manifest** | All 14 tools registered with valid JSON Schemas (`required` ⊆ `properties`, every property typed and described). `plugin.yaml` `provides_tools` / `provides_hooks` match the registrations exactly. Legacy-host fallback works. |
| **Persona & enforcement** | Prompt ≤ 4000 chars, contains every contract element and names every tool. All four enforcement modes (including an invalid value → strict) behave per [§5.3](#53-enforcement-modes). `audit_response` accepts and rejects the right patterns. |
| **Black–Scholes** | Hull reference values. Every Greek, including ρ with dividends, matches central finite differences. IV round-trip to 10⁻⁶. Arbitrage-violating quotes are rejected. |
| **Binomial** | Hull's 5-step example (4.4885). 1000-step convergence to the literature value. American call with q = 0 has zero premium and no boundary, and gains a premium with q = 10%. Boundary is monotone and ≤ K. Deep-ITM put exercises immediately. Greeks are consistent with BS. Rejects $p\notin(0,1)$. |
| **Heston** | Fang–Oosterlee benchmark to 10⁻⁶. **BS limit** as η → 0, with the error decaying as $O(\eta^2)$ when ρ = 0. Parity to 10⁻⁹. Δ_call − Δ_put = $e^{-qT}$. Skew sign follows ρ. **Agrees with an independent full-truncation Euler Monte Carlo** of the SDE. |
| **Monte Carlo** | European vs BS within 4 SE (call and put). Geometric Asian vs closed form. Arithmetic Asian bounded by geometric and European, with CV variance reduction > 50×. **Knock-in + knock-out = vanilla exactly.** Closed-form barrier in/out parity for all 8 type/strike combinations. MC vs continuous closed form for 12 barrier configurations. Breached barriers and rebates. Seed determinism. Antithetic toggle. |
| **Avellaneda–Stoikov** | Paper parameters (spread 1.69077). Reservation skew is linear in q while the spread is q-invariant. τ = 0 limit. GLFT skew identity $\delta^a-\delta^b=-2qc$. Tick snapping, inventory caps, intensities. |
| **VPIN** | All-buys gives VPIN = 1; alternating gives 0. Volume conservation across split buckets. BVC and tick-rule direction. **Recovers a planted Kyle λ** (0.002 ± 4·10⁻⁴, R² > 0.8). |
| **Optimal stopping** | Secretary exact (n = 10) and the 1/e asymptote (n = 10⁵). Uniform recursion 1/2, 5/8, 89/128 with thresholds and expected draws. Exact die value 14/3. Closed-form reservation values for uniform with cost, uniform with discount, and normal with cost. |
| **Matrix games** | Matching pennies, RPS, 2×2 closed form, saddle points, fractional entries, and **40 random games up to 7×7 with the duality certificate re-verified independently** in the test. |
| **Almgren–Chriss** | κ and trajectory vs the independent sinh closed form (paper example: κ = 0.6063). TWAP cost exactly 662,500. λ = 0 gives TWAP. Urgency is monotone (E↑, V↓). The optimum beats 200 random perturbations. **A direct simulation of the AC price process reproduces E and Var.** Buy/sell symmetry, the `kyle_lambda` alias, and rejection of $\tilde\eta \le 0$. |
| **Longstaff–Schwartz** | Python backend vs the converged Bermudan CRR (LS Table 1). NumPy backend with **low bias across 6 seeds**. Backends agree. Boundary within 1.5 of CRR. An American call with q = 0 equals the European (< 2% early exercise). Basis choices agree. Determinism. Budget errors. Graceful fallback when NumPy is requested but missing. |
| **Black–Litterman** | **Matches the independent precision-form posterior** (mean and covariance) to 10⁻¹². No-view identities ($w_{\text{mkt}}$, $w_{\text{mkt}}/(1+\tau)$). Gap closed is monotone in confidence (= ½ at c = ½, = 1 at c = 1). Tilt lies in span($P^\top$). Long-only KKT plus brute-force dominance over 3,000 feasible portfolios. Error paths (non-PD, unknown asset, infeasible cap, dependent certain views). |
| **EV trainer** | Deterministic generation that hides the answer. A 300-puzzle sweep: valid probability measures, the verdict rule, the Kelly first-order condition, all three verdicts occurring. Brute-force cross-checks (ev_calculator, Pig 8.14179, every card hand, every auction bid, a martingale simulation). Scoring: perfect = 100, wrong tag, rounded EV, full-Kelly half credit, hedge ratio, strict-tag rejection. |
| **EV & order book** | Every puzzle against its known exact answer. HTH waiting time cross-checked by simulation. Microprice, imbalance, walk-the-book, crossed-book detection. |
| **Robustness (fuzz)** | ~2,500 randomised calls across all tools with adversarial values (NaN, ±∞, 1e308, `"1/0"`, `"1e999999"`, booleans, wrong container types, 10³⁰). Every response must be **strict JSON** (no NaN tokens), carry `ok`, and **never** be an `internal error`. Handlers also accept extra kwargs. |

---

## 8. Numerical validation summary

| Quantity | Reference | Plugin | Abs. error |
|---|---|---|---|
| BS ATM call (S=K=100, T=1, r=5%, σ=20%) | 10.450584 | 10.450584 | < 10⁻⁶ |
| CRR American put, 5 steps (Hull) | 4.4885 | 4.488459 | < 5·10⁻⁵ |
| American put, ATM 1y, N = 1000 | ≈ 6.0904 | 6.0896 (6.0916 with CV) | ≈ 10⁻³ |
| Heston (Fang–Oosterlee 2008) | 5.785155450 | 5.785155431 | 2·10⁻⁸ |
| Almgren–Chriss κ (AC 2000 example) | ≈ 0.6 | 0.6063 | (closed form, residual 10⁻¹⁶) |
| LSM put, S=36 K=40, 50 dates (12 × 100k paths) | 4.4780 (Bermudan CRR) | 4.4748 ± 0.0024 | −0.003 (expected low bias) |
| Black–Litterman posterior vs precision form | — | — | 1.4·10⁻¹⁷ |
| Pig d6, hold at 20 | 8.14179 | 492303203/60466176 | exact |
| Arithmetic Asian, 52 fixings | (no closed form) | 5.85514 ± 0.00168 | — |
| DOC barrier H=90, continuous | 8.66547 | 8.6268 ± 0.0400 | z = −0.97 |
| Secretary n = 10 | 3349/8400 | 3349/8400 | exact |
| Penney THH vs HHT | 3/4 | 3/4 | exact |

---

## 9. Limitations & model risk

| Area | Limitation |
|---|---|
| Performance | Pure-Python Monte Carlo (no NumPy, by design: zero dependencies) does roughly 5–6·10⁶ path-steps per second, with a hard cap of 5·10⁶ path-steps per call. Use it for research-grade estimates, not risk runs. |
| Dividends | Continuous yield only; no discrete cash dividends in BS, CRR or MC. |
| Heston | Pricing only, no calibration. Greeks by finite differences. Truncation is flagged rather than corrected for extremely short maturities or tiny variance. |
| Binomial | The boundary resolution is one tree step. Greeks are lattice estimates and inherit CRR's odd/even oscillation. |
| Control variates | $\hat\beta$ is estimated in-sample, which introduces an $O(1/n)$ bias that is negligible at the default path counts. |
| Avellaneda–Stoikov | Assumes arithmetic-BM mid, exponential fill intensity and CARA utility. Model inputs ($k$, $A$, σ) must be estimated from your venue's data. |
| VPIN | Per-trade BVC variant. The 0.2 / 0.4 levels are heuristics; alert on the CDF percentile against your own history. |
| Matrix games | Returns one optimal vertex when the equilibrium is not unique. |
| Almgren–Chriss | Linear impact, arithmetic random walk, static (pre-committed) schedule; no adaptive or intraday-volume-curve execution. Impact parameters must be calibrated to your venue. |
| Longstaff–Schwartz | GBM only. `std_error` is conditional on the fitted policy (regression noise excluded, so compare seeds). The European control variate gives only 1.1–2.2× for American puts. The pure-Python budget is 10⁶ path-dates. |
| Black–Litterman | Normal returns, static single-period mean-variance. The confidence→Ω mapping is the proportional (He–Litterman-scaled) one, not Idzorek's iterative calibration. |
| EV trainer | Puzzle families are fixed templates with randomised parameters. HEDGE puzzles occur at difficulty ≥ 2 (≈5% of conditional-expectation puzzles). |
| Enforcement | The audit is syntactic (format, not truth). |
| Scope | **Not investment advice and not an execution system.** No market data access, order routing or position keeping. |

---

## 10. Repository layout

```
.
├── LICENSE                          MIT
├── README.md
└── waterloo-jane-street-qr/
    ├── plugin.yaml                  manifest v2: tools, hooks, config_schema (enforcement)
    ├── __init__.py                  ~4.1k LOC: validation, kernels, 14 tools, persona, hooks, register()
    ├── .gitignore                   __pycache__/, *.pyc, .pytest_cache/
    └── tests/
        └── test_plugin.py           81 tests (pytest or standalone; NumPy tests skip cleanly)
```

---

### References

- Black, F. & Scholes, M. (1973); Merton, R. (1973). *The pricing of options and corporate liabilities / Theory of rational option pricing.*
- Cox, J., Ross, S. & Rubinstein, M. (1979). *Option pricing: a simplified approach.* JFE.
- Heston, S. (1993). *A closed-form solution for options with stochastic volatility.* RFS.
- Lewis, A. (2001). *A simple option formula for general jump-diffusion and other exponential Lévy processes.*
- Albrecher, H., Mayer, P., Schoutens, W. & Tistaert, J. (2007). *The little Heston trap.* Wilmott.
- Fang, F. & Oosterlee, C. (2008). *A novel pricing method for European options based on Fourier-cosine series expansions.* SISC.
- Broadie, M., Glasserman, P. & Kou, S. (1997). *A continuity correction for discrete barrier options.* Math. Finance.
- Reiner, E. & Rubinstein, M. (1991). *Breaking down the barriers.* Risk.
- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.* Quant. Finance.
- Guéant, O., Lehalle, C.-A. & Fernandez-Tapia, J. (2013). *Dealing with the inventory risk.* Math. Financ. Econ.
- Easley, D., López de Prado, M. & O'Hara, M. (2012). *Flow toxicity and liquidity in a high-frequency world.* RFS.
- Kyle, A. (1985). *Continuous auctions and insider trading.* Econometrica.
- Almgren, R. & Chriss, N. (2000). *Optimal execution of portfolio transactions.* J. Risk.
- Longstaff, F. & Schwartz, E. (2001). *Valuing American options by simulation: a simple least-squares approach.* RFS.
- Black, F. & Litterman, R. (1992). *Global portfolio optimization.* FAJ.
- He, G. & Litterman, R. (1999). *The intuition behind Black-Litterman model portfolios.* Goldman Sachs.
- Beck, A. & Teboulle, M. (2009). *A fast iterative shrinkage-thresholding algorithm.* SIAM J. Imaging Sci.
- Hull, J. *Options, Futures, and Other Derivatives.*

---

© 2026 Zhixuan Xu (George). Released under the MIT License.
