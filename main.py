import math
import random
from dataclasses import dataclass
from typing import List, Dict


def norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def bs_d2(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * math.sqrt(T)


def bs_call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    d1 = bs_d1(S, K, r, sigma, T)
    d2 = bs_d2(d1, sigma, T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_put_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return max(K - S, 0.0)
    d1 = bs_d1(S, K, r, sigma, T)
    d2 = bs_d2(d1, sigma, T)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_delta_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = bs_d1(S, K, r, sigma, T)
    return norm_cdf(d1)


def bs_delta_put(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return -1.0 if S < K else 0.0
    d1 = bs_d1(S, K, r, sigma, T)
    return norm_cdf(d1) - 1.0


def simulate_gbm_path(S0: float, mu: float, sigma_real: float, T: float, steps: int, seed: int = None) -> List[float]:
    if seed is not None:
        random.seed(seed)
    dt = T / steps
    S = S0
    path = [S0]
    for _ in range(steps):
        z = random.gauss(0.0, 1.0)
        S = S * math.exp((mu - 0.5 * sigma_real * sigma_real) * dt + sigma_real * math.sqrt(dt) * z)
        path.append(S)
    return path


@dataclass
class BacktestParams:
    S0: float = 100.0
    K: float = 100.0
    r: float = 0.03
    T: float = 30 / 252
    steps: int = 30
    iv: float = 0.25
    sigma_real: float = 0.25
    mu: float = 0.0
    contracts: int = 1
    contract_multiplier: int = 1
    include_financing: bool = True


def backtest_delta_hedge_straddle(params: BacktestParams, seed: int = None) -> Dict[str, List[float]]:
    # Long 1x ATM call + 1x ATM put (straddle). Delta-hedge with underlying.
    S_path = simulate_gbm_path(params.S0, params.mu, params.sigma_real, params.T, params.steps, seed=seed)
    dt = params.T / params.steps

    T_rem = params.T
    call0 = bs_call_price(S_path[0], params.K, params.r, params.iv, T_rem)
    put0 = bs_put_price(S_path[0], params.K, params.r, params.iv, T_rem)

    opt_value0 = (call0 + put0) * params.contracts * params.contract_multiplier

    delta0 = bs_delta_call(S_path[0], params.K, params.r, params.iv, T_rem) + bs_delta_put(S_path[0], params.K,
                                                                                           params.r, params.iv, T_rem)
    delta0 *= params.contracts * params.contract_multiplier

    hedge_shares = -delta0
    cash = -opt_value0 - hedge_shares * S_path[0]

    times = [0.0]
    option_values = [opt_value0]
    hedge_values = [hedge_shares * S_path[0]]
    total_values = [opt_value0 + hedge_shares * S_path[0] + cash]
    deltas = [delta0]
    cash_series = [cash]


    for i in range(1, len(S_path)):
        if params.include_financing:
            cash *= math.exp(params.r * dt)

        T_rem = max(params.T - i * dt, 0.0)
        S = S_path[i]

        call = bs_call_price(S, params.K, params.r, params.iv, T_rem)
        put = bs_put_price(S, params.K, params.r, params.iv, T_rem)
        opt_value = (call + put) * params.contracts * params.contract_multiplier

        delta = bs_delta_call(S, params.K, params.r, params.iv, T_rem) + bs_delta_put(S, params.K, params.r, params.iv,
                                                                                      T_rem)
        delta *= params.contracts * params.contract_multiplier

        target_hedge = -delta
        dH = target_hedge - hedge_shares
        cash -= dH * S
        hedge_shares = target_hedge

        times.append(i * dt)
        option_values.append(opt_value)
        hedge_values.append(hedge_shares * S)
        total_values.append(opt_value + hedge_shares * S + cash)
        deltas.append(delta)
        cash_series.append(cash)


    return {
        "S": S_path,
        "t": times,
        "option_value": option_values,
        "hedge_value": hedge_values,
        "cash": cash_series,
        "total": total_values,
        "delta": deltas,
        "pnl": [v - total_values[0] for v in total_values],
    }

def run_many(params: BacktestParams, n_sims: int = 200):
    finals = []

    for j in range(n_sims):
        res = backtest_delta_hedge_straddle(params,)
        finals.append(res["pnl"][-1])

    mean = sum(finals) / len(finals)
    minimum = min(final)
    maximum = max(final)

    variance = sum((x - mean) ** 2 for x in finals) / len(finals)
    std_dev = math.sqrt(variance)

    profitable_runs = sum(1 for x in finals if x > 0)
    profit_probability = profitable_runs / len(finals)


    return {
        "mean": mean
        "max": maximum
        "risk - volatility": std_dev
        "profit probability": profit_probability
        "min": minimum
    }

if __name__ == "__main__":
    params = BacktestParams(
        S0=100,
        K=100,
        r=0.03,
        T=30/252,
        steps=30,
        iv=0.25,
        sigma_real=0.5,
        mu=0.0,
        contracts=100,
        contract_multiplier=1,
        include_financing=True,
    )

stats = run_many(params, 500)
call0 = bs_call_price(params.S0, params.K, params.r, params.iv, params.T)
put0  = bs_put_price(params.S0, params.K, params.r, params.iv, params.T)

result = backtest_delta_hedge_straddle(params, seed=42)
print("Final PnL:", result["pnl"][-1])
print("Final total value:", result["total"][-1])
print("Final delta:", result["delta"][-1])
print("Initial options value:", result["option_value"][0])
print("Initial call value:", call0)
print("Initial put value:", put0)
print("Mean final PnL:", stats)
print("Min final PnL:", stats["min"])
print("Max final PnL:", stats["max"])
print("Std dev:", stats["std_dev"])
print("Profit probability:", stats["profit_probability"])
