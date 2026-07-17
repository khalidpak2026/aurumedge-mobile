from __future__ import annotations

import math
from dataclasses import dataclass

from .models import PositionRiskPlan


@dataclass(slots=True)
class RiskInputs:
    account_balance: float = 10000.0
    risk_percent: float = 1.0
    requested_lot: float = 0.10
    contract_size: float = 100.0
    lot_step: float = 0.01
    min_lot: float = 0.01
    maximum_risk_dollars: float = 0.0
    spread_price: float = 0.50
    slippage_price: float = 0.20
    minimum_stop_atr: float = 0.55
    maximum_stop_atr: float = 1.55


def _floor_step(value: float, step: float) -> float:
    if value <= 0 or step <= 0:
        return 0.0
    precision = max(0, int(round(-math.log10(step)))) if step < 1 else 0
    return round(math.floor(value / step + 1e-12) * step, precision)


def risk_budget(inputs: RiskInputs) -> float:
    percentage_budget = max(0.0, inputs.account_balance) * max(0.0, inputs.risk_percent) / 100.0
    if inputs.maximum_risk_dollars > 0:
        return min(percentage_budget, inputs.maximum_risk_dollars)
    return percentage_budget


def estimate_loss(
    entry: float,
    stop: float,
    lot: float,
    contract_size: float,
    spread_price: float = 0.0,
    slippage_price: float = 0.0,
) -> float:
    distance = abs(float(entry) - float(stop)) + max(0.0, spread_price) + max(0.0, slippage_price)
    return max(0.0, distance * max(0.0, lot) * max(0.0, contract_size))


def build_position_risk_plan(entry: float, stop: float, inputs: RiskInputs) -> PositionRiskPlan:
    budget = risk_budget(inputs)
    stop_distance = abs(float(entry) - float(stop))
    all_in_distance = stop_distance + max(0.0, inputs.spread_price) + max(0.0, inputs.slippage_price)
    denominator = max(1e-12, all_in_distance * max(0.01, inputs.contract_size))
    raw_safe_lot = budget / denominator if budget > 0 else 0.0
    max_safe_lot = _floor_step(raw_safe_lot, inputs.lot_step)
    recommended_lot = max_safe_lot if max_safe_lot >= inputs.min_lot else 0.0
    requested_loss = estimate_loss(
        entry,
        stop,
        inputs.requested_lot,
        inputs.contract_size,
        inputs.spread_price,
        inputs.slippage_price,
    )
    recommended_loss = estimate_loss(
        entry,
        stop,
        recommended_lot,
        inputs.contract_size,
        inputs.spread_price,
        inputs.slippage_price,
    )

    if recommended_lot < inputs.min_lot or budget <= 0:
        status = "NO_TRADE"
        message = (
            "The structural stop cannot be traded within the configured risk budget, even at the minimum lot. "
            "Increase account risk only deliberately, wait for a closer entry, or skip the setup."
        )
    elif inputs.requested_lot <= max_safe_lot + inputs.lot_step / 10:
        status = "OK"
        message = (
            f"Requested lot fits the risk budget. Estimated worst-case loss is about ${requested_loss:,.2f}, "
            "including the configured spread/slippage allowance."
        )
    else:
        status = "REDUCE_LOT"
        message = (
            f"Requested {inputs.requested_lot:.2f} lot is too large for this structural stop. "
            f"Use no more than about {recommended_lot:.2f} lot to keep estimated loss near ${budget:,.2f}."
        )

    return PositionRiskPlan(
        account_balance=round(inputs.account_balance, 2),
        risk_percent=round(inputs.risk_percent, 3),
        risk_budget=round(budget, 2),
        requested_lot=round(inputs.requested_lot, 3),
        recommended_lot=round(recommended_lot, 3),
        maximum_safe_lot=round(max_safe_lot, 3),
        contract_size=round(inputs.contract_size, 3),
        lot_step=round(inputs.lot_step, 4),
        stop_distance=round(stop_distance, 4),
        estimated_loss_requested_lot=round(requested_loss, 2),
        estimated_loss_recommended_lot=round(recommended_loss, 2),
        status=status,  # type: ignore[arg-type]
        message=message,
    )
