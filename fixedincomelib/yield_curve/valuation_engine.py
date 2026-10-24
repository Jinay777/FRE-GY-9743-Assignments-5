"""Analytic valuation of a signed fixed-accrual cashflow on an IFR model."""

import numpy as np

from fixedincomelib.model import ModelType
from fixedincomelib.product import ProductFixedAccruedCashflow
from fixedincomelib.valuation import (
    AnalyticValParam, FundingIndexParameter, ValuationEngineProduct,
    ValuationEngineProductRegistry,
)


class ValuationEngineProductFixedAccrued(ValuationEngineProduct):
    def __init__(self, model, valuation_parameters_collection, product, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        # I pick the curve to discount on from the funding parameter.
        funding_parameter = valuation_parameters_collection.get_vp_from_build_method_collection(
            FundingIndexParameter._vp_type
        )
        self.funding_index_ = funding_parameter.get_funding_index(product.currency)
        self.payment_date_ = product.payment_date
        # Notional is signed and accrued is a year fraction, so C = N * alpha carries the direction.
        self.cash_amount_ = product.notional * product.accrued

    def calculate_value(self) -> None:
        if self.payment_date_ > self.value_date:
            # Future payment: discount it on the funding curve; nothing is paid today.
            discount_factor = self.model.discount_factor(self.funding_index_, self.payment_date_)
            self.value_ = self.cash_amount_ * discount_factor
            self.cash_ = 0.0
        elif self.payment_date_ == self.value_date:
            # Paid today: worth exactly its cash amount, and that cash is today's cash.
            self.value_ = self.cash_amount_
            self.cash_ = self.cash_amount_
        else:
            # Already paid: nothing left to value.
            self.value_ = 0.0
            self.cash_ = 0.0

    def calculate_risk(
        self, gradient: list[np.ndarray], scaler: float = 1.0, accumulate: bool = False,
    ) -> None:
        if self.payment_date_ > self.value_date:
            # PV = C * D(T), so the risk is C times the DF gradient; I fold C into the scaler.
            self.model.discount_factor_gradient_wrt_state(
                self.funding_index_, self.payment_date_, gradient,
                scaler=scaler * self.cash_amount_, accumulate=accumulate,
            )
        else:
            # Paid today or in the past: no curve exposure. Overwriting still has to clear
            # the output, while accumulating adds zero and leaves it as it was.
            self.model.resize_gradient(gradient)
            if not accumulate:
                for block in gradient:
                    block[:] = 0.0


ValuationEngineProductRegistry().register(
    (ModelType.YIELD_CURVE.to_string(), ProductFixedAccruedCashflow._product_type, AnalyticValParam._vp_type),
    ValuationEngineProductFixedAccrued,
)