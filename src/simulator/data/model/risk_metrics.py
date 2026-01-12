from pydantic import BaseModel
from pydantic import Field, field_validator, model_validator


class RiskMetrics(BaseModel):
    expected_return: float
    median_return: float
    var_95: float 
    cvar_95: float
    max_drawdown: float = Field(..., le=0) # Must be <= 0
    volatility: float = Field(..., gt=0)   # Must be > 0

    @model_validator(mode='after')
    def validate_risk_logic(self) -> 'RiskMetrics':
        # 1. Logic Check: CVaR must be more extreme (lower) than VaR
        # Since these are returns (e.g., -0.15), CVaR should be <= VaR
        if self.cvar_95 > self.var_95:
            raise ValueError(
                f"Mathematical Inconsistency: CVaR ({self.cvar_95}) "
                f"cannot be better than VaR ({self.var_95})"
            )
        
        # 2. Logic Check: Max Drawdown is usually worse than VaR
        if self.max_drawdown > self.var_95:
            # This isn't strictly impossible, but highly unlikely in a 1-year sim.
            # You could trigger a warning or a strict validation here.
            pass
            
        return self

    @field_validator("expected_return", "var_95", "max_drawdown")
    @classmethod
    def check_realistic_bounds(cls, v: float) -> float:
        # Prevent "Exploding Gradients" or simulation errors 
        # (e.g., a return of 1,000,000% is usually a code bug)
        if abs(v) > 100.0: 
            raise ValueError(f"Value {v} is outside realistic financial bounds (>10,000%)")
        return v