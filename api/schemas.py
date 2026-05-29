from pydantic import BaseModel, Field
from typing import Optional, List


class TransactionRequest(BaseModel):
    # ── Transaction ───────────────────────────────────────────────────────────
    amount: float = Field(..., description="Transaction amount (can be negative)")
    use_chip: str = Field(..., description="'Swipe Transaction' | 'Chip Transaction' | 'Online Transaction'")
    mcc: int = Field(..., description="Merchant Category Code")
    merchant_state: str = Field(..., description="Merchant state abbreviation e.g. 'CA'")
    timestamp: str = Field(..., description="ISO 8601 datetime e.g. '2023-06-15T14:30:00'")

    # ── User ─────────────────────────────────────────────────────────────────
    birth_year: int
    current_age: int
    retirement_age: int
    gender: str = Field(..., description="'Male' | 'Female'")
    yearly_income: float
    total_debt: float
    credit_limit: float
    credit_score: float
    num_credit_cards: int
    per_capita_income: float

    # ── Card ─────────────────────────────────────────────────────────────────
    card_type: str = Field(..., description="'Credit' | 'Debit'")
    card_brand: str = Field(..., description="'Visa' | 'Mastercard' | 'Amex' | 'Discover'")
    has_chip: str = Field(..., description="'YES' | 'NO'")
    card_on_dark_web: str = Field(default="No", description="'Yes' | 'No'")
    num_cards_issued: int = Field(default=1)

    # ── Velocity (provide from transaction history; defaults = new/unknown user) ──
    time_since_last_tx: float = Field(default=720.0, description="Hours since last tx, capped at 720 (30 days)")
    tx_count_24h: float = Field(default=0.0, description="Number of transactions in past 24h")
    amount_sum_24h: float = Field(default=0.0, description="Total amount in past 24h")
    tx_count_7d: float = Field(default=0.0, description="Number of transactions in past 7 days")
    amount_sum_7d: float = Field(default=0.0, description="Total amount in past 7 days")
    user_mean_amt: float = Field(default=67.0, description="User historical mean transaction amount")
    user_std_amt: float = Field(default=120.0, description="User historical std of transaction amount")

    model_config = {
        "json_schema_extra": {
            "example": {
                "amount": 150.75,
                "use_chip": "Online Transaction",
                "mcc": 5411,
                "merchant_state": "CA",
                "timestamp": "2023-06-15T14:30:00",
                "birth_year": 1985,
                "current_age": 38,
                "retirement_age": 65,
                "gender": "Male",
                "yearly_income": 75000.0,
                "total_debt": 12000.0,
                "credit_limit": 10000.0,
                "credit_score": 720.0,
                "num_credit_cards": 2,
                "per_capita_income": 35000.0,
                "card_type": "Credit",
                "card_brand": "Visa",
                "has_chip": "YES",
                "card_on_dark_web": "No",
                "num_cards_issued": 1,
                "time_since_last_tx": 24.0,
                "tx_count_24h": 2.0,
                "amount_sum_24h": 85.0,
                "tx_count_7d": 8.0,
                "amount_sum_7d": 450.0,
                "user_mean_amt": 67.0,
                "user_std_amt": 120.0,
            }
        }
    }


class ReasonItem(BaseModel):
    text: str
    flag: bool  # True = risk factor (red), False = reassuring (green)


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., description="Model confidence score (0–1)")
    is_fraud: bool = Field(..., description="True if probability >= threshold")
    threshold: float = Field(..., description="Decision threshold used")
    model: str = Field(..., description="Model name")
    risk_level: str = Field(..., description="'LOW' | 'MEDIUM' | 'HIGH'")
    reasons: List[ReasonItem] = Field(default_factory=list, description="Risk and reassuring factors")


class HealthResponse(BaseModel):
    status: str
    model: str
    threshold: float
    features: int
