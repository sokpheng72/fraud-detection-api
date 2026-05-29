import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
CONFIG_DIR = BASE_DIR / "config"


class FraudPredictor:
    def __init__(self):
        with open(MODEL_DIR / "model.pkl", "rb") as f:
            self.model = pickle.load(f)

        with open(MODEL_DIR / "encoders.pkl", "rb") as f:
            enc = pickle.load(f)
        self.card_brand_encoder = enc["card_brand_encoder"]
        self.merchant_state_encoder = enc["merchant_state_encoder"]
        self.chip_map = enc["chip_map"]

        meta = json.load(open(CONFIG_DIR / "model_metadata.json"))
        self.feature_order = meta["feature_order"]
        self.threshold = meta["deploy_threshold"]
        self.model_name = meta["deploy_model"]

    def _encode_label(self, encoder, value: str) -> int:
        """LabelEncoder transform with fallback to 0 for unseen values."""
        try:
            return int(encoder.transform([value])[0])
        except ValueError:
            return 0

    def build_features(self, req) -> pd.DataFrame:
        dt = datetime.fromisoformat(req.timestamp)
        amount_abs = abs(req.amount)
        tx_year = dt.year

        use_chip_enc = self.chip_map.get(req.use_chip, -1)
        card_brand_enc = self._encode_label(self.card_brand_encoder, req.card_brand)
        merchant_state_enc = self._encode_label(self.merchant_state_encoder, req.merchant_state)

        amount_to_limit_ratio = min(amount_abs / (abs(req.credit_limit) + 1), 10.0)
        amount_to_income_ratio = min(amount_abs / (abs(req.yearly_income) + 1), 5.0)
        amount_log = np.log1p(amount_abs)
        amount_vs_user_mean = float(np.clip(amount_abs - req.user_mean_amt, -5000, 5000))
        user_std_amt = max(req.user_std_amt, 1.0)

        row = {
            "tx_hour": dt.hour,
            "tx_day": dt.day,
            "tx_month": dt.month,
            "tx_dayofweek": dt.weekday(),
            "tx_is_weekend": int(dt.weekday() >= 5),
            "tx_is_night": int(dt.hour >= 22 or dt.hour <= 5),
            "amount_abs": amount_abs,
            "is_negative": int(req.amount < 0),
            "amount_log": amount_log,
            "amount_to_limit_ratio": amount_to_limit_ratio,
            "amount_to_income_ratio": amount_to_income_ratio,
            "age_at_tx": tx_year - req.birth_year,
            "years_to_retirement": req.retirement_age - req.current_age,
            "debt_to_income": req.total_debt / (abs(req.yearly_income) + 1),
            "credit_score": req.credit_score,
            "num_credit_cards": req.num_credit_cards,
            "num_cards_issued": req.num_cards_issued,
            "credit_limit": req.credit_limit,
            "yearly_income": req.yearly_income,
            "total_debt": req.total_debt,
            "per_capita_income": req.per_capita_income,
            "on_dark_web": int(req.card_on_dark_web.lower() == "yes"),
            "has_chip_flag": int(req.has_chip.lower() == "yes"),
            "use_chip_enc": use_chip_enc,
            "gender_enc": int(req.gender.lower() == "male"),
            "card_type_enc": int(req.card_type.lower() == "credit"),
            "card_brand_enc": card_brand_enc,
            "merchant_state_enc": merchant_state_enc,
            "mcc_enc": req.mcc,
            "time_since_last_tx": min(req.time_since_last_tx, 720.0),
            "tx_count_24h": req.tx_count_24h,
            "amount_sum_24h": req.amount_sum_24h,
            "tx_count_7d": req.tx_count_7d,
            "amount_sum_7d": req.amount_sum_7d,
            "user_mean_amt": req.user_mean_amt,
            "user_std_amt": user_std_amt,
            "amount_vs_user_mean": amount_vs_user_mean,
        }

        return pd.DataFrame([row])[self.feature_order]

    def generate_reasons(self, req, row: dict, proba: float) -> list[dict]:
        dt = datetime.fromisoformat(req.timestamp)
        amount_abs = abs(req.amount)
        user_std = max(req.user_std_amt, 1.0)

        risk_flags = []
        reassuring = []

        # ── Collect risk factors ──────────────────────────────────────────────
        if row["tx_is_night"]:
            risk_flags.append(f"Transaction at {dt.strftime('%I:%M %p')} — unusual night-time hours (10 PM–5 AM)")

        if row["amount_vs_user_mean"] > 1.5 * user_std:
            risk_flags.append(
                f"Amount ${amount_abs:,.2f} is well above your usual spending "
                f"(avg ${req.user_mean_amt:,.2f})"
            )

        if req.use_chip == "Online Transaction":
            risk_flags.append("Online transaction — no physical card present (higher risk channel)")

        if req.tx_count_24h >= 5:
            risk_flags.append(f"High transaction frequency: {int(req.tx_count_24h)} transactions in the last 24 hours")

        if req.time_since_last_tx < 1.0:
            mins = int(req.time_since_last_tx * 60)
            risk_flags.append(f"Previous transaction only {mins} minute(s) ago — rapid succession")

        if row["amount_to_limit_ratio"] > 0.5:
            pct = int(row["amount_to_limit_ratio"] * 100)
            risk_flags.append(f"Transaction uses ~{pct}% of the available credit limit")

        if req.amount_sum_24h > 3 * req.user_mean_amt and req.amount_sum_24h > 0:
            risk_flags.append(
                f"Total 24-hour spend (${req.amount_sum_24h:,.2f}) is unusually high for this account"
            )

        if req.credit_score < 580:
            risk_flags.append(f"Below-average credit score ({int(req.credit_score)})")

        if req.has_chip.lower() == "no":
            risk_flags.append("Card has no chip — older, less secure card format")

        # ── Collect reassuring factors ────────────────────────────────────────
        if req.use_chip == "Chip Transaction":
            reassuring.append("Secure chip transaction — physical card verified")
        if not row["tx_is_night"]:
            reassuring.append(f"Transaction at {dt.strftime('%I:%M %p')} — normal business hours")
        if row["amount_vs_user_mean"] <= user_std:
            reassuring.append(f"Amount ${amount_abs:,.2f} is within your normal spending range")
        if req.tx_count_24h < 3:
            reassuring.append("Normal transaction velocity — no unusual activity pattern")
        if req.credit_score >= 700:
            reassuring.append(f"Good credit score ({int(req.credit_score)}) on this account")

        # ── Combine: risk factors always shown in red; reassuring in green ────
        # For low-risk results show mostly reassuring; for high-risk show mostly flags
        reasons = []
        if proba >= 0.3:
            reasons += [{"text": t, "flag": True} for t in risk_flags[:4]]
            if len(reasons) < 3:
                reasons += [{"text": t, "flag": False} for t in reassuring[:2]]
        else:
            reasons += [{"text": t, "flag": False} for t in reassuring[:3]]
            if risk_flags:
                reasons += [{"text": t, "flag": True} for t in risk_flags[:2]]

        return reasons[:5]

    def predict(self, req):
        X = self.build_features(req)
        proba = float(self.model.predict_proba(X)[0, 1])
        is_fraud = proba >= self.threshold

        if proba < 0.3:
            risk = "LOW"
        elif proba < 0.7:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        row = X.iloc[0].to_dict()
        reasons = self.generate_reasons(req, row, proba)

        return {
            "fraud_probability": round(proba, 6),
            "is_fraud": is_fraud,
            "threshold": self.threshold,
            "model": self.model_name,
            "risk_level": risk,
            "reasons": reasons,
        }
