"""
Performance Deviation Detection and Root Cause Attribution Module.
Analyzes residuals (Actual - Predicted AC_POWER) using context-aware, MAD,
and standard error thresholds computed strictly on training/validation partitions.
Distinguishes capacity derating, thermal losses, sensor discrepancies, and cloud transients.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class SolarDeviationDetector:
    """
    Context-aware deviation detector for solar power generation.
    
    Residual definition:
      residual = actual_ac_power - predicted_ac_power
      positive residual: actual generation exceeded model expectation
      negative residual: actual generation was lower than expectation (underperformance)
      
    Severity classifications:
      - Normal: Within standard operational noise band
      - Moderate Deviation: Residual exceeds 1.5x dynamic MAD threshold
      - Significant Deviation: Residual exceeds 2.5x dynamic MAD threshold
      - Extreme Deviation: Residual exceeds 4.0x dynamic MAD threshold or severe daylight loss
    """

    NOMINAL_INVERTER_COUNT = 22

    def __init__(
        self,
        baseline_mad: float = 350.0,
        min_power_threshold: float = 50.0,
    ):
        self.baseline_mad = baseline_mad
        self.min_power_threshold = min_power_threshold
        self.thresholds: Dict[str, float] = {}

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray, irradiation: np.ndarray) -> "SolarDeviationDetector":
        """
        Calculates baseline MAD and residual distribution strictly from training data.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        irradiation = np.asarray(irradiation)

        # Focus MAD calibration strictly on daylight observations (irradiation > 0.05 kW/m²)
        day_mask = irradiation > 0.05
        if day_mask.sum() > 0:
            res = y_true[day_mask] - y_pred[day_mask]
        else:
            res = y_true - y_pred

        median_res = float(np.median(res))
        abs_dev = np.abs(res - median_res)
        mad = float(np.median(abs_dev))
        std_res = float(np.std(res))

        effective_mad = mad if mad > 0 else self.baseline_mad

        self.thresholds = {
            "median_residual": round(median_res, 2),
            "mad": round(effective_mad, 2),
            "std_residual": round(std_res, 2),
            "moderate_threshold": round(1.5 * effective_mad, 2),
            "significant_threshold": round(2.5 * effective_mad, 2),
            "extreme_threshold": round(4.0 * effective_mad, 2),
        }
        return self

    def analyze_single(
        self,
        actual_ac_power: float,
        predicted_ac_power: float,
        irradiation: float,
        ambient_temperature: float,
        module_temperature: float,
        plant_id: int,
        timestamp: Optional[str] = None,
        active_inverters: Optional[int] = None,
    ) -> Dict[str, Union[float, str, List[str], Optional[float]]]:
        """
        Analyzes a single observation for deviation severity, percentage error,
        and physical contributing conditions.
        """
        residual = float(actual_ac_power - predicted_ac_power)
        abs_residual = abs(residual)

        # Dynamic context scaling: irradiance scale factor
        # At high noon (1.0 kW/m²), physical measurement uncertainty is naturally higher
        # than at early sunrise / late dusk (0.1 kW/m²).
        irr_scale = max(0.2, min(1.5, float(irradiation) / 0.5)) if irradiation > 0 else 0.2

        mod_thresh = self.thresholds.get("moderate_threshold", 500.0) * irr_scale
        sig_thresh = self.thresholds.get("significant_threshold", 850.0) * irr_scale
        ext_thresh = self.thresholds.get("extreme_threshold", 1400.0) * irr_scale

        # Severity categorization
        is_severe_deficit = (irradiation > 0.35 and actual_ac_power < 0.15 * predicted_ac_power)
        if abs_residual >= ext_thresh or is_severe_deficit:
            severity = "Extreme Deviation"
        elif abs_residual >= sig_thresh:
            severity = "Significant Deviation"
        elif abs_residual >= mod_thresh:
            severity = "Moderate Deviation"
        else:
            severity = "Normal"

        # Percentage deviation
        if predicted_ac_power >= self.min_power_threshold:
            pct_dev = round((residual / predicted_ac_power) * 100.0, 2)
        else:
            pct_dev = None

        # Physical Root Cause Attribution
        contributing_factors = []
        if severity != "Normal":
            # 1. Capacity Derating vs Hardware Outage
            if active_inverters is not None and active_inverters < self.NOMINAL_INVERTER_COUNT:
                lost_inverters = self.NOMINAL_INVERTER_COUNT - active_inverters
                contributing_factors.append(
                    f"Reduced active inverter count ({active_inverters}/{self.NOMINAL_INVERTER_COUNT}). "
                    f"{lost_inverters} inverter(s) disconnected explains {(lost_inverters/self.NOMINAL_INVERTER_COUNT)*100:.1f}% capacity loss."
                )

            if is_severe_deficit:
                contributing_factors.append(
                    "Severe generation deficit under strong solar irradiance (> 0.35 kW/m²). "
                    "Indicates probable inverter trip, AC breaker trip, or substation curtailment."
                )
            elif residual < 0 and module_temperature > 55.0:
                contributing_factors.append(
                    f"Elevated photovoltaic module temperature ({module_temperature:.1f}°C). "
                    "Photovoltaic semiconductor efficiency derating (-0.4%/°C) is heavily reducing panel voltage."
                )
            elif residual < 0 and (module_temperature - ambient_temperature) < 5.0 and irradiation > 0.25:
                contributing_factors.append(
                    "Low module-to-ambient temperature gradient (< 5°C) despite significant irradiance. "
                    "Possible localized panel shading, sensor calibration discrepancy, or pyranometer misalignment."
                )
            elif residual > 0 and irradiation > 0.8:
                contributing_factors.append(
                    "Actual power generation significantly exceeded model baseline during peak sun. "
                    "Favorable environmental factors: clean panel surfaces, cool wind gusts, or pyranometer calibration offset."
                )
            elif irradiation < 0.05 and actual_ac_power > self.min_power_threshold:
                contributing_factors.append(
                    "Non-zero power generation recorded under near-zero measured irradiance. "
                    "Likely pyranometer shading or sensor soiling."
                )
            else:
                contributing_factors.append(
                    "Operational variance exceeding dynamic statistical tolerance band. "
                    "Consistent with rapid cloud cover transients or intermittent grid curtailment."
                )

        return {
            "timestamp": timestamp,
            "plant_id": plant_id,
            "actual_ac_power": round(float(actual_ac_power), 2),
            "predicted_ac_power": round(float(predicted_ac_power), 2),
            "residual": round(residual, 2),
            "absolute_deviation": round(abs_residual, 2),
            "deviation_percentage": pct_dev,
            "severity": severity,
            "contributing_conditions": contributing_factors,
            "investigation_status": "REQUIRES_INSPECTION" if severity in ["Significant Deviation", "Extreme Deviation"] else "CLEARED",
        }

    def analyze_batch(
        self,
        df: pd.DataFrame,
        actual_col: str = "AC_POWER",
        pred_col: str = "pred_AC_POWER",
    ) -> pd.DataFrame:
        """
        Runs deviation analysis across a batch DataFrame.
        """
        results = []
        for _, row in df.iterrows():
            res = self.analyze_single(
                actual_ac_power=row[actual_col],
                predicted_ac_power=row[pred_col],
                irradiation=row.get("IRRADIATION", 0.0),
                ambient_temperature=row.get("AMBIENT_TEMPERATURE", 25.0),
                module_temperature=row.get("MODULE_TEMPERATURE", 25.0),
                plant_id=int(row.get("PLANT_ID", 4135001)),
                timestamp=str(row.get("DATE_TIME", "")),
                active_inverters=row.get("INVERTER_COUNT", None),
            )
            results.append(res)
        return pd.DataFrame(results)
