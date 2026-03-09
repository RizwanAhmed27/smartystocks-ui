# smarty_stocks_pro_subsystems.py
# Assignment 2 structure following team layout:
# AI Sub-system 1: Demand Forecasting Sub-system
# AI Sub-system 2: Inventory Decision and Control Sub-system

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score

import skfuzzy as fuzz
import skfuzzy.control as ctrl


# =========================================================
# Helper functions
# =========================================================
def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def validate_dataset(df: pd.DataFrame) -> list:
    required = ["Units Sold", "Inventory Level"]
    missing = [c for c in required if c not in df.columns]
    return missing


def preprocess_features(df_raw: pd.DataFrame):
    df = df_raw.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["Weekday"] = df["Date"].dt.weekday
        df = df.drop(columns=["Date"])

    for c in ["Demand Forecast", "Predicted Demand", "Residual", "Abs Residual"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    y = pd.to_numeric(df["Units Sold"], errors="coerce").fillna(0.0).astype(float)
    X = df.drop(columns=["Units Sold"])

    inventory = None
    if "Inventory Level" in df_raw.columns:
        inventory = pd.to_numeric(
            df_raw["Inventory Level"], errors="coerce"
        ).fillna(0.0).astype(float).values

    X = pd.get_dummies(X, drop_first=True)

    return X, y, inventory


# =========================================================
# AI Sub-system 1: Demand Forecasting Sub-system
# =========================================================
class DemandForecastingSubsystem:
    """
    AI Sub-system 1
    Random Forest forecasting model
    Predicts future product demand using historical retail data
    """

    def __init__(self, n_estimators: int = 140, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Forecasting model has not been trained yet.")
        return self.model.predict(X_test)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)
        }


# =========================================================
# AI Sub-system 2A: Fuzzy Logic Decision Engine
# =========================================================
class FuzzyDecisionEngine:
    """
    Supporting component inside AI Sub-system 2
    Determines inventory action and priority level
    """

    def __init__(self, demand_max: float = 500, inventory_max: float = 500):
        self.demand_max = float(demand_max)
        self.inventory_max = float(inventory_max)
        self.system = self._build_system()

    def _build_system(self):
        demand = ctrl.Antecedent(np.arange(0, int(self.demand_max) + 1, 1), "demand")
        inventory = ctrl.Antecedent(np.arange(0, int(self.inventory_max) + 1, 1), "inventory")

        action = ctrl.Consequent(np.arange(0, 101, 1), "action")
        priority = ctrl.Consequent(np.arange(0, 101, 1), "priority")

        demand["low"] = fuzz.trimf(demand.universe, [0, 0, 0.4 * self.demand_max])
        demand["medium"] = fuzz.trimf(demand.universe, [0.25 * self.demand_max, 0.55 * self.demand_max, 0.85 * self.demand_max])
        demand["high"] = fuzz.trimf(demand.universe, [0.60 * self.demand_max, self.demand_max, self.demand_max])

        inventory["low"] = fuzz.trimf(inventory.universe, [0, 0, 0.4 * self.inventory_max])
        inventory["medium"] = fuzz.trimf(inventory.universe, [0.25 * self.inventory_max, 0.55 * self.inventory_max, 0.85 * self.inventory_max])
        inventory["high"] = fuzz.trimf(inventory.universe, [0.60 * self.inventory_max, self.inventory_max, self.inventory_max])

        action["reduce"] = fuzz.trimf(action.universe, [0, 0, 40])
        action["maintain"] = fuzz.trimf(action.universe, [30, 50, 70])
        action["reorder"] = fuzz.trimf(action.universe, [60, 100, 100])

        priority["low"] = fuzz.trimf(priority.universe, [0, 0, 40])
        priority["medium"] = fuzz.trimf(priority.universe, [30, 50, 70])
        priority["high"] = fuzz.trimf(priority.universe, [60, 100, 100])

        rules = [
            ctrl.Rule(demand["high"] & inventory["low"], (action["reorder"], priority["high"])),
            ctrl.Rule(demand["high"] & inventory["medium"], (action["reorder"], priority["high"])),
            ctrl.Rule(demand["high"] & inventory["high"], (action["maintain"], priority["medium"])),

            ctrl.Rule(demand["medium"] & inventory["low"], (action["reorder"], priority["medium"])),
            ctrl.Rule(demand["medium"] & inventory["medium"], (action["maintain"], priority["medium"])),
            ctrl.Rule(demand["medium"] & inventory["high"], (action["reduce"], priority["low"])),

            ctrl.Rule(demand["low"] & inventory["low"], (action["maintain"], priority["low"])),
            ctrl.Rule(demand["low"] & inventory["medium"], (action["reduce"], priority["low"])),
            ctrl.Rule(demand["low"] & inventory["high"], (action["reduce"], priority["low"])),
        ]

        return ctrl.ControlSystem(rules)

    def infer(self, predicted_demand: float, inventory_level: float) -> dict:
        sim = ctrl.ControlSystemSimulation(self.system)
        sim.input["demand"] = float(np.clip(predicted_demand, 0, self.demand_max))
        sim.input["inventory"] = float(np.clip(inventory_level, 0, self.inventory_max))
        sim.compute()

        action_score = float(sim.output.get("action", 50))
        priority_score = float(sim.output.get("priority", 50))

        if action_score < 33:
            action_label = "Reduce excess stock"
        elif action_score < 66:
            action_label = "Maintain current level"
        else:
            action_label = "Restock inventory"

        if priority_score < 33:
            priority_label = "Low"
        elif priority_score < 66:
            priority_label = "Medium"
        else:
            priority_label = "High"

        return {
            "action_score": round(action_score, 2),
            "priority_score": round(priority_score, 2),
            "recommended_action": action_label,
            "priority_level": priority_label,
        }


# =========================================================
# AI Sub-system 2B: Anomaly Detection Supporting Component
# =========================================================
class AnomalyDetectionModule:
    """
    Supporting component inside AI Sub-system 2
    Monitors abnormal sales / forecast behaviour
    """

    def __init__(self, contamination: float = 0.02, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state
        )

    def detect(self, actual: np.ndarray, predicted: np.ndarray, z_thresh: float = 3.0) -> pd.DataFrame:
        residual = actual - predicted
        abs_residual = np.abs(residual)

        feat = pd.DataFrame({
            "actual": actual,
            "predicted": predicted,
            "residual": residual,
            "abs_residual": abs_residual
        })

        self.model.fit(feat)
        iso_pred = self.model.predict(feat)
        iso_score = self.model.decision_function(feat)

        mu = float(np.mean(abs_residual))
        sd = float(np.std(abs_residual)) if float(np.std(abs_residual)) > 1e-9 else 1.0
        z_scores = (abs_residual - mu) / sd

        anomaly_flag = (iso_pred == -1) | (z_scores >= z_thresh)

        return pd.DataFrame({
            "Actual Units Sold": actual,
            "Predicted Demand": predicted,
            "Residual": residual,
            "Abs Residual": abs_residual,
            "Residual Z": np.round(z_scores, 2),
            "IsolationForest Score": np.round(iso_score, 4),
            "Anomaly": anomaly_flag
        })


# =========================================================
# AI Sub-system 2: Inventory Decision and Control Sub-system
# =========================================================
class InventoryDecisionControlSubsystem:
    """
    AI Sub-system 2
    Contains:
    - Fuzzy Logic Decision Engine
    - Anomaly Detection module
    """

    def __init__(self, demand_max: float = 500, inventory_max: float = 500, contamination: float = 0.02):
        self.fuzzy_engine = FuzzyDecisionEngine(demand_max=demand_max, inventory_max=inventory_max)
        self.anomaly_detector = AnomalyDetectionModule(contamination=contamination)

    def process(self, predicted: np.ndarray, actual: np.ndarray, inventory: np.ndarray, z_thresh: float = 3.0) -> pd.DataFrame:
        anomaly_df = self.anomaly_detector.detect(actual=actual, predicted=predicted, z_thresh=z_thresh)

        decisions = []
        for pred_val, inv_val in zip(predicted, inventory):
            decision = self.fuzzy_engine.infer(predicted_demand=float(pred_val), inventory_level=float(inv_val))
            decisions.append(decision)

        decision_df = pd.DataFrame(decisions)
        decision_df["Inventory Level"] = np.round(inventory, 2)

        return pd.concat([anomaly_df.reset_index(drop=True), decision_df.reset_index(drop=True)], axis=1)


# =========================================================
# Full system runner
# =========================================================
class SmartyStockProSystem:
    """
    Full integrated AI system:
    1. Demand Forecasting Sub-system
    2. Inventory Decision and Control Sub-system
    """

    def __init__(self, n_estimators: int = 140, contamination: float = 0.02):
        self.forecasting_subsystem = DemandForecastingSubsystem(n_estimators=n_estimators)
        self.control_subsystem = InventoryDecisionControlSubsystem(contamination=contamination)

    def run(self, df_raw: pd.DataFrame, split_ratio: float = 0.8, z_thresh: float = 3.0):
        missing = validate_dataset(df_raw)
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        df_raw = ensure_datetime(df_raw)
        X, y, inventory = preprocess_features(df_raw)

        split_idx = int(len(X) * split_ratio)

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:].values
        inv_test = inventory[split_idx:]

        # Sub-system 1
        self.forecasting_subsystem.train(X_train, y_train)
        y_pred = self.forecasting_subsystem.predict(X_test)
        metrics = self.forecasting_subsystem.evaluate(y_test, y_pred)

        # Sub-system 2
        results_df = self.control_subsystem.process(
            predicted=y_pred,
            actual=y_test,
            inventory=inv_test,
            z_thresh=z_thresh
        )

        return {
            "metrics": metrics,
            "forecast_actual": y_test,
            "forecast_pred": y_pred,
            "results_table": results_df
        }


# =========================================================
# Example usage
# =========================================================
if __name__ == "__main__":
    file_path = "retail_store_inventory.csv"
    df = pd.read_csv(file_path)

    system = SmartyStockProSystem(n_estimators=140, contamination=0.02)
    output = system.run(df, split_ratio=0.8, z_thresh=3.0)

    print("Model Evaluation:")
    print(output["metrics"])
    print("\nSample Results:")
    print(output["results_table"].head(10))
