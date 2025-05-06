"""Functions for estimating Average Treatment Effects (ATE) using various methods."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, Optional, List, Tuple
import traceback

# Optional dependency handling
try:
    from dowhy import CausalModel
except ImportError:
    CausalModel = None
    print("Optional dependency DoWhy not found. DoWhy-based estimators will not be available.")

try:
    from econml.dml import CausalForestDML
except ImportError:
    CausalForestDML = None
    print("Optional dependency EconML not found. CausalForestDML estimator will not be available.")

try:
    from doubleml.data import DoubleMLData
    from doubleml.double_ml import DoubleMLIRM
except ImportError:
    DoubleMLData = None
    DoubleMLIRM = None
    print("Optional dependency DoubleML not found. DoubleMLIRM estimator will not be available.")

# --- Helper Function for Data Validation ---

def _validate_input_data(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    common_causes: List[str],
    required_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Validates presence of necessary columns and handles potential NaNs."""
    if required_cols is None:
        required_cols = [outcome, treatment] + common_causes

    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Input data is missing required columns: {missing_cols}")

    # Basic NaN check and handling (consider more sophisticated imputation if needed)
    data_subset = data[required_cols].copy()
    initial_rows = data_subset.shape[0]
    data_subset.dropna(inplace=True)
    dropped_rows = initial_rows - data_subset.shape[0]
    if dropped_rows > 0:
        print(f"Warning: Dropped {dropped_rows} rows with NaNs in required columns ({required_cols}).")

    return data_subset

# --- Naive Estimation ---

def estimate_naive_ate(data: pd.DataFrame, treatment: str, outcome: str) -> Dict[str, float]:
    """
    Calculates the naive difference in means between treated and control groups.

    Args:
        data: DataFrame containing treatment and outcome variables.
        treatment: Name of the binary treatment column (0 or 1).
        outcome: Name of the outcome column.

    Returns:
        Dictionary containing means for treated, control, and the naive ATE.
    """
    validated_data = _validate_input_data(data, treatment, outcome, common_causes=[], required_cols=[treatment, outcome])

    treated_mean = validated_data.loc[validated_data[treatment] == 1, outcome].mean()
    control_mean = validated_data.loc[validated_data[treatment] == 0, outcome].mean()
    naive_ate = treated_mean - control_mean

    print("\nNaive Analysis:")
    print(f"  Mean outcome for treated ({treatment}=1): {treated_mean:.4f}")
    print(f"  Mean outcome for control ({treatment}=0): {control_mean:.4f}")
    print(f"  Naive Difference (ATE estimate): {naive_ate:.4f}")

    return {
        "treated_mean": treated_mean,
        "control_mean": control_mean,
        "naive_ate": naive_ate,
    }

# --- Regression Adjustment ---

def estimate_regression_ate(
    data: pd.DataFrame, treatment: str, outcome: str, common_causes: List[str]
) -> Dict[str, Any]:
    """
    Estimates ATE using linear regression adjustment.

    Args:
        data: DataFrame with outcome, treatment, and common cause variables.
        treatment: Name of the treatment column.
        outcome: Name of the outcome column.
        common_causes: List of common cause variable names.

    Returns:
        Dictionary containing the regression results summary and the ATE estimate.
    """
    validated_data = _validate_input_data(data, treatment, outcome, common_causes)

    # Prepare data for statsmodels
    Y = validated_data[outcome]
    X = validated_data[[treatment] + common_causes]
    X = sm.add_constant(X) # Add intercept

    # Fit OLS model
    try:
        model = sm.OLS(Y, X)
        results = model.fit()

        ate_estimate = results.params[treatment]
        ate_pvalue = results.pvalues[treatment]
        conf_int = results.conf_int(alpha=0.05).loc[treatment].tolist()

        print("\nRegression Adjustment Analysis:")
        print(results.summary())
        print(f"\nEstimated ATE (coefficient for {treatment}): {ate_estimate:.4f}")
        print(f"P-value: {ate_pvalue:.4f}")
        print(f"95% Confidence Interval: [{conf_int[0]:.4f}, {conf_int[1]:.4f}]")

        return {
            "summary": results.summary().as_text(), # Or return the results object itself
            "ate": ate_estimate,
            "p_value": ate_pvalue,
            "conf_int": conf_int,
            "model_results": results # Optional: return full results
        }
    except Exception as e:
        print(f"Error during Regression Adjustment: {e}")
        traceback.print_exc()
        return {"error": str(e)}

# --- Propensity Score Matching ---

def estimate_matching_ate(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    common_causes: List[str],
    caliper: float = 0.05,
    ratio: int = 1,
) -> Dict[str, Any]:
    """
    Estimates ATE using Propensity Score Matching (Nearest Neighbors).

    Args:
        data: DataFrame with outcome, treatment, and common cause variables.
        treatment: Name of the treatment column.
        outcome: Name of the outcome column.
        common_causes: List of common cause variable names.
        caliper: Maximum distance for matching (std deviations of propensity score logit).
        ratio: Number of control units to match to each treated unit.

    Returns:
        Dictionary containing matching statistics and the ATE estimate.
    """
    validated_data = _validate_input_data(data, treatment, outcome, common_causes)

    try:
        # 1. Estimate Propensity Scores
        X = validated_data[common_causes]
        T = validated_data[treatment]
        # Scale covariates for logistic regression stability
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        propensity_model = LogisticRegression(solver="liblinear", random_state=123)
        propensity_model.fit(X_scaled, T)
        propensity_scores = propensity_model.predict_proba(X_scaled)[:, 1]
        validated_data["propensity_score"] = propensity_scores
        # Use logit for matching as it often has better distribution properties
        logit_propensity = np.log(propensity_scores / (1 - propensity_scores))
        validated_data["logit_propensity"] = logit_propensity

        # 2. Perform Matching
        treated_units = validated_data[validated_data[treatment] == 1]
        control_units = validated_data[validated_data[treatment] == 0]

        if treated_units.empty or control_units.empty:
            print("Error: No units in treated or control group after validation.")
            return {"error": "No units in treated or control group."}

        # Use NearestNeighbors on the logit propensity score
        nn = NearestNeighbors(
            n_neighbors=ratio, radius=caliper * np.std(logit_propensity), metric="minkowski"
        )
        nn.fit(control_units[["logit_propensity"]])
        distances, indices = nn.kneighbors(treated_units[["logit_propensity"]])

        # 3. Calculate ATE for Matched Sample
        matched_control_outcomes = []
        matched_treated_outcomes = []
        unmatched_treated_count = 0

        for i in range(len(treated_units)):
            treated_outcome = treated_units.iloc[i][outcome]
            # Check if any matches were found within the caliper
            valid_matches = indices[i][distances[i] <= caliper * np.std(logit_propensity)]
            if len(valid_matches) > 0:
                control_match_outcomes = control_units.iloc[valid_matches][outcome].values
                # Store the treated outcome and the *average* outcome of its matches
                matched_treated_outcomes.append(treated_outcome)
                matched_control_outcomes.append(np.mean(control_match_outcomes))
            else:
                unmatched_treated_count += 1

        if not matched_treated_outcomes:
            print("Error: No matches found within the specified caliper.")
            return {"error": "No matches found within caliper."}

        matched_ate = np.mean(np.array(matched_treated_outcomes) - np.array(matched_control_outcomes))

        print("\nPropensity Score Matching Analysis:")
        print(f"  Number of treated units: {len(treated_units)}")
        print(f"  Number of control units: {len(control_units)}")
        print(f"  Number of treated units matched: {len(matched_treated_outcomes)}")
        print(f"  Number of treated units unmatched (caliper): {unmatched_treated_count}")
        print(f"  Estimated ATE (Matched Sample): {matched_ate:.4f}")

        # Optional: Add statistical significance test (e.g., paired t-test on differences)
        from scipy import stats
        if len(matched_treated_outcomes) > 1:
            diff = np.array(matched_treated_outcomes) - np.array(matched_control_outcomes)
            t_stat, p_val = stats.ttest_1samp(diff, 0)
            print(f"  Paired t-test on matched differences: t={t_stat:.4f}, p-value={p_val:.4f}")
        else:
            t_stat, p_val = None, None

        return {
            "ate": matched_ate,
            "num_treated": len(treated_units),
            "num_control": len(control_units),
            "num_matched_treated": len(matched_treated_outcomes),
            "num_unmatched_treated": unmatched_treated_count,
            "t_stat": t_stat,
            "p_value": p_val
        }

    except Exception as e:
        print(f"Error during Matching Analysis: {e}")
        traceback.print_exc()
        return {"error": str(e)}

# --- DoWhy Integration ---

def estimate_dowhy_ate(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    common_causes: List[str],
    graph_str: str,
    method_name: str = "backdoor.linear_regression",
) -> Optional[float]:
    """
    Estimates the Average Treatment Effect (ATE) using DoWhy library.

    Args:
        data: DataFrame with analysis data.
        treatment: Name of the treatment variable.
        outcome: Name of the outcome variable.
        common_causes: List of common cause variable names.
        graph_str: Causal graph in DOT format string.
        method_name: The estimation method to use within DoWhy
                      (e.g., "backdoor.linear_regression",
                       "backdoor.propensity_score_matching").

    Returns:
        The estimated ATE as a float, or None if estimation fails or DoWhy is not installed.
    """
    if CausalModel is None:
        print("DoWhy library not found. Skipping DoWhy estimation.")
        return None

    try:
        # Validate and prepare data (DoWhy handles some internal checks too)
        validated_data = _validate_input_data(data, treatment, outcome, common_causes)
        # Ensure treatment is integer type for some methods
        if validated_data[treatment].dtype == "bool":
            validated_data[treatment] = validated_data[treatment].astype(int)

        # Ensure common causes are numeric for model compatibility
        numeric_common_causes = []
        for cause in common_causes:
            if pd.api.types.is_numeric_dtype(validated_data[cause]):
                numeric_common_causes.append(cause)
            else:
                print(f"Warning: Non-numeric common cause '{cause}' excluded from DoWhy model.")

        if not numeric_common_causes:
            print("Warning: No numeric common causes found for DoWhy model. Results may be biased.")
            # Proceeding without numeric common causes might be problematic

        model = CausalModel(
            data=validated_data,
            treatment=treatment,
            outcome=outcome,
            graph=graph_str,
            common_causes=numeric_common_causes, # Use only valid numeric ones
        )

        identified_estimand = model.identify_effect(proceed_when_unidentified=True)
        estimate = model.estimate_effect(
            identified_estimand, method_name=method_name, test_significance=True
        )

        ate = estimate.value
        print(f"\nDoWhy Estimation ({method_name}):")
        print(estimate)
        print(f"Estimated ATE: {ate:.4f}")
        return float(ate) if ate is not None else None

    except Exception as e:
        print(f"Error during DoWhy estimation ({method_name}): {e}")
        traceback.print_exc()
        return None

# --- EconML Causal Forest ---

def estimate_causalforest_ate(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    common_causes: List[str],
    n_estimators: int = 100,
    min_samples_leaf: int = 10,
) -> Optional[float]:
    """
    Estimates ATE using Causal Forest DML from EconML.

    Args:
        data: DataFrame with analysis data.
        treatment: Name of the treatment variable.
        outcome: Name of the outcome variable.
        common_causes: List of common cause variable names (confounders/controls).
        n_estimators: Number of trees in the forest.
        min_samples_leaf: Minimum number of samples required at a leaf node.

    Returns:
        The estimated ATE as a float, or None if estimation fails or EconML is not installed.
    """
    if CausalForestDML is None:
        print("EconML library not found. Skipping CausalForestDML estimation.")
        return None

    try:
        validated_data = _validate_input_data(data, treatment, outcome, common_causes)

        Y = validated_data[outcome]
        T = validated_data[treatment]
        X = validated_data[common_causes]

        # Ensure X contains only numeric data and handle missing values
        X = X.select_dtypes(include=np.number)
        if X.isnull().any().any():
            print("Warning: Missing values found in features (X) for CausalForest. Filling with median.")
            X = X.fillna(X.median())

        if X.empty or X.shape[1] == 0:
             print("Error: No valid numeric common causes (features X) found for CausalForest.")
             return None

        # Define outcome and treatment models (nuisance functions)
        model_y = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=123)
        model_t = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=123)

        # Initialize and fit CausalForestDML
        est = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=True, # Crucial for binary/categorical treatment
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=123,
        )
        est.fit(Y, T, X=X)

        # Calculate Average Treatment Effect
        ate = est.ate(X=X)
        print("\nCausalForestDML Estimation:")
        print(f"Estimated ATE: {ate:.4f}")

        # Optional: Confidence intervals
        try:
            ate_interval = est.ate_interval(X=X, alpha=0.05) # 95% CI
            print(f"95% CI for ATE: [{ate_interval[0]:.4f}, {ate_interval[1]:.4f}]")
        except Exception as ci_err:
            print(f"Could not compute confidence intervals: {ci_err}")

        return float(ate) if ate is not None else None

    except Exception as e:
        print(f"Error during CausalForestDML estimation: {e}")
        traceback.print_exc()
        return None

# --- DoubleML IRM ---

def estimate_doubleml_irm_ate(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    common_causes: List[str],
    ml_learner_name: str = "RandomForest",
) -> Optional[float]:
    """
    Estimates ATE using DoubleML's Interactive Regression Model (IRM).

    Args:
        data: DataFrame with analysis data.
        treatment: Name of the treatment variable.
        outcome: Name of the outcome variable.
        common_causes: List of common cause variable names (confounders).
        ml_learner_name: Name of the ML learner ("RandomForest" or "Lasso").

    Returns:
        The estimated ATE as a float, or None if estimation fails or DoubleML is not installed.
    """
    if DoubleMLData is None or DoubleMLIRM is None:
        print("DoubleML library not found. Skipping DoubleML IRM estimation.")
        return None

    try:
        # Prepare data for DoubleMLData object
        y_col = outcome
        d_cols = treatment # Treatment variable
        x_cols = common_causes # Confounders

        # Ensure confounders are numeric
        numeric_confounders = data[x_cols].select_dtypes(include=np.number).columns.tolist()
        if len(numeric_confounders) < len(x_cols):
            excluded = set(x_cols) - set(numeric_confounders)
            print(f"Warning: Non-numeric confounders excluded from DoubleML: {excluded}")
        if not numeric_confounders:
            print("Error: No numeric confounders available for DoubleML.")
            return None
        x_cols = numeric_confounders

        # Validate and handle NaNs using the helper
        validated_data = _validate_input_data(data, treatment, outcome, x_cols)

        # Create DoubleMLData object
        dml_data = DoubleMLData(validated_data, y_col=y_col, d_cols=d_cols, x_cols=x_cols)

        # Choose ML learners
        if ml_learner_name.lower() == "randomforest":
            ml_g = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=123)
            ml_m = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=123)
        elif ml_learner_name.lower() == "lasso":
            ml_g = LassoCV(cv=5, random_state=123)
            ml_m = LogisticRegressionCV(cv=5, solver="liblinear", random_state=123)
        else:
            print(f"Unsupported ml_learner_name: {ml_learner_name}. Use 'RandomForest' or 'Lasso'.")
            return None

        # Initialize DoubleMLIRM model
        dml_irm_model = DoubleMLIRM(dml_data, ml_g=ml_g, ml_m=ml_m)

        # Fit and estimate ATE
        dml_irm_model.fit()
        # Explicitly get the value and cast, maybe helps with mock interaction?
        ate_value = dml_irm_model.coef_[0]
        ate = float(ate_value) if ate_value is not None else None

        print(f"\nDoubleML IRM Estimation ({ml_learner_name}):")
        print(dml_irm_model.summary)
        # Use the casted float variable in the print statement
        print(f"Estimated ATE: {ate:.4f}")

        # Return the casted float
        return ate

    except Exception as e:
        print(f"Error during DoubleML IRM estimation: {e}")
        traceback.print_exc()
        return None


# --- Wrapper/Simplified Call Functions (Optional - can be handled by orchestrator) ---
# These functions were present in the original class but largely wrapped the core estimators.
# They can be simplified or removed if the main orchestrator class handles calling the core estimators directly.

def run_dowhy_analysis(data: pd.DataFrame, treatment: str, outcome: str, common_causes: List[str], graph_str: str, method_name: str = "backdoor.linear_regression") -> Dict[str, Any]:
    """Simplified wrapper to call DoWhy estimation."""
    ate = estimate_dowhy_ate(data, treatment, outcome, common_causes, graph_str, method_name)
    return {f"dowhy_{method_name.split('.')[-1]}_ate": ate}

def run_double_ml_forest(data: pd.DataFrame, treatment: str, outcome: str, common_causes: List[str], **kwargs) -> Dict[str, Any]:
    """Simplified wrapper to call Causal Forest estimation."""
    # Note: Original class had double_ml_forest call estimate_ate_causalforest.
    # Keeping similar logic here, but name is potentially confusing.
    ate = estimate_causalforest_ate(data, treatment, outcome, common_causes, **kwargs)
    return {"causal_forest_ate": ate} 