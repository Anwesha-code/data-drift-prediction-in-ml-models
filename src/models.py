from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from config import RANDOM_STATE


def get_model(model_type: str):
    """
    Return the requested model with sensible defaults.

    Available models:
      random_forest  - ensemble bagging, strongest on tabular data
      xgboost        - gradient boosted trees, fast and accurate
      logistic       - linear baseline, fast, interpretable
      decision_tree  - single tree, fully interpretable, drift-sensitive
      svm            - LinearSVC, large-margin linear classifier

    Note on SVM: Full kernel SVC is O(n^2) in memory and is unusable on
    2.8M records. LinearSVC scales linearly and gives equivalent results
    for linearly separable data. It does not expose predict_proba but
    predict() works normally for accuracy evaluation and cross-eval.
    """
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=1,
        ),
        "logistic": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=50,
            random_state=RANDOM_STATE,
        ),
        "svm": LinearSVC(
            C=1.0,
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Choose from: {list(models.keys())}"
        )

    return models[model_type]


def get_shap_explainer_type(model_type: str) -> str:
    """
    Return which SHAP explainer backend to use for a given model type.
    Called by shap_analysis.py to select the correct explainer.
    """
    if model_type in {"random_forest", "xgboost", "decision_tree"}:
        return "tree"
    elif model_type in {"logistic", "svm"}:
        return "linear"
    return "unknown"
