# ============================================================
# src/model_training.py
# Train Multiple Models: LR, RF, XGBoost, LightGBM
# With Cross-Validation, SMOTE, Hyperparameter Tuning
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics         import (classification_report, confusion_matrix,
                                      roc_auc_score, f1_score, accuracy_score,
                                      precision_score, recall_score)
from imblearn.over_sampling  import SMOTE
from colorama                import Fore, Style, init
init(autoreset=True)


def load_splits(data_dir: str) -> tuple:
    """Load pre-processed train/test splits."""
    data_dir = Path(data_dir)
    X_train  = pd.read_csv(data_dir / "X_train.csv")
    X_test   = pd.read_csv(data_dir / "X_test.csv")
    y_train  = pd.read_csv(data_dir / "y_train.csv").squeeze()
    y_test   = pd.read_csv(data_dir / "y_test.csv").squeeze()
    print(f"[LOAD] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[LOAD] Class balance (train) → 0: {(y_train==0).sum()} | 1: {(y_train==1).sum()}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state: int = 42):
    """Apply SMOTE to handle class imbalance."""
    sm = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[SMOTE] Before: {y_train.value_counts().to_dict()}")
    print(f"[SMOTE] After : {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


def get_models() -> dict:
    """
    Return all models with tuned hyperparameters.
    Option C: Full model suite.
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=3,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, n_jobs=-1,
            verbose=-1
        ),
    }


def cross_validate_models(models: dict, X_train, y_train, cv: int = 5) -> dict:
    """Run stratified k-fold CV on all models."""
    print("\n" + "="*60)
    print("  CROSS VALIDATION (5-Fold Stratified)")
    print("="*60)

    cv_results = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train,
                                  cv=skf, scoring="roc_auc", n_jobs=-1)
        cv_results[name] = {
            "mean_auc"  : scores.mean(),
            "std_auc"   : scores.std(),
            "all_scores": scores.tolist()
        }
        print(f"  {Fore.CYAN}{name:<25}{Style.RESET_ALL} "
              f"AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    return cv_results


def train_and_evaluate(models: dict, X_train, y_train, X_test, y_test) -> tuple:
    """
    Train all models and evaluate on test set.
    Returns trained models and metrics dictionary.
    """
    print("\n" + "="*60)
    print("  MODEL TRAINING & EVALUATION")
    print("="*60)

    results      = {}
    trained_mods = {}

    for name, model in models.items():
        print(f"\n  Training: {Fore.YELLOW}{name}{Style.RESET_ALL} ...")
        model.fit(X_train, y_train)

        y_pred      = model.predict(X_test)
        y_prob      = (model.predict_proba(X_test)[:, 1]
                       if hasattr(model, "predict_proba")
                       else model.decision_function(X_test))

        acc         = accuracy_score(y_test,  y_pred)
        prec        = precision_score(y_test, y_pred, zero_division=0)
        rec         = recall_score(y_test,    y_pred, zero_division=0)
        f1          = f1_score(y_test,        y_pred, zero_division=0)
        auc         = roc_auc_score(y_test,   y_prob)
        cm          = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy"  : acc,
            "precision" : prec,
            "recall"    : rec,
            "f1_score"  : f1,
            "roc_auc"   : auc,
            "confusion_matrix": cm,
            "y_pred"    : y_pred,
            "y_prob"    : y_prob
        }
        trained_mods[name] = model

        print(f"    Accuracy : {acc:.4f}  |  Precision: {prec:.4f}")
        print(f"    Recall   : {rec:.4f}  |  F1 Score : {f1:.4f}")
        print(f"    ROC-AUC  : {Fore.GREEN}{auc:.4f}{Style.RESET_ALL}")

    return trained_mods, results


def select_best_model(results: dict) -> str:
    """Select best model based on ROC-AUC score."""
    best = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\n{Fore.GREEN}[BEST MODEL] {best} "
          f"(ROC-AUC: {results[best]['roc_auc']:.4f}){Style.RESET_ALL}")
    return best


def print_leaderboard(results: dict):
    """Print a sorted leaderboard of all models."""
    print("\n" + "="*60)
    print("  MODEL LEADERBOARD (sorted by ROC-AUC)")
    print("="*60)
    sorted_r = sorted(results.items(), key=lambda x: x[1]["roc_auc"], reverse=True)
    print(f"  {'Rank':<5} {'Model':<25} {'AUC':>8} {'F1':>8} {'Recall':>8} {'Prec':>8}")
    print("  " + "-"*58)
    for rank, (name, m) in enumerate(sorted_r, 1):
        star = " ⭐" if rank == 1 else ""
        print(f"  {rank:<5} {name:<25} {m['roc_auc']:>8.4f} "
              f"{m['f1_score']:>8.4f} {m['recall']:>8.4f} {m['precision']:>8.4f}{star}")


def run_training_pipeline(data_dir: str, models_dir: str) -> tuple:
    """
    Full training pipeline:
    load → SMOTE → CV → train → evaluate → save best model
    """
    data_dir   = Path(data_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_splits(data_dir)

    # Apply SMOTE on training set only
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    models       = get_models()
    cv_results   = cross_validate_models(models, X_train_bal, y_train_bal)
    trained_mods, results = train_and_evaluate(models, X_train_bal, y_train_bal, X_test, y_test)

    print_leaderboard(results)
    best_name = select_best_model(results)

    # Save all models
    for name, model in trained_mods.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, models_dir / f"{safe_name}.pkl")
        print(f"[SAVED] {name} → {models_dir / f'{safe_name}.pkl'}")

    # Save best model reference
    best_model = trained_mods[best_name]
    joblib.dump(best_model, models_dir / "best_model.pkl")

    # Save results summary
    summary = {name: {k: v for k, v in m.items()
                       if k not in ("y_pred", "y_prob", "confusion_matrix")}
               for name, m in results.items()}
    pd.DataFrame(summary).T.to_csv(models_dir / "model_results_summary.csv")
    print(f"[SAVED] Results summary → {models_dir / 'model_results_summary.csv'}")

    # Save CV results
    pd.DataFrame(cv_results).T.to_csv(models_dir / "cv_results.csv")

    return trained_mods, results, best_name, X_test, y_test


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    run_training_pipeline(
        data_dir   = str(base / "data"),
        models_dir = str(base / "models")
    )
