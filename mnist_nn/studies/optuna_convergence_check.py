"""
Convergence analysis for Optuna studies.
Checks if performance is still improving or if the study has plateaued.
"""

import optuna
from pathlib import Path
import argparse


def analyze_convergence(db_path: str, study_name: str) -> None:
    """Analyze if a study is still making progress."""

    db = Path(db_path)
    if not db.exists():
        print(f"Error: Database not found: {db_path}")
        return

    try:
        storage_url = f"sqlite:///{db.as_posix()}"
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception as e:
        print(f"Error loading study: {e}")
        return

    # Get all completed trials
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]

    if not trials:
        print("No completed trials found.")
        return

    trial_ids = [t.number for t in trials]
    values = [t.value for t in trials]

    total_trials = len(trials)

    print("\n" + "="*70)
    print("  CONVERGENCE ANALYSIS")
    print("="*70)

    # 1. Best value and when it was achieved
    best_value = max(values)
    best_trial_idx = values.index(best_value)
    best_trial_id = trial_ids[best_trial_idx]
    trials_since_best = total_trials - best_trial_idx - 1

    print(f"\nBest Value: {best_value:.2f}% (Trial #{best_trial_id})")
    print(f"Trials since last improvement: {trials_since_best}")

    # 2. Last 10 vs Top 10
    top_10_values = sorted(values, reverse=True)[:10]
    top_10_threshold = top_10_values[-1]

    last_10_trials = trial_ids[-10:] if len(trial_ids) >= 10 else trial_ids
    last_10_in_top_10 = sum(1 for i, tid in enumerate(last_10_trials)
                             if values[trial_ids.index(tid)] >= top_10_threshold)

    print(f"\nLast 10 trials in Top 10: {last_10_in_top_10}/10")
    if last_10_in_top_10 == 0:
        print("  ⚠️  No recent trials in top 10 — study may be converged")
    elif last_10_in_top_10 >= 5:
        print("  ✓ Active improvement — study is still exploring")
    else:
        print("  ≈ Moderate activity — slowing down")

    # 3. Improvement trend
    if total_trials >= 20:
        first_10_mean = sum(values[:10]) / 10
        last_10_mean = sum(values[-10:]) / 10
        improvement_pct = ((last_10_mean - first_10_mean) / first_10_mean) * 100

        print(f"\nImprovement trend (first 10 vs last 10):")
        print(f"  First 10 mean: {first_10_mean:.2f}%")
        print(f"  Last 10 mean:  {last_10_mean:.2f}%")
        print(f"  Improvement:   {improvement_pct:+.2f}%")

        if improvement_pct > 1:
            print("  ✓ Positive trend — keep running")
        elif improvement_pct > -1:
            print("  ≈ Flat — diminishing returns")
        else:
            print("  ✗ Negative trend — consider stopping")

    # 4. Convergence score
    if total_trials >= 20:
        recent_variance = max(values[-10:]) - min(values[-10:])
        overall_variance = max(values) - min(values)

        if overall_variance > 0:
            compression = 1 - (recent_variance / overall_variance)
        else:
            compression = 0

        print(f"\nConvergence score: {compression*100:.1f}%")
        if compression > 0.8:
            print("  ✓ Highly converged — consider stopping")
        elif compression > 0.5:
            print("  ≈ Moderately converged — limited upside")
        else:
            print("  ✗ Still exploring — room for improvement")

    # 5. Recommendation
    print("\n" + "-"*70)
    print("Recommendation:")

    should_continue = False
    reasons = []

    if trials_since_best < 10:
        should_continue = True
        reasons.append("  • Recent improvement found (<10 trials ago)")

    if last_10_in_top_10 >= 4:
        should_continue = True
        reasons.append("  • Active optimization (4+ of last 10 in top 10)")

    if total_trials < 40:
        should_continue = True
        reasons.append("  • Insufficient trials for strong convergence signal")

    if total_trials >= 20:
        if improvement_pct > 1:
            should_continue = True
            reasons.append("  • Positive trend continuing")

    if should_continue:
        if reasons:
            print("CONTINUE — Valid reasons to run more trials:")
            for r in reasons:
                print(r)
        else:
            print("CONTINUE — Study still has potential")
    else:
        print("CONSIDER STOPPING — Study appears converged:")
        print("  • No recent improvements")
        print("  • Last 10 trials not in top 10")
        print("  • Diminishing returns visible")

    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-db", required=True, help="Path to Optuna database")
    parser.add_argument("--study-name", required=True, help="Study name")
    args = parser.parse_args()

    analyze_convergence(args.study_db, args.study_name)
