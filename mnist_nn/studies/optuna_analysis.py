import argparse

import optuna
import pandas as pd


def print_list_of_studies(storage_path):
    """Helper function to print a list of all Optuna studies in the given storage."""
    print("List of Studies in Storage".center(60, "="))
    studies = optuna.study.get_all_study_summaries(storage=f"sqlite:///{storage_path}")
    for study_summary in studies:
        print(f" Study: {study_summary.study_name} ")


def print_top_10_values(study):
    """Helper function to print the values for the top 10 trials in the study."""
    print(" Study values for top 10 cases ".center(60, "="))

    df_trial = study.trials_dataframe()
    parameters_to_show = ["params_" + key for key in study.best_trial.params]
    parameters_to_show.extend(["value", "state"])
    df_trial = df_trial[df_trial["state"] == "COMPLETE"]
    df_trial = df_trial[df_trial["value"].notna()]
    df_trial = df_trial[parameters_to_show].sort_values(by="value", ascending=False)
    print("\nTop 10 Trials:")
    print(df_trial.head(10))


def print_study_summary(study):
    """Helper function to print a summary of the optuna study - number of trials etc."""
    print(f" Main Summary for Study: {study.study_name} ".center(60, "="))
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        print(f"Best trial value: {study.best_trial.value}")
    else:
        print("Best trial value: N/A (no completed trials yet)")

    df_trial = study.trials_dataframe()

    trial_states = df_trial["state"]
    state_counts = trial_states.value_counts()
    tot = {"Total": sum(state_counts)}
    trial_states_tot = pd.concat([state_counts, pd.Series(tot)], axis=0)
    print("\nTrial States:")
    print(trial_states_tot)


def parse_args():
    """Helper function to parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analysis of Optuna Study Results")
    parser.add_argument(
        "--study-name",
        dest="study_name",
        type=str,
        required=False,
        help="Name of the Optuna study",
        default="Optuna_batchsize",
    )
    parser.add_argument(
        "--study-db",
        dest="study_db",
        type=str,
        required=False,
        help="Path to the Optuna study database file",
        default=r"studies\Optuna_batchsize.db",
    )
    return parser.parse_args()


def print_best_trial(study):
    """Print out the best trial details from the study."""
    print(" Best trial ".center(60, "="))
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params: ")

    for param, value in best_trial.params.items():
        print(f"    {param}: {value}")


def print_most_important_params(study):
    """Print the most important parameters from the study, ranked most to least important."""
    print(" Most important params: ".center(60, "="))
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    values = [t.value for t in completed if t.value is not None]
    if len(set(values)) < 2:
        print("  Cannot compute importances yet: all completed trials have the same")
        print(f"  objective value ({values[0] if values else 'N/A'}) - need variance across trials.")
        return
    if len(completed) < 5:
        print(f"  Only {len(completed)} completed trials - need at least 5 to compute importances.")
        return
    try:
        evaluator = optuna.importance.MeanDecreaseImpurityImportanceEvaluator(seed=42)
        importances = optuna.importance.get_param_importances(study, evaluator=evaluator)
        bar_width = 30
        print(f"  {'Parameter':<30} {'Importance':>10}   {'Relative weight'}")
        print(f"  {'-'*30} {'-'*10}   {'-'*bar_width}")
        for param, importance in importances.items():
            bar = "#" * int(importance * bar_width)
            print(f"  {param:<30} {importance:>10.4f}   {bar}")
    except Exception as e:
        print(f"  Cannot compute importances: {e}")


def print_recommended_params(study):
    """Print out the recommended parameters from the study."""
    print(" Recommended values for our parameters ".center(60, "="))

    recommended_params = study.best_params
    for param_name, param_value in recommended_params.items():
        print(f"  {param_name}: {param_value}")


def main():
    args = parse_args()
    print_list_of_studies(args.study_db)

    study = optuna.load_study(
        study_name=args.study_name,
        storage=f"sqlite:///{args.study_db}",
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    n_complete = len(completed)
    n_total = len(study.trials)
    print(f"\nTrials: {n_complete} complete of {n_total} total")

    print_study_summary(study)

    if not completed:
        print("\nNo completed trials yet - skipping best-trial and importance analysis.")
        return

    print_best_trial(study)
    print_most_important_params(study)
    # print_recommended_params(study)
    print_top_10_values(study)


if __name__ == "__main__":
    main()