import subprocess
import pandas as pd


def run_experiment(
    dataset_type,
    data_train_pth,
    data_dev_pth,
    data_test_pth,
    has_demonstrations,
    is_digit_base,
):
    output_model_path = f"./models/{data_train_pth.split('/')[-1].split('.')[0]}_demonstrations_{has_demonstrations}_digit_{is_digit_base}"
    output_file_name = f"results_{data_train_pth.split('/')[-1].split('.')[0]}_demonstrations_{has_demonstrations}_digit_{is_digit_base}.json"
    command = [
        "python",
        "instruction_tuning_qp.py",
        "--data_train_pth",
        data_train_pth,
        "--data_dev_pth",
        data_dev_pth,
        "--data_test_pth",
        data_test_pth,
        "--has_demonstrations",
        str(has_demonstrations),
        "--is_digit_base",
        str(is_digit_base),
        "--output_model_path",
        output_model_path,
        "--output_file_name",
        output_file_name,
        "--task",
        "predict",
        "--has_dev",
        str(True),
        "--dataset_type",
        dataset_type,
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    micro_f1, macro_f1 = None, None
    for line in process.stdout.split("\n"):
        if "micro_f1:" in line:
            micro_f1 = float(line.split(":")[1].strip())
            print(micro_f1)
        elif "macro_f1:" in line:
            macro_f1 = float(line.split(":")[1].strip())
            print(macro_f1)
    return micro_f1, macro_f1


def automate_experiments():
    data_folder_pth = "../Quantitative-101/QP"
    datasets = [
        {
            "dataset_type": "comment",
            "train": f"{data_folder_pth}/comment_subsampled_train.json",
            "dev": f"{data_folder_pth}/comment_subsampled_dev.json",
            "test": f"{data_folder_pth}/Numeracy600K_comment_test.json",
        },
        {
            "dataset_type": "headline",
            "train": f"{data_folder_pth}/headline_subsampled_train.json",
            "dev": f"{data_folder_pth}/headline_subsampled_dev.json",
            "test": f"{data_folder_pth}/Numeracy600K_headline_test.json",
        },
    ]
    # Create multi-level columns for DataFrame
    dataset_names = [d["dataset_type"] for d in datasets]
    metrics = ["micro_f1", "macro_f1"]
    columns = pd.MultiIndex.from_product([dataset_names, metrics])

    # Initialize results DataFrame
    results = pd.DataFrame(
        index=["icl_org", "inst_org", "icl_digit", "inst_digit"], columns=columns
    )

    for dataset in datasets:
        dataset_type, data_train_pth, data_dev_pth, data_test_pth = dataset.values()
        # Run all combinations and store both metrics
        for setting, (demo, digit) in {
            "icl_org": (True, False),
            "inst_org": (False, False),
            "icl_digit": (True, True),
            "inst_digit": (False, True),
        }.items():
            micro, macro = run_experiment(
                dataset_type, data_train_pth, data_dev_pth, data_test_pth, demo, digit
            )
            results.loc[setting, (dataset_type, "micro_f1")] = micro
            results.loc[setting, (dataset_type, "macro_f1")] = macro

    # Save results
    results.to_csv("qp_experiment_results.csv")
    print("\nFinal Results:")
    print(results)


if __name__ == "__main__":
    automate_experiments()
