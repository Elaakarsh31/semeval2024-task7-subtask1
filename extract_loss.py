import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd


def extract_metrics_from_runs(root_dir, output_csv="training_metrics.csv"):
    """
    Extract metrics like Training Loss, Validation Loss, Micro F1, and Macro F1 from TensorBoard logs.

    Args:
        root_dir (str): Path to the root directory containing subfolders with `runs`.
        output_csv (str): Path to save the extracted metrics as a CSV file.

    Returns:
        None
    """
    metrics = []

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_file = os.path.join(subdir, file)
                print(f"Processing: {event_file}")

                try:
                    # Load the TensorBoard event file
                    if "runs" in subdir:
                        parent_folder = (
                            subdir.split("runs")[0].rstrip("/\\").split(os.sep)[-1]
                        )
                    else:
                        parent_folder = "unknown"
                    event_acc = EventAccumulator(event_file)
                    event_acc.Reload()

                    # Extract available scalar tags
                    tags = event_acc.Tags().get("scalars", [])

                    # Match specific tags
                    training_loss_tag = next(
                        (
                            tag
                            for tag in tags
                            if "training_loss" in tag.lower()
                            or "train_loss" in tag.lower()
                        ),
                        None,
                    )
                    validation_loss_tag = next(
                        (
                            tag
                            for tag in tags
                            if "validation_loss" in tag.lower()
                            or "val_loss" in tag.lower()
                        ),
                        None,
                    )
                    micro_f1_tag = next(
                        (tag for tag in tags if "micro_f1" in tag.lower()), None
                    )
                    macro_f1_tag = next(
                        (tag for tag in tags if "macro_f1" in tag.lower()), None
                    )

                    # Fetch scalar values for all metrics
                    max_epochs = max(
                        (
                            len(event_acc.Scalars(training_loss_tag))
                            if training_loss_tag
                            else 0
                        ),
                        (
                            len(event_acc.Scalars(validation_loss_tag))
                            if validation_loss_tag
                            else 0
                        ),
                        len(event_acc.Scalars(micro_f1_tag)) if micro_f1_tag else 0,
                        len(event_acc.Scalars(macro_f1_tag)) if macro_f1_tag else 0,
                    )

                    for epoch in range(max_epochs):
                        metrics.append(
                            {
                                "Folder": parent_folder,
                                "Epoch": epoch + 1,
                                "Training Loss": (
                                    event_acc.Scalars(training_loss_tag)[epoch].value
                                    if training_loss_tag
                                    and epoch
                                    < len(event_acc.Scalars(training_loss_tag))
                                    else None
                                ),
                                "Validation Loss": (
                                    event_acc.Scalars(validation_loss_tag)[epoch].value
                                    if validation_loss_tag
                                    and epoch
                                    < len(event_acc.Scalars(validation_loss_tag))
                                    else None
                                ),
                                "Micro F1": (
                                    event_acc.Scalars(micro_f1_tag)[epoch].value
                                    if micro_f1_tag
                                    and epoch < len(event_acc.Scalars(micro_f1_tag))
                                    else None
                                ),
                                "Macro F1": (
                                    event_acc.Scalars(macro_f1_tag)[epoch].value
                                    if macro_f1_tag
                                    and epoch < len(event_acc.Scalars(macro_f1_tag))
                                    else None
                                ),
                            }
                        )
                except Exception as e:
                    print(f"Error processing {event_file}: {e}")

    # Save results to a CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_csv, index=False)
    print(f"Metrics saved to {output_csv}")


# Example usage
root_directory = "./T5/models"  # Replace with your actual root directory
extract_metrics_from_runs(root_directory)
