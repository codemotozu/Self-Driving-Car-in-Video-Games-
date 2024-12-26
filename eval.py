import os  # Import the os module to interact with the operating system, e.g., to create directories.
import argparse  # Import argparse for command-line argument parsing.
from model import Tedd1104ModelPL  # Import the model class used for evaluation from the 'model' module.
from dataset import Tedd1104Dataset  # Import the dataset class for loading test data from the 'dataset' module.
import pytorch_lightning as pl  # Import PyTorch Lightning for simplifying the training and evaluation process.
from typing import List, Union  # Import List and Union for type hinting in Python.
from torch.utils.data import DataLoader  # Import DataLoader to handle batching and data loading.
from tabulate import tabulate  # Import tabulate to format the results into a readable table.
from dataset import collate_fn, set_worker_sharing_strategy  # Import specific functions for collating and worker strategies.
from pytorch_lightning import loggers as pl_loggers  # Import logging tools for tracking experiments (TensorBoard, WandB).

# Function to evaluate a trained model
def eval_model(
    checkpoint_path: str,  # Path to the model checkpoint.
    test_dirs: List[str],  # List of directories with test data.
    batch_size: int,  # Batch size for the DataLoader.
    dataloader_num_workers: int = 16,  # Number of workers for the DataLoader.
    output_path: str = None,  # Path to save the output results.
    devices: str = 1,  # Number of devices (GPUs/TPUs) to use.
    accelerator: str = "auto",  # Type of accelerator (GPU/TPU).
    precision: str = "16",  # Precision to use (e.g., float16, float32).
    strategy=None,  # Strategy for data parallelism (None, ddp, etc.).
    report_to: str = "none",  # Reporting destination (none, tensorboard, wandb).
    experiment_name: str = "test",  # Experiment name for reporting.
):
    """
    Evaluates a trained model on a set of test data.
    Bewertet ein trainiertes Modell anhand einer Reihe von Testdaten.

    :param str checkpoint_path: Path to the checkpoint file. (Pfad zur Checkpoint-Datei)
    :param List[str] test_dirs: List of directories containing test data. (Liste der Verzeichnisse mit Testdaten)
    :param int batch_size: Batch size for the dataloader. (Batchgröße für den Dataloader)
    :param int dataloader_num_workers: Number of workers for the dataloader. (Anzahl der Worker für den Dataloader)
    :param str output_path: Path to where the results should be saved. (Pfad, an dem die Ergebnisse gespeichert werden sollen)
    :param str devices: Number of devices to use. (Anzahl der zu verwendenden Geräte)
    :param str accelerator: Accelerator to use. (Beschleuniger, der verwendet werden soll)
    :param str precision: Precision to use (16, 32, 64). (Präzision, die verwendet werden soll)
    :param str strategy: Strategy for data parallelism. (Strategie für Datenparallelismus)
    :param str report_to: Where to report the results (none, tensorboard, wandb). (Wo die Ergebnisse gemeldet werden sollen)
    :param str experiment_name: Name of the experiment for reporting. (Name des Experiments zur Berichterstattung)
    """
    
    # Check if the output directory exists, create it if not.
    # Überprüfen, ob das Ausgabeverzeichnis existiert, andernfalls erstellen.
    if not os.path.exists(os.path.dirname(output_path)):  # If output directory doesn't exist
        os.makedirs(os.path.dirname(output_path))  # Create it.

    # Load the model from the checkpoint.
    # Modell aus dem Checkpoint laden.
    print(f"Restoring model from {checkpoint_path}")  # Print message showing checkpoint path.
    model = Tedd1104ModelPL.load_from_checkpoint(checkpoint_path=checkpoint_path)  # Load the model.

    # Initialize the logger depending on the 'report_to' parameter.
    # Initialisiere den Logger abhängig vom Parameter 'report_to'.
    if report_to == "tensorboard":
        logger = pl_loggers.TensorBoardLogger(  # Initialize TensorBoard logger.
            save_dir=os.path.dirname(checkpoint_path),
            name=experiment_name,
        )
    elif report_to == "wandb":
        logger = pl_loggers.WandbLogger(  # Initialize WandB logger.
            name=experiment_name,
            project="TEDD1104",
            save_dir=os.path.dirname(checkpoint_path),
        )
    elif report_to == "none":
        logger = None  # No logging if report_to is "none".
    else:
        raise ValueError(
            f"Unknown logger: {report_to}. Please use 'tensorboard' or 'wandb'."  # Error if unknown logger.
        )

    # Initialize the PyTorch Lightning Trainer.
    # Initialisiere den PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        devices=devices,  # Number of devices to use.
        accelerator=accelerator,  # Type of accelerator (auto, gpu, tpu).
        precision=precision if precision == "bf16" else int(precision),  # Set precision (16, 32, 64).
        strategy=strategy,  # Data parallelism strategy.
    )

    results: List[List[Union[str, float]]] = []  # Initialize an empty list for storing results.

    # Loop through the test directories and evaluate the model on each.
    # Schleife durch die Testverzeichnisse und bewerte das Modell in jedem.
    for test_dir in test_dirs:
        # Initialize the DataLoader for the test dataset.
        # Initialisiere den DataLoader für das Test-Dataset.
        dataloader = DataLoader(
            Tedd1104Dataset(  # Create the dataset instance.
                dataset_dir=test_dir,
                hide_map_prob=0.0,
                dropout_images_prob=[0.0, 0.0, 0.0, 0.0, 0.0],
                control_mode="keyboard",
                token_mask_prob=0.0,
                train=False,
                transformer_nheads=None if model.encoder_type == "lstm" else model.nhead,
            ),
            batch_size=batch_size,  # Set batch size.
            num_workers=dataloader_num_workers,  # Number of workers for loading data.
            pin_memory=True,  # Pin memory for faster data transfer.
            shuffle=False,  # Don't shuffle the data.
            persistent_workers=True,  # Keep workers alive between iterations.
            collate_fn=collate_fn,  # Use custom collate function.
            worker_init_fn=set_worker_sharing_strategy,  # Use custom worker init function.
        )

        print(f"Testing dataset: {os.path.basename(test_dir)}: ")  # Print the current test dataset name.
        print()

        # Perform model testing.
        # Durchführung des Modelltests.
        out = trainer.test(
            ckpt_path=checkpoint_path,  # Path to checkpoint.
            model=model,  # Model to test.
            dataloaders=[dataloader],  # Test dataloader.
            verbose=False,  # Don't print detailed output.
        )[0]

        # Collect the results and append to the results list.
        # Sammle die Ergebnisse und füge sie der Ergebnisliste hinzu.
        results.append(
            [
                os.path.basename(test_dir),
                round(out["Test/acc_k@1_micro"] * 100, 1),  # Accuracy at K@1 (micro).
                round(out["Test/acc_k@3_micro"] * 100, 1),  # Accuracy at K@3 (micro).
                round(out["Test/acc_k@1_macro"] * 100, 1),  # Accuracy at K@1 (macro).
                round(out["Test/acc_k@3_macro"] * 100, 1),  # Accuracy at K@3 (macro).
            ]
        )

        # Log the metrics to the logger if available.
        # Protokolliere die Metriken, wenn ein Logger vorhanden ist.
        if logger is not None:
            log_metric_dict = {}  # Initialize the dictionary for logging metrics.
            for metric_name, metric_value in out.items():
                log_metric_dict[
                    f"{os.path.basename(test_dir)}/{metric_name.split('/')[-1]}"
                ] = metric_value
            logger.log_metrics(log_metric_dict, step=0)  # Log the metrics.

    # Print the results in a tabular format.
    # Drucke die Ergebnisse in einem tabellarischen Format.
    print(
        tabulate(
            results,
            headers=[
                "Micro-Accuracy K@1",  # Header for micro accuracy at K@1.
                "Micro-Accuracy K@3",  # Header for micro accuracy at K@3.
                "Macro-Accuracy K@1",  # Header for macro accuracy at K@1.
                "Macro-Accuracy K@3",  # Header for macro accuracy at K@3.
            ],
        )
    )

    # If an output path is provided, save the results to a file.
    # Wenn ein Ausgabepfad angegeben ist, speichere die Ergebnisse in einer Datei.
    if output_path:
        with open(output_path, "w+", encoding="utf8") as output_file:
            print(
                tabulate(
                    results,
                    headers=[
                        "Micro-Accuracy K@1",  # Header for micro accuracy at K@1.
                        "Micro-Accuracy K@3",  # Header for micro accuracy at K@3.
                        "Macro-Accuracy K@1",  # Header for macro accuracy at K@1.
                        "Macro-Accuracy K@3",  # Header for macro accuracy at K@3.
                    ],
                    tablefmt="tsv",  # Use tab-separated format.
                ),
                file=output_file,
            )

# Main function to parse arguments and call eval_model.
if __name__ == "__main__":  # If the script is executed directly (not imported).
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")  # Create argument parser.
    parser.add_argument(  # Define command-line arguments.
        "--checkpoint_path",  # Checkpoint file path argument.
        type=str,
        help="Path to the checkpoint file.",
    )

    parser.add_argument(
        "--test_dirs",  # Directories with test data.
        type=str,
        nargs="+",
        help="List of directories containing test data.",
    )

    parser.add_argument(
        "--batch_size",  # Batch size argument.
        type=int,
        required=True,
        help="Batch size for the dataloader.",
    )

    parser.add_argument(
        "--dataloader_num_workers",  # Number of workers for data loading.
        type=int,
        default=min(os.cpu_count(), 16),
        help="Number of workers for the dataloader.",
    )

    parser.add_argument(
        "--output_path",  # Path to save output results.
        type=str,
        default=None,
        help="Path to where the results should be saved.",
    )

    parser.add_argument(
        "--devices",  # Number of devices (GPUs/TPUs).
        type=int,
        default=1,
        help="Number of GPUs/TPUs to use. ",
    )

    parser.add_argument(
        "--accelerator",  # Type of accelerator to use.
        type=str,
        default="auto",
        choices=["auto", "tpu", "gpu", "cpu", "ipu"],
        help="Accelerator to use. If 'auto', tries to automatically detect TPU, GPU, CPU or IPU system",
    )

    parser.add_argument(
        "--precision",  # Precision (e.g., 16, 32, 64).
        type=str,
        default="32",
        choices=["bf16", "16", "32", "64"],
        help=" Double precision (64), full precision (32), "
        "half precision (16) or bfloat16 precision (bf16). "
        "Can be used on CPU, GPU or TPUs.",
    )

    parser.add_argument(
        "--strategy",  # Data parallelism strategy.
        type=str,
        default=None,
        help="Supports passing different training strategies with aliases (ddp, ddp_spawn, etc)",
    )

    parser.add_argument(
        "--report_to",  # Where to report results (e.g., wandb, tensorboard).
        type=str,
        default="wandb",
        choices=["wandb", "tensorboard", "none"],
        help="Report to wandb or tensorboard",
    )

    parser.add_argument(
        "--experiment_name",  # Name of the experiment for reporting.
        type=str,
        default="test",
        help="Experiment name for wandb",
    )

    args = parser.parse_args()  # Parse command-line arguments.

    eval_model(  # Call the eval_model function with parsed arguments.
        checkpoint_path=args.checkpoint_path,
        test_dirs=args.test_dirs,
        batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        output_path=args.output_path,
        devices=args.devices,
        accelerator=args.accelerator,
        precision=args.precision,
        strategy=args.strategy,
        report_to=args.report_to,
        experiment_name=args.experiment_name,
    )
