import os  # Imports the os module to interact with the operating system, like file paths. / Importiert das os-Modul zur Interaktion mit dem Betriebssystem, z.B. Dateipfade.
import argparse  # Imports argparse to handle command-line arguments. / Importiert argparse, um Befehlszeilenargumente zu verarbeiten.
from model import Tedd1104ModelPLForImageReordering  # Imports a specific model class for image reordering. / Importiert eine spezielle Modellklasse für die Bildumordnung.
from dataset_image_reordering import Tedd1104Dataset  # Imports a dataset class related to image reordering. / Importiert eine Datensatzklasse zur Bildumordnung.
import pytorch_lightning as pl  # Imports PyTorch Lightning for high-level model training and testing. / Importiert PyTorch Lightning für das Training und Testen von Modellen.
from typing import List, Union  # Imports types for annotations, specifically List and Union. / Importiert Typen für Anmerkungen, speziell List und Union.
from torch.utils.data import DataLoader  # Imports DataLoader for batch loading of datasets. / Importiert DataLoader zum Laden von Datensätzen in Batches.
from tabulate import tabulate  # Imports tabulate for printing data in table format. / Importiert tabulate zum Drucken von Daten im Tabellenformat.
from dataset import collate_fn, set_worker_sharing_strategy  # Imports dataset utility functions. / Importiert Hilfsfunktionen für den Datensatz.
from pytorch_lightning import loggers as pl_loggers  # Imports PyTorch Lightning loggers for reporting metrics. / Importiert PyTorch Lightning Logger zum Berichten von Metriken.

def eval_model(  # Defines the function to evaluate a model. / Definiert die Funktion zur Evaluierung eines Modells.
    checkpoint_path: str,  # Path to the model checkpoint file. / Pfad zur Modell-Checkpoint-Datei.
    test_dirs: List[str],  # List of directories containing test datasets. / Liste von Verzeichnissen mit Testdatensätzen.
    batch_size: int,  # Batch size for loading data. / Batch-Größe zum Laden von Daten.
    dataloader_num_workers: int = 16,  # Number of workers for loading data. / Anzahl der Arbeiter zum Laden von Daten.
    output_path: str = None,  # Path to save the results. / Pfad zum Speichern der Ergebnisse.
    devices: str = 1,  # Number of devices (GPUs/TPUs) to use. / Anzahl der Geräte (GPUs/TPUs), die verwendet werden sollen.
    accelerator: str = "auto",  # Type of accelerator (auto, GPU, TPU). / Art des Beschleunigers (auto, GPU, TPU).
    precision: str = "bf16",  # Precision to use for model computations (bf16, 16, 32, etc.). / Präzision, die für Modellberechnungen verwendet wird (bf16, 16, 32 usw.).
    strategy=None,  # Data parallelism strategy (None, ddp, etc.). / Strategie für Datenparallelität (None, ddp usw.).
    report_to: str = "none",  # Where to report metrics (None, TensorBoard, W&B). / Wo Metriken gemeldet werden sollen (None, TensorBoard, W&B).
    experiment_name: str = "test",  # Name of the experiment for logging. / Name des Experiments für das Logging.
):
    """
    Evaluates a trained model on a set of test data.
    / Bewertet ein trainiertes Modell auf einem Satz von Testdaten.
    """

    if not os.path.exists(os.path.dirname(output_path)):  # Checks if the output path directory exists. / Überprüft, ob das Verzeichnis des Ausgabe-Pfades existiert.
        os.makedirs(os.path.dirname(output_path))  # Creates the directory if it doesn't exist. / Erstellt das Verzeichnis, falls es nicht existiert.

    print(f"Restoring model from {checkpoint_path}")  # Prints the checkpoint path. / Gibt den Checkpoint-Pfad aus.
    model = Tedd1104ModelPLForImageReordering.load_from_checkpoint(  # Loads the trained model from the checkpoint. / Lädt das trainierte Modell aus dem Checkpoint.
        checkpoint_path=checkpoint_path
    )

    if report_to == "tensorboard":  # Checks if results should be reported to TensorBoard. / Überprüft, ob Ergebnisse an TensorBoard gemeldet werden sollen.
        logger = pl_loggers.TensorBoardLogger(  # Creates a TensorBoard logger. / Erstellt einen TensorBoard Logger.
            save_dir=os.path.dirname(checkpoint_path),
            name=experiment_name,
        )
    elif report_to == "wandb":  # Checks if results should be reported to Weights & Biases (W&B). / Überprüft, ob Ergebnisse an W&B gemeldet werden sollen.
        logger = pl_loggers.WandbLogger(  # Creates a W&B logger. / Erstellt einen W&B Logger.
            name=experiment_name,
            project="TEDD1104",
            save_dir=os.path.dirname(checkpoint_path),
        )
    elif report_to == "none":  # No logging if 'none'. / Kein Logging, wenn 'none'.
        logger = None
    else:
        raise ValueError(  # Raises an error if an unknown logger is specified. / Wirft einen Fehler, wenn ein unbekannter Logger angegeben wird.
            f"Unknown logger: {report_to}. Please use 'tensorboard' or 'wandb'."
        )

    trainer = pl.Trainer(  # Initializes the PyTorch Lightning trainer for testing. / Initialisiert den PyTorch Lightning Trainer für Tests.
        devices=devices,  # Specifies the number of devices (GPUs/TPUs) to use. / Gibt die Anzahl der zu verwendenden Geräte an.
        accelerator=accelerator,  # Specifies the type of accelerator to use (auto, GPU, etc.). / Gibt den Typ des zu verwendenden Beschleunigers an (auto, GPU usw.).
        precision=precision if precision == "bf16" else int(precision),  # Sets the precision (bf16, 16, etc.). / Setzt die Präzision (bf16, 16 usw.).
        strategy=strategy,  # Sets the parallelization strategy. / Setzt die Parallelisierungsstrategie.
    )

    results: List[List[Union[str, float]]] = []  # List to store the results of the tests. / Liste zur Speicherung der Testergebnisse.
    for test_dir in test_dirs:  # Loops through each test directory. / Schleift durch jedes Testverzeichnis.

        dataloader = DataLoader(  # Initializes the DataLoader for batch loading. / Initialisiert den DataLoader zum Laden von Batches.
            Tedd1104Dataset(  # Loads the dataset for image reordering. / Lädt den Datensatz für die Bildumordnung.
                dataset_dir=test_dir,
                hide_map_prob=0.0,
                dropout_images_prob=[0.0, 0.0, 0.0, 0.0, 0.0],
                token_mask_prob=0.0,
                train=False,
                transformer_nheads=None
                if model.encoder_type == "lstm"
                else model.nhead,
            ),
            batch_size=batch_size,  # Batch size for the dataloader. / Batch-Größe für den DataLoader.
            num_workers=dataloader_num_workers,  # Number of workers for loading data. / Anzahl der Arbeiter für das Laden von Daten.
            pin_memory=True,  # Pin memory to accelerate data loading. / Speichert Daten im RAM, um das Laden zu beschleunigen.
            shuffle=False,  # Disables shuffling as we are testing. / Deaktiviert das Mischen, da wir testen.
            persistent_workers=True,  # Keeps workers alive across epochs. / Hält die Arbeiter zwischen den Epochen am Leben.
            collate_fn=collate_fn,  # Function to merge batches of data. / Funktion zum Zusammenführen von Daten-Batches.
            worker_init_fn=set_worker_sharing_strategy,  # Initializes worker sharing strategy. / Initialisiert die Strategie zum Teilen von Arbeitern.
        )
        print(f"Testing dataset: {os.path.basename(test_dir)}: ")  # Prints the current test directory. / Gibt das aktuelle Testverzeichnis aus.
        print()

        out = trainer.test(  # Runs the testing process. / Führt den Testprozess aus.
            ckpt_path=checkpoint_path,  # Path to the checkpoint file. / Pfad zur Checkpoint-Datei.
            model=model,  # The model to be tested. / Das zu testende Modell.
            dataloaders=[dataloader],  # Passes the dataloaders to the trainer. / Übergibt die DataLoader an den Trainer.
            verbose=False,  # Disables verbose output. / Deaktiviert die ausführliche Ausgabe.
        )[0]

        results.append(  # Appends the result to the results list. / Fügt das Ergebnis zur Ergebnisliste hinzu.
            [
                os.path.basename(test_dir),  # Name of the test directory. / Name des Testverzeichnisses.
                round(out["Test/acc"] * 100, 1),  # Test accuracy rounded to 1 decimal place. / Testgenauigkeit auf eine Dezimalstelle gerundet.
            ]
        )

        if logger is not None:  # Checks if logging is enabled. / Überprüft, ob Logging aktiviert ist.
            log_metric_dict = {}  # Initializes a dictionary to store metrics. / Initialisiert ein Wörterbuch zur Speicherung der Metriken.
            for metric_name, metric_value in out.items():  # Loops through each metric. / Schleift durch jede Metrik.
                log_metric_dict[
                    f"{os.path.basename(test_dir)}/{metric_name.split('/')[-1]}"
                ] = metric_value  # Formats the metric name and stores the value. / Formatiert den Metriknamen und speichert den Wert.
            logger.log_metrics(log_metric_dict, step=0)  # Logs the metrics. / Protokolliert die Metriken.

    print(
        tabulate(  # Prints the results as a table. / Gibt die Ergebnisse als Tabelle aus.
            results,
            headers=[
                "Accuracy",  # Table header for the accuracy column. / Tabellenüberschrift für die Genauigkeitsspalte.
            ],
        )
    )

    if output_path:  # Checks if an output path is provided. / Überprüft, ob ein Ausgabe-Pfad angegeben wurde.
        with open(output_path, "w+", encoding="utf8") as output_file:  # Opens the output file for writing. / Öffnet die Ausgabedatei zum Schreiben.
            print(
                tabulate(  # Prints the results in a TSV format. / Gibt die Ergebnisse im TSV-Format aus.
                    results,
                    headers=[
                        "Accuracy",  # Table header for the accuracy column. / Tabellenüberschrift für die Genauigkeitsspalte.
                    ],
                    tablefmt="tsv",  # Sets the table format to TSV. / Setzt das Tabellenformat auf TSV.
                ),
                file=output_file,  # Writes the table to the file. / Schreibt die Tabelle in die Datei.
            )


if __name__ == "__main__":  # Checks if the script is being run directly. / Überprüft, ob das Skript direkt ausgeführt wird.

    parser = argparse.ArgumentParser(  # Creates an argument parser. / Erstellt einen Argument-Parser.
        description="Evaluate a trained model on the image reordering task."  # Description of the script's purpose. / Beschreibung des Zwecks des Skripts.
    )

    parser.add_argument(
        "--checkpoint_path",  # Command-line argument for the checkpoint file path. / Befehlszeilenargument für den Pfad zur Checkpoint-Datei.
        type=str,  # Specifies the type of the argument. / Gibt den Typ des Arguments an.
        help="Path to the checkpoint file.",  # Description of the argument. / Beschreibung des Arguments.
    )

    parser.add_argument(
        "--test_dirs",  # Command-line argument for directories with test data. / Befehlszeilenargument für Verzeichnisse mit Testdaten.
        type=str,
        nargs="+",  # Allows multiple directories to be specified. / Ermöglicht die Angabe mehrerer Verzeichnisse.
        help="List of directories containing test data.",  # Description of the argument. / Beschreibung des Arguments.
    )

    parser.add_argument(
        "--batch_size",  # Command-line argument for batch size. / Befehlszeilenargument für die Batch-Größe.
        type=int,  # Specifies the type of the argument. / Gibt den Typ des Arguments an.
        required=True,  # Makes the argument mandatory. / Macht das Argument erforderlich.
        help="Batch size for the dataloader.",  # Description of the argument. / Beschreibung des Arguments.
    )

    parser.add_argument(
        "--dataloader_num_workers",  # Command-line argument for the number of workers for data loading. / Befehlszeilenargument für die Anzahl der Arbeiter zum Laden von Daten.
        type=int,
        default=min(os.cpu_count(), 16),  # Default value is the minimum of the CPU count and 16. / Der Standardwert ist das Minimum der CPU-Anzahl und 16.
        help="Number of workers for the dataloader.",  # Description of the argument. / Beschreibung des Arguments.
    )

    parser.add_argument(
        "--output_path",  # Command-line argument for the output file path. / Befehlszeilenargument für den Ausgabe-Dateipfad.
        type=str,
        default=None,  # Default is None (no output path). / Der Standardwert ist None (kein Ausgabe-Pfad).
        help="Path to where the results should be saved.",  # Description of the argument. / Beschreibung des Arguments.
    )

    parser.add_argument(
        "--devices",  # Command-line argument for the number of devices (GPUs/TPUs). / Befehlszeilenargument für die Anzahl der Geräte (GPUs/TPUs).
        type=int,
        default=1,  # Default is 1 device. / Der Standardwert ist 1 Gerät.
        help="Number of GPUs/TPUs to use. ",  # Description of the argument. / Beschreibung des Arguments.
    )

    parser.add_argument(
        "--accelerator",  # Command-line argument for the accelerator type (GPU, TPU, etc.). / Befehlszeilenargument für den Beschleuniger-Typ (GPU, TPU usw.).
        type=str,
        default="auto",  # Default is "auto" (detects the system). / Der Standardwert ist "auto" (erkennt das System).
        choices=["auto", "tpu", "gpu", "cpu", "ipu"],  # Specifies valid options for the argument. / Gibt gültige Optionen für das Argument an.
        help="Accelerator to use. If 'auto', tries to automatically detect TPU, GPU, CPU or IPU system",  # Description of the argument. / Beschreibung des Arguments.
    )

    parser.add_argument(
        "--precision",  # Command-line argument for precision (16, 32, 64, bf16). / Befehlszeilenargument für Präzision (16, 32, 64, bf16).
        type=str,
        default="16",  # Default is 16-bit precision. / Der Standardwert ist 16-Bit-Präzision.
        choices=["bf16", "16", "32", "64"],  # Specifies valid options for precision. / Gibt gültige Optionen für Präzision an.
        help=" Double precision (64), full precision (32), "
        "half precision (16) or bfloat16 precision (bf16). "
        "Can be used on CPU, GPU or TPUs.",  # Description of the argument. / Beschreibung des Arguments.
    )

    parser.add_argument(
        "--strategy",  # Command-line argument for data parallelism strategy. / Befehlszeilenargument für die Strategie der Datenparallelität.
        type=str,
        default=None,  # Default is None (no parallelism). / Der Standardwert ist None (keine Parallelität).
        help="Supports passing different training strategies with aliases (ddp, ddp_spawn, etc)",  # Description of the argument. / Beschreibung des Arguments.
    )

    parser.add_argument(
        "--report_to",  # Command-line argument for where to report the metrics (wandb, tensorboard). / Befehlszeilenargument für den Bericht über Metriken (wandb, tensorboard).
        type=str,
        default="wandb",  # Default is "wandb" for reporting. / Der Standardwert ist "wandb" für das Reporting.
        choices=["wandb", "tensorboard", "none"],  # Specifies valid options for reporting. / Gibt gültige Optionen für das Reporting an.
        help="Report to wandb or tensorboard",  # Description of the argument. / Beschreibung des Arguments.
    )

    parser.add_argument(
        "--experiment_name",  # Command-line argument for the experiment name. / Befehlszeilenargument für den Experimentnamen.
        type=str,
        default="wandb",  # Default is "wandb". / Der Standardwert ist "wandb".
        choices=["wandb", "tensorboard", "none"],  # Specifies valid options for experiment name. / Gibt gültige Optionen für den Experimentnamen an.
        help="Report to wandb or tensorboard",  # Description of the argument. / Beschreibung des Arguments.
    )

    args = parser.parse_args()  # Parses the command-line arguments. / Parst die Befehlszeilenargumente.

    eval_model(  # Calls the eval_model function with the parsed arguments. / Ruft die Funktion eval_model mit den geparsten Argumenten auf.
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
