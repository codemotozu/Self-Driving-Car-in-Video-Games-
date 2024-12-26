from model import Tedd1104ModelPL  # Importing the Tedd1104 model from the model module. / Importieren des Tedd1104-Modells aus dem Modul model.
from typing import List  # Importing the List type for type hints. / Importieren des List-Typs für Typ-Hinweise.
import argparse  # Importing argparse for command-line argument parsing. / Importieren von argparse zur Analyse von Befehlszeilenargumenten.
from dataset import Tedd1104DataModule  # Importing a data handling module for the dataset. / Importieren eines Moduls zur Datenverarbeitung für den Datensatz.
import os  # Importing os module for operating system interactions. / Importieren des os-Moduls für Interaktionen mit dem Betriebssystem.
from pytorch_lightning import loggers as pl_loggers  # Importing logging utilities from PyTorch Lightning. / Importieren von Logging-Dienstprogrammen aus PyTorch Lightning.
import pytorch_lightning as pl  # Importing PyTorch Lightning for model training and experimentation. / Importieren von PyTorch Lightning für Modelltraining und Experimente.
from dataset import count_examples  # Importing a function to count dataset examples. / Importieren einer Funktion zum Zählen von Datensatzbeispielen.
import math  # Importing math module for mathematical functions. / Importieren des math-Moduls für mathematische Funktionen.

try:
    import wandb  # Trying to import Weights & Biases for experiment tracking. / Versuch, Weights & Biases zur Experimentverfolgung zu importieren.
    wandb.require("service")  # Ensures the use of a specific Wandb service mode. / Stellt sicher, dass ein bestimmter Wandb-Service-Modus verwendet wird.
except ImportError:
    wandb = None  # If Wandb is not available, set it to None. / Falls Wandb nicht verfügbar ist, wird es auf None gesetzt.

def train_new_model(  # Function definition to train a new model with various configurable options. / Funktionsdefinition zum Trainieren eines neuen Modells mit verschiedenen Konfigurationsoptionen.
    train_dir: str,  # Path to the training data directory. / Pfad zum Verzeichnis mit Trainingsdaten.
    val_dir: str,  # Path to the validation data directory. / Pfad zum Verzeichnis mit Validierungsdaten.
    output_dir: str,  # Path to save model outputs. / Pfad zum Speichern der Modellausgaben.
    batch_size: int,  # Batch size for training. / Batch-Größe für das Training.
    max_epochs: int,  # Maximum number of training epochs. / Maximale Anzahl von Trainingsepochen.
    cnn_model_name: str,  # Name of the CNN model to use. / Name des zu verwendenden CNN-Modells.
    devices: int = 1,  # Number of devices to use (e.g., GPUs). / Anzahl der zu verwendenden Geräte (z. B. GPUs).
    accelerator: str = "auto",  # Accelerator type (e.g., CPU, GPU). / Typ des Beschleunigers (z. B. CPU, GPU).
    precision: str = "bf16",  # Precision setting for training (e.g., bf16). / Präzisionseinstellung für das Training (z. B. bf16).
    strategy=None,  # Strategy for distributed training. / Strategie für verteiltes Training.
    accumulation_steps: int = 1,  # Gradient accumulation steps. / Schritte zur Gradientenakkumulation.
    hide_map_prob: float = 0.0,  # Probability of hiding map data. / Wahrscheinlichkeit, Kartendaten auszublenden.
    test_dir: str = None,  # Path to the test data directory. / Pfad zum Verzeichnis mit Testdaten.
    dropout_images_prob=None,  # Probability of dropping out images during training. / Wahrscheinlichkeit, Bilder während des Trainings auszuschließen.
    variable_weights: List[float] = None,  # List of weights for variable weighting. / Liste von Gewichten für variable Gewichtung.
    control_mode: str = "keyboard",  # Control mode (e.g., keyboard). / Steuerungsmodus (z. B. Tastatur).
    val_check_interval: float = 0.25,  # Validation check interval. / Intervall für Validierungsüberprüfungen.
    dataloader_num_workers=os.cpu_count(),  # Number of workers for data loading. / Anzahl der Arbeiter für das Laden von Daten.
    pretrained_cnn: bool = True,  # Use a pretrained CNN model. / Verwendung eines vortrainierten CNN-Modells.
    embedded_size: int = 512,  # Size of embedding vectors. / Größe der Einbettungsvektoren.
    nhead: int = 8,  # Number of attention heads in a transformer. / Anzahl der Attention-Köpfe in einem Transformer.
    num_layers_encoder: int = 1,  # Number of encoder layers. / Anzahl der Encoder-Schichten.
    lstm_hidden_size: int = 512,  # Hidden size for LSTM layers. / Verborgene Größe für LSTM-Schichten.
    dropout_cnn_out: float = 0.1,  # Dropout rate for CNN outputs. / Dropout-Rate für CNN-Ausgaben.
    positional_embeddings_dropout: float = 0.1,  # Dropout for positional embeddings. / Dropout für Positions-Einbettungen.
    dropout_encoder: float = 0.1,  # Dropout rate for encoder layers. / Dropout-Rate für Encoder-Schichten.
    dropout_encoder_features: float = 0.8,  # Dropout for encoder features. / Dropout für Encoder-Features.
    mask_prob: float = 0.0,  # Probability of masking inputs. / Wahrscheinlichkeit, Eingaben zu maskieren.
    sequence_size: int = 5,  # Sequence size for input data. / Sequenzgröße für Eingabedaten.
    encoder_type: str = "transformer",  # Type of encoder (e.g., transformer). / Typ des Encoders (z. B. Transformer).
    bidirectional_lstm=True,  # Use a bidirectional LSTM. / Verwendung eines bidirektionalen LSTM.
    checkpoint_path: str = None,  # Path to a checkpoint file. / Pfad zu einer Checkpoint-Datei.
    label_smoothing: float = None,  # Label smoothing value. / Wert für Label Glättung.
    report_to: str = "wandb",  # Tool to report results to (e.g., Wandb). / Tool zur Berichterstattung (z. B. Wandb).
    find_lr: bool = False,  # Whether to find a suitable learning rate. / Ob eine geeignete Lernrate gefunden werden soll.
    optimizer_name: str = "adamw",  # Optimizer name (e.g., AdamW). / Name des Optimierers (z. B. AdamW).
    scheduler_name: str = "linear",  # Scheduler name (e.g., linear). / Name des Schedulers (z. B. linear).
    learning_rate: float = 1e-5,  # Learning rate for the optimizer. / Lernrate für den Optimierer.
    weight_decay: float = 1e-3,  # Weight decay for regularization. / Gewichtszunahme zur Regularisierung.
    warmup_factor: float = 0.05,  # Warmup factor for learning rate scheduling. / Warmup-Faktor für die Lernratenplanung.
):


"""
Train a new model. # Explanation: This script trains a machine learning model. // Erklärung: Dieses Skript trainiert ein maschinelles Lernmodell.

:param str train_dir: The directory containing the training data. # Explanation: Path where training data is stored. // Erklärung: Pfad, in dem die Trainingsdaten gespeichert sind.
:param str val_dir: The directory containing the validation data. # Explanation: Path where validation data is stored. // Erklärung: Pfad, in dem die Validierungsdaten gespeichert sind.
:param str output_dir: The directory to save the model to. # Explanation: Path where the trained model will be saved. // Erklärung: Pfad, in dem das trainierte Modell gespeichert wird.
:param int batch_size: The batch size. # Explanation: Number of samples processed together during training. // Erklärung: Anzahl der gleichzeitig verarbeiteten Datenproben während des Trainings.
:param int accumulation_steps: The number of steps to accumulate gradients. # Explanation: Gradients will accumulate over these steps before update. // Erklärung: Gradienten werden über diese Schritte akkumuliert, bevor ein Update erfolgt.
:param int max_epochs: The maximum number of epochs to train for. # Explanation: Total iterations over the entire dataset. // Erklärung: Gesamte Anzahl der Durchläufe über den Datensatz.
:param float hide_map_prob: Probability of hiding the minimap (0<=hide_map_prob<=1). # Explanation: Chance to hide minimap during training. // Erklärung: Wahrscheinlichkeit, die Minikarte während des Trainings auszublenden.
:param float dropout_images_prob: Probability of dropping an image (0<=dropout_images_prob<=1). # Explanation: Chance to exclude an image from the dataset. // Erklärung: Wahrscheinlichkeit, ein Bild aus dem Datensatz auszuschließen.
:param str test_dir: The directory containing the test data. # Explanation: Path where test data is stored. // Erklärung: Pfad, in dem die Testdaten gespeichert sind.
:param str control_mode: Model output format: keyboard (Classification task: 9 classes) or controller (Regression task: 2 variables). # Explanation: Specifies task type (classification or regression). // Erklärung: Legt den Aufgabentyp fest (Klassifikation oder Regression).
:param int dataloader_num_workers: The number of workers to use for the dataloader. # Explanation: Number of threads for loading data. // Erklärung: Anzahl der Threads zum Laden von Daten.
:param int embedded_size: Size of the output embedding. # Explanation: Dimension of the model's feature embeddings. // Erklärung: Dimension der Feature-Embeddings des Modells.
:param float dropout_cnn_out: Dropout rate for the output of the CNN. # Explanation: Regularization for CNN output. // Erklärung: Regularisierung für die Ausgabe des CNN.
:param str cnn_model_name: Name of the CNN model from torchvision.models. # Explanation: Specifies the CNN model architecture. // Erklärung: Gibt die Architektur des CNN-Modells an.
:param float val_check_interval: The interval to check the validation accuracy. # Explanation: How often validation is performed during training. // Erklärung: Wie oft die Validierung während des Trainings durchgeführt wird.
:param int devices: Number of devices to use. # Explanation: Number of GPUs or CPUs available for training. // Erklärung: Anzahl der verfügbaren GPUs oder CPUs für das Training.
:param str accelerator: Accelerator to use. If 'auto', tries to automatically detect TPU, GPU, CPU, or IPU system. # Explanation: Specifies hardware acceleration for training. // Erklärung: Legt die Hardwarebeschleunigung für das Training fest.
:param str precision: Precision to use. Double precision (64), full precision (32), half precision (16), or bfloat16 precision (bf16). Can be used on CPU, GPU, or TPUs. # Explanation: Defines numeric precision for computations. // Erklärung: Legt die numerische Präzision für Berechnungen fest.
:param str strategy: Strategy to use for data parallelism. "None" for no data parallelism, ddp_find_unused_parameters_false for DDP. # Explanation: Parallel computing strategy. // Erklärung: Strategie für paralleles Rechnen.
:param str report_to: Where to report the results. "tensorboard" for TensorBoard, "wandb" for W&B. # Explanation: Specifies tools for logging results. // Erklärung: Gibt Tools zum Protokollieren der Ergebnisse an.
:param bool pretrained_cnn: If True, the model will be loaded with pretrained weights. # Explanation: Use pre-trained CNN weights. // Erklärung: Vortrainierte CNN-Gewichte verwenden.
:param int embedded_size: Size of the input feature vectors. # Explanation: Input embedding dimension. // Erklärung: Eingabe-Embedding-Dimension.
:param int nhead: Number of heads in the multi-head attention. # Explanation: Multi-head attention count in Transformer. // Erklärung: Anzahl der Multi-Head-Attention im Transformer.
:param int num_layers_encoder: Number of transformer layers in the encoder. # Explanation: Encoder depth in Transformer. // Erklärung: Tiefe des Encoders im Transformer.
:param float mask_prob: Probability of masking each input vector in the transformer. # Explanation: Chance of masking inputs for training. // Erklärung: Wahrscheinlichkeit, Eingaben während des Trainings zu maskieren.
:param float positional_embeddings_dropout: Dropout rate for the positional embeddings. # Explanation: Regularization for positional encodings. // Erklärung: Regularisierung für Positionskodierungen.
:param int sequence_size: Length of the input sequence. # Explanation: Number of tokens in an input sequence. // Erklärung: Anzahl der Token in einer Eingabesequenz.
:param float dropout_encoder: Dropout rate for the encoder. # Explanation: Regularization for encoder layers. // Erklärung: Regularisierung für Encoder-Schichten.
:param float dropout_encoder_features: Dropout probability of the encoder output. # Explanation: Regularization for encoder's output. // Erklärung: Regularisierung für die Ausgabe des Encoders.
:param int lstm_hidden_size: LSTM hidden size. # Explanation: Number of features in LSTM hidden state. // Erklärung: Anzahl der Features im versteckten Zustand des LSTMs.
:param bool bidirectional_lstm: Forward or bidirectional LSTM. # Explanation: Whether the LSTM processes input in two directions. // Erklärung: Ob das LSTM Eingaben in zwei Richtungen verarbeitet.
:param List[float] variable_weights: List of weights for the loss function [9] if control_mode == "keyboard" or [2] if control_mode == "controller". # Explanation: Weights assigned to loss for different tasks. // Erklärung: Gewichte für den Verlust bei verschiedenen Aufgaben.
:param str encoder_type: Encoder type: transformer or lstm. # Explanation: Specifies the type of encoder. // Erklärung: Gibt den Typ des Encoders an.
:param float label_smoothing: Label smoothing for the classification task. # Explanation: Adds uncertainty to labels for regularization. // Erklärung: Fügt Unsicherheit zu Labels hinzu zur Regularisierung.
:param str checkpoint_path: Path to a checkpoint to load the model from (Useful if you want to load a model pretrained in the Image Reordering Task). # Explanation: Load previously trained model weights. // Erklärung: Vorher trainierte Modellgewichte laden.
:param bool find_lr: Whether to find the learning rate. We will use PytorchLightning's find_lr function. See: https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#learning-rate-finder. # Explanation: Automatically search for optimal learning rate. // Erklärung: Automatische Suche nach der optimalen Lernrate.
:param str optimizer_name: Optimizer to use: adamw or adafactor. # Explanation: Algorithm to optimize model parameters. // Erklärung: Algorithmus zur Optimierung der Modellparameter.
:param str scheduler_name: Scheduler to use: linear, polynomial, cosine, plateau. # Explanation: Adjust learning rate during training. // Erklärung: Lernrate während des Trainings anpassen.
:param float learning_rate: Learning rate. # Explanation: Speed of parameter updates during training. // Erklärung: Geschwindigkeit der Parameteranpassungen während des Trainings.
:param float weight_decay: Weight decay. # Explanation: Regularization to avoid overfitting. // Erklärung: Regularisierung, um Überanpassung zu vermeiden.
:param float warmup_factor: Percentage of the total training steps to perform warmup. # Explanation: Gradually increase learning rate at start of training. // Erklärung: Lernrate zu Beginn des Trainings schrittweise erhöhen.
"""

assert control_mode.lower() in [  # Ensure the control_mode is either "keyboard" or "controller".  # Überprüft, ob der Steuerungsmodus entweder "keyboard" oder "controller" ist.
    "keyboard",  # Valid control mode: "keyboard".  # Gültiger Steuerungsmodus: "keyboard".
    "controller",  # Valid control mode: "controller".  # Gültiger Steuerungsmodus: "controller".
], f"{control_mode.lower()} control mode not supported. Supported dataset types: [keyboard, controller]."  
# Throws an error if the control mode is invalid.  # Gibt einen Fehler aus, wenn der Steuerungsmodus ungültig ist.

if dropout_images_prob is None:  # Set default dropout probabilities if none are provided.  # Setzt Standardwerte für Dropout-Wahrscheinlichkeiten, wenn keine angegeben sind.
    dropout_images_prob = [0.0, 0.0, 0.0, 0.0, 0.0]  # Default values for dropout probabilities.  # Standardwerte für Dropout-Wahrscheinlichkeiten.

num_examples = count_examples(dataset_dir=train_dir)  # Count the number of training examples in the dataset.  # Zählt die Anzahl der Trainingsbeispiele im Datensatz.

num_update_steps_per_epoch = math.ceil(  # Calculate update steps per epoch by rounding up.  # Berechnet die Aktualisierungsschritte pro Epoche und rundet auf.
    math.ceil(math.ceil(num_examples / batch_size) / accumulation_steps / devices)  
)  

max_train_steps = max_epochs * num_update_steps_per_epoch  # Compute total training steps from epochs and updates per epoch.  # Berechnet die gesamten Trainingsschritte aus den Epochen und den Aktualisierungen pro Epoche.

num_warmup_steps = int(max_train_steps * warmup_factor)  # Compute warmup steps based on max training steps and warmup factor.  # Berechnet Warmup-Schritte basierend auf den maximalen Trainingsschritten und dem Warmup-Faktor.

print(  # Print training configuration details.  # Gibt Trainingskonfigurationsdetails aus.
    f"\n*** Training info ***\n"
    f"Number of training examples: {num_examples}\n"  # Print total number of training examples.  # Gibt die Gesamtanzahl der Trainingsbeispiele aus.
    f"Number of update steps per epoch: {num_update_steps_per_epoch}\n"  # Print steps per epoch.  # Gibt die Schritte pro Epoche aus.
    f"Max training steps: {max_train_steps}\n"  # Print max training steps.  # Gibt die maximalen Trainingsschritte aus.
    f"Number of warmup steps: {num_warmup_steps}\n"  # Print warmup steps.  # Gibt die Warmup-Schritte aus.
    f"Optimizer: {optimizer_name}\n"  # Name of the optimizer.  # Name des Optimierers.
    f"Scheduler: {scheduler_name}\n"  # Name of the scheduler.  # Name des Schedulers.
    f"Learning rate: {learning_rate}\n"  # Print learning rate.  # Gibt die Lernrate aus.
)

if not checkpoint_path:  # Check if checkpoint path is empty and initialize the model if so.  # Prüft, ob ein Checkpoint-Pfad vorhanden ist, und initialisiert das Modell, falls nicht.
    model: Tedd1104ModelPL = Tedd1104ModelPL(  # Create the model instance with given parameters.  # Erstellt die Modellinstanz mit den angegebenen Parametern.
        cnn_model_name=cnn_model_name,  # Name of the CNN model.  # Name des CNN-Modells.
        pretrained_cnn=pretrained_cnn,  # Use pretrained weights for the CNN.  # Verwendet vortrainierte Gewichte für das CNN.
        embedded_size=embedded_size,  # Size of the embedding vector.  # Größe des Einbettungsvektors.
        nhead=nhead,  # Number of attention heads in the transformer.  # Anzahl der Attention-Köpfe im Transformer.
        num_layers_encoder=num_layers_encoder,  # Number of encoder layers.  # Anzahl der Encoder-Schichten.
        lstm_hidden_size=lstm_hidden_size,  # Size of the hidden state in the LSTM.  # Größe des verborgenen Zustands im LSTM.
        dropout_cnn_out=dropout_cnn_out,  # Dropout rate for CNN output.  # Dropout-Rate für die CNN-Ausgabe.
        positional_embeddings_dropout=positional_embeddings_dropout,  # Dropout rate for positional embeddings.  # Dropout-Rate für die Positions-Einbettungen.
        dropout_encoder=dropout_encoder,  # Dropout rate for the encoder.  # Dropout-Rate für den Encoder.
        dropout_encoder_features=dropout_encoder_features,  # Dropout rate for encoder features.  # Dropout-Rate für Encoder-Features.
        control_mode=control_mode,  # Control mode for training.  # Steuerungsmodus für das Training.
        sequence_size=sequence_size,  # Size of the input sequence.  # Größe der Eingabesequenz.
        encoder_type=encoder_type,  # Type of encoder (e.g., transformer).  # Typ des Encoders (z. B. Transformer).
        bidirectional_lstm=bidirectional_lstm,  # Whether the LSTM is bidirectional.  # Gibt an, ob das LSTM bidirektional ist.
        weights=variable_weights,  # Weights used for model initialization.  # Gewichte zur Modellinitialisierung.
        label_smoothing=label_smoothing,  # Smoothing applied to labels.  # Glättung, die auf Labels angewendet wird.
        accelerator=accelerator,  # Accelerator used for computation (e.g., GPU).  # Beschleuniger für Berechnungen (z. B. GPU).
        learning_rate=learning_rate,  # Learning rate used for training.  # Lernrate für das Training.
        weight_decay=weight_decay,  # Weight decay for regularization.  # Gewichtszunahme zur Regularisierung.
        optimizer_name=optimizer_name,  # Name of the optimizer.  # Name des Optimierers.
        scheduler_name=scheduler_name,  # Name of the scheduler.  # Name des Schedulers.
        num_warmup_steps=num_warmup_steps,  # Number of warmup steps for the scheduler.  # Anzahl der Warmup-Schritte für den Scheduler.
        num_training_steps=max_train_steps,  # Total number of training steps.  # Gesamtzahl der Trainingsschritte.
    )


else:  # Block executed if the `if` condition above is not satisfied.  # Wird ausgeführt, wenn die vorherige `if`-Bedingung nicht erfüllt ist.

    print(f"Restoring model from {checkpoint_path}.")  # Print the checkpoint path being restored.  # Gibt den Pfad des wiederherzustellenden Modells aus.
    model = Tedd1104ModelPL.load_from_checkpoint(  # Load the model from a checkpoint file.  # Lädt das Modell aus einer Checkpoint-Datei.
        checkpoint_path=checkpoint_path,  # Path to the checkpoint file.  # Pfad zur Checkpoint-Datei.
        dropout_cnn_out=dropout_cnn_out,  # Dropout rate for CNN output.  # Dropout-Rate für die CNN-Ausgabe.
        positional_embeddings_dropout=positional_embeddings_dropout,  # Dropout for positional embeddings.  # Dropout für Positions-Einbettungen.
        dropout_encoder=dropout_encoder,  # Dropout rate in the encoder.  # Dropout-Rate im Encoder.
        dropout_encoder_features=dropout_encoder_features,  # Dropout rate for encoder features.  # Dropout-Rate für Encoder-Features.
        mask_prob=mask_prob,  # Probability of masking certain data.  # Wahrscheinlichkeit, bestimmte Daten zu maskieren.
        control_mode=control_mode,  # Mode controlling certain behaviors.  # Steuerungsmodus für bestimmte Verhaltensweisen.
        lstm_hidden_size=lstm_hidden_size,  # Size of the LSTM hidden layer.  # Größe der versteckten Schicht des LSTM.
        bidirectional_lstm=bidirectional_lstm,  # Use bidirectional LSTM or not.  # Aktiviert oder deaktiviert bidirektionales LSTM.
        strict=False,  # Allow relaxed checkpoint loading.  # Erlaubt flexibles Laden des Checkpoints.
        learning_rate=learning_rate,  # Learning rate for training.  # Lernrate für das Training.
        weight_decay=weight_decay,  # Weight decay for regularization.  # Gewichtsabnahme zur Regularisierung.
        optimizer_name=optimizer_name,  # Name of the optimizer.  # Name des Optimierers.
        scheduler_name=scheduler_name,  # Name of the learning rate scheduler.  # Name des Lernraten-Schedulers.
        num_warmup_steps=num_warmup_steps,  # Number of warmup steps.  # Anzahl der Aufwärmschritte.
        num_training_steps=max_train_steps,  # Total number of training steps.  # Gesamtzahl der Trainingsschritte.
    )

if not os.path.exists(output_dir):  # Check if output directory does not exist.  # Prüft, ob das Ausgabeverzeichnis nicht existiert.
    print(f"{output_dir} does not exits. We will create it.")  # Inform that the directory will be created.  # Gibt an, dass das Verzeichnis erstellt wird.
    os.makedirs(output_dir)  # Create the output directory.  # Erstellt das Ausgabeverzeichnis.

data = Tedd1104DataModule(  # Initialize a data module for managing data.  # Initialisiert ein Datenmodul zur Datenverwaltung.
    train_dir=train_dir,  # Directory containing training data.  # Verzeichnis mit Trainingsdaten.
    val_dir=val_dir,  # Directory containing validation data.  # Verzeichnis mit Validierungsdaten.
    test_dir=test_dir,  # Directory containing test data.  # Verzeichnis mit Testdaten.
    batch_size=batch_size,  # Batch size for data loaders.  # Batch-Größe für Datenlader.
    hide_map_prob=hide_map_prob,  # Probability of hiding map features.  # Wahrscheinlichkeit, Kartenmerkmale auszublenden.
    dropout_images_prob=dropout_images_prob,  # Dropout rate for images.  # Dropout-Rate für Bilder.
    control_mode=control_mode,  # Mode controlling data behavior.  # Modus zur Steuerung des Datenverhaltens.
    num_workers=dataloader_num_workers,  # Number of workers for data loading.  # Anzahl der Arbeitsprozesse für das Laden von Daten.
    token_mask_prob=mask_prob,  # Probability of masking tokens.  # Wahrscheinlichkeit, Token zu maskieren.
    transformer_nheads=None if model.encoder_type == "lstm" else model.nhead,  # Set the number of heads for transformers, or None for LSTM.  # Legt die Anzahl der Köpfe für Transformer fest oder None für LSTM.
    sequence_length=model.sequence_size,  # Length of the data sequences.  # Länge der Daten-Sequenzen.
)

experiment_name = os.path.basename(  # Get the base name of the output directory.  # Ruft den Basisnamen des Ausgabeverzeichnisses ab.
    output_dir if output_dir[-1] != "/" else output_dir[:-1]  # Remove trailing slash if present.  # Entfernt einen abschließenden Schrägstrich, falls vorhanden.
)


if report_to == "tensorboard":  # Check if reporting is set to 'tensorboard'.  # Überprüfen, ob das Reporting auf 'tensorboard' gesetzt ist.
        logger = pl_loggers.TensorBoardLogger(  # Initialize the TensorBoard logger.  # Initialisiert den TensorBoard-Logger.
            save_dir=output_dir,  # Directory where logs are saved.  # Verzeichnis, in dem Logs gespeichert werden.
            name=experiment_name,  # Name of the experiment.  # Name des Experiments.
        )
    elif report_to == "wandb":  # Check if reporting is set to 'wandb'.  # Überprüfen, ob das Reporting auf 'wandb' gesetzt ist.
        logger = pl_loggers.WandbLogger(  # Initialize the Wandb logger.  # Initialisiert den Wandb-Logger.
            name=experiment_name,  # Name of the experiment.  # Name des Experiments.
            # id=experiment_name,  # Optional: Experiment ID.  # Optional: Experiment-ID.
            # resume=None,  # Optional: Resume from a previous run.  # Optional: Fortsetzung eines vorherigen Laufs.
            project="TEDD1104",  # Name of the Wandb project.  # Name des Wandb-Projekts.
            save_dir=output_dir,  # Directory where logs are saved.  # Verzeichnis, in dem Logs gespeichert werden.
        )
    else:  # If neither 'tensorboard' nor 'wandb' is specified.  # Wenn weder 'tensorboard' noch 'wandb' angegeben ist.
        raise ValueError(  # Raise an error for unknown logger.  # Wirft einen Fehler für unbekannten Logger.
            f"Unknown logger: {report_to}. Please use 'tensorboard' or 'wandb'."  # Error message.  # Fehlermeldung.
        )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")  # Monitor and log learning rates at each step.  # Überwacht und protokolliert die Lernraten bei jedem Schritt.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(  # Set up a model checkpoint callback.  # Richtet einen Checkpoint-Callback für das Modell ein.
        dirpath=output_dir,  # Directory to save checkpoints.  # Verzeichnis zum Speichern von Checkpoints.
        monitor="Validation/acc_k@1_macro",  # Metric to monitor for saving.  # Metrik, die überwacht wird.
        mode="max",  # Save based on the maximum value of the monitored metric.  # Speichern basierend auf dem Maximalwert der überwachten Metrik.
        save_last=True,  # Save the last checkpoint.  # Speichert den letzten Checkpoint.
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"  # Define the naming format for the last checkpoint.  # Definiert das Namensformat für den letzten Checkpoint.

    model.accelerator = accelerator  # Assign the accelerator to the model.  # Weist dem Modell den Accelerator zu.

    trainer = pl.Trainer(  # Create a PyTorch Lightning Trainer instance.  # Erstellt eine Instanz des PyTorch Lightning Trainers.
        devices=devices,  # Specify the devices to use for training.  # Gibt die zu verwendenden Geräte für das Training an.
        accelerator=accelerator,  # Specify the type of accelerator (e.g., GPU, CPU).  # Gibt den Typ des Accelerators an (z. B. GPU, CPU).
        precision=precision if precision == "bf16" else int(precision),  # Use bf16 precision or convert precision to an integer.  # Verwendet bf16-Präzision oder konvertiert Präzision in eine Ganzzahl.
        strategy=strategy,  # Specify the training strategy (e.g., 'ddp').  # Gibt die Trainingsstrategie an (z. B. 'ddp').
        val_check_interval=val_check_interval,  # Interval for validation checks.  # Intervall für Validierungsprüfungen.
        accumulate_grad_batches=accumulation_steps,  # Accumulate gradients over several batches.  # Akkumuliert Gradienten über mehrere Batches.
        max_epochs=max_epochs,  # Set the maximum number of training epochs.  # Legt die maximale Anzahl von Trainingsepochen fest.
        logger=logger,  # Assign the logger to the trainer.  # Weist dem Trainer den Logger zu.
        callbacks=[  # List of callbacks to use during training.  # Liste der Callbacks, die während des Trainings verwendet werden.
            # pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),  # Optional: Apply stochastic weight averaging.  # Optional: Stochastisches Gewichtsmittel anwenden.
            checkpoint_callback,  # Add the checkpoint callback.  # Fügt den Checkpoint-Callback hinzu.
            lr_monitor,  # Add the learning rate monitor callback.  # Fügt den Lernraten-Monitoring-Callback hinzu.
        ],
        gradient_clip_val=1.0 if optimizer_name.lower() != "adafactor" else 0.0,  # Clip gradients unless optimizer is Adafactor.  # Begrenzen der Gradienten, außer der Optimierer ist Adafactor.
        log_every_n_steps=100,  # Log metrics every 100 steps.  # Protokolliert Metriken alle 100 Schritte.
        auto_lr_find=find_lr,  # Automatically find the best learning rate.  # Findet automatisch die beste Lernrate.
    )

if find_lr:  # Check if the learning rate finder should be executed.  # Überprüft, ob der Lernratenfinder ausgeführt werden soll.
    print(f"We will try to find the optimal learning rate.")  # Inform the user that the process of finding the learning rate is starting.  # Informiert den Benutzer, dass der Prozess zur Bestimmung der optimalen Lernrate startet.
    lr_finder = trainer.tuner.lr_find(model, datamodule=data)  # Use the trainer's tuner to find an optimal learning rate using the data module.  # Verwendet den Tuner des Trainers, um eine optimale Lernrate mit dem Datenmodul zu finden.
    print(lr_finder.results)  # Print the results of the learning rate finder.  # Gibt die Ergebnisse des Lernratenfinders aus.
    fig = lr_finder.plot(suggest=True)  # Plot the learning rate curve and suggest the best one.  # Plottet die Lernratenkurve und schlägt die beste vor.
    fig.savefig(os.path.join(output_dir, "lr_finder.png"))  # Save the learning rate plot as an image in the output directory.  # Speichert das Diagramm der Lernrate als Bild im Ausgabeverzeichnis.
    new_lr = lr_finder.suggestion()  # Get the suggested optimal learning rate from the finder.  # Holt die empfohlene optimale Lernrate vom Finder.
    print(f"We will train with the suggested learning rate: {new_lr}")  # Inform the user about the chosen learning rate.  # Informiert den Benutzer über die ausgewählte Lernrate.
    model.hparams.learning_rate = new_lr  # Set the model's learning rate to the suggested value.  # Setzt die Lernrate des Modells auf den vorgeschlagenen Wert.

trainer.fit(model, datamodule=data)  # Train the model using the specified data module.  # Trainiert das Modell mit dem angegebenen Datenmodul.

print(f"Best model path: {checkpoint_callback.best_model_path}")  # Print the path to the best model saved during training.  # Gibt den Pfad zum besten während des Trainings gespeicherten Modell aus.
if test_dir:  # Check if a testing directory is provided.  # Überprüft, ob ein Testverzeichnis angegeben ist.
    trainer.test(datamodule=data, ckpt_path="best")  # Test the model using the best checkpoint and the data module.  # Testet das Modell mit dem besten Checkpoint und dem Datenmodul.

def continue_training(  # Define a function for resuming training with given parameters.  # Definiert eine Funktion zum Fortsetzen des Trainings mit gegebenen Parametern.
    checkpoint_path: str,  # Path to the checkpoint from which to resume training.  # Pfad zum Checkpoint, von dem das Training fortgesetzt wird.
    train_dir: str,  # Path to the training data directory.  # Pfad zum Verzeichnis der Trainingsdaten.
    val_dir: str,  # Path to the validation data directory.  # Pfad zum Verzeichnis der Validierungsdaten.
    batch_size: int,  # Number of samples per batch for training.  # Anzahl der Proben pro Batch für das Training.
    max_epochs: int,  # Maximum number of epochs to train.  # Maximale Anzahl an Epochen für das Training.
    output_dir,  # Directory where outputs like logs and checkpoints will be saved.  # Verzeichnis, in dem Ausgaben wie Protokolle und Checkpoints gespeichert werden.
    accumulation_steps,  # Number of steps to accumulate gradients before updating weights.  # Anzahl der Schritte zur Akkumulation von Gradienten vor der Gewichtsaktualisierung.
    devices: int = 1,  # Number of devices (e.g., GPUs) to use.  # Anzahl der Geräte (z. B. GPUs), die verwendet werden sollen.
    accelerator: str = "auto",  # Accelerator to use, e.g., GPU or CPU, determined automatically.  # Beschleuniger zur Verwendung, z. B. GPU oder CPU, automatisch ermittelt.
    precision: str = "16",  # Precision for computations, e.g., 16-bit floating point.  # Präzision für Berechnungen, z. B. 16-Bit-Gleitkomma.
    strategy=None,  # Strategy for distributed training, if applicable.  # Strategie für verteiltes Training, falls anwendbar.
    test_dir: str = None,  # Directory for testing data, if any.  # Verzeichnis für Testdaten, falls vorhanden.
    mask_prob: float = 0.0,  # Probability of masking elements in the training data.  # Wahrscheinlichkeit, Elemente in den Trainingsdaten zu maskieren.
    hide_map_prob: float = 0.0,  # Probability of hiding map data during training.  # Wahrscheinlichkeit, Kartendaten während des Trainings zu verbergen.
    dropout_images_prob=None,  # Probability of dropping images during training, if applicable.  # Wahrscheinlichkeit, Bilder während des Trainings wegzulassen, falls anwendbar.
    dataloader_num_workers=os.cpu_count(),  # Number of workers for data loading, defaults to CPU count.  # Anzahl der Worker für das Laden von Daten, standardmäßig die Anzahl der CPU-Kerne.
    val_check_interval: float = 0.25,  # Interval at which validation is performed during training.  # Intervall, in dem die Validierung während des Trainings durchgeführt wird.
    report_to: str = "wandb",  # Platform to report logs and metrics to, e.g., Weights & Biases.  # Plattform, auf der Protokolle und Metriken gemeldet werden, z. B. Weights & Biases.
):

"""
Continues training a model from a checkpoint.

:param str checkpoint_path: Path to the checkpoint to continue training from.  # Path zur Checkpoint-Datei, von der aus das Training fortgesetzt wird.
:param str train_dir: The directory containing the training data.  # Verzeichnis, das die Trainingsdaten enthält.
:param str val_dir: The directory containing the validation data.  # Verzeichnis, das die Validierungsdaten enthält.
:param str output_dir: The directory to save the model to.  # Verzeichnis, in dem das Modell gespeichert wird.
:param int batch_size: The batch size.  # Größe der Datenblöcke (Batches).
:param int accumulation_steps: The number of steps to accumulate gradients.  # Anzahl der Schritte, um Gradienten zu akkumulieren.
:param int devices: Number of devices to use.  # Anzahl der zu verwendenden Geräte.
:param str accelerator: Accelerator to use. If 'auto', tries to automatically detect TPU, GPU, CPU or IPU system.  # Beschleuniger, der verwendet werden soll. Bei 'auto' wird versucht, TPU, GPU, CPU oder IPU automatisch zu erkennen.
:param str precision: Precision to use. Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16). Can be used on CPU, GPU or TPUs.  # Verwendete Präzision: doppelte Präzision (64), volle Präzision (32), halbe Präzision (16) oder bfloat16-Präzision (bf16). Kann auf CPU, GPU oder TPU verwendet werden.
:param str strategy: Strategy to use for data parallelism. "None" for no data parallelism, ddp_find_unused_parameters_false for DDP.  # Strategie für Datenparallelität. "None" für keine Parallelität, "ddp_find_unused_parameters_false" für DDP.
:param str report_to: Where to report the results. "tensorboard" for TensorBoard, "wandb" for W&B.  # Wohin die Ergebnisse gemeldet werden sollen. "tensorboard" für TensorBoard, "wandb" für W&B.
:param int max_epochs: The maximum number of epochs to train for.  # Maximale Anzahl von Epochen, die trainiert werden.
:param bool hide_map_prob: Probability of hiding the minimap (0<=hide_map_prob<=1).  # Wahrscheinlichkeit, die Minimap zu verstecken (0<=hide_map_prob<=1).
:param float mask_prob: Probability of masking each input vector in the transformer.  # Wahrscheinlichkeit, jeden Eingabevektor im Transformer zu maskieren.
:param float dropout_images_prob: Probability of dropping an image (0<=dropout_images_prob<=1).  # Wahrscheinlichkeit, ein Bild fallen zu lassen (0<=dropout_images_prob<=1).
:param str test_dir: The directory containing the test data.  # Verzeichnis, das die Testdaten enthält.
:param int dataloader_num_workers: The number of workers to use for the dataloaders.  # Anzahl der Arbeiter, die für die Datenlader verwendet werden.
:param float val_check_interval: The interval in epochs to check the validation accuracy.  # Intervall (in Epochen), um die Validierungsgenauigkeit zu prüfen.
"""


if dropout_images_prob is None:  # Check if `dropout_images_prob` is not provided.  # Überprüfen, ob `dropout_images_prob` nicht angegeben ist.
    dropout_images_prob = [0.0, 0.0, 0.0, 0.0, 0.0]  # Set default dropout probabilities for images.  # Standardwerte für Dropout-Wahrscheinlichkeiten für Bilder setzen.

print(f"Restoring checkpoint: {checkpoint_path}")  # Print a message indicating the restoration of a checkpoint.  # Nachricht ausgeben, dass ein Checkpoint wiederhergestellt wird.

model = Tedd1104ModelPL.load_from_checkpoint(checkpoint_path=checkpoint_path)  # Load the model from the specified checkpoint.  # Modell aus dem angegebenen Checkpoint laden.

print("Done! Preparing to continue training...")  # Print a message indicating readiness to continue training.  # Nachricht ausgeben, dass das Training fortgesetzt wird.

data = Tedd1104DataModule(  # Initialize the data module for training, validation, and testing.  # Datenmodul für Training, Validierung und Tests initialisieren.
    train_dir=train_dir,  # Path to the training data directory.  # Pfad zum Verzeichnis der Trainingsdaten.
    val_dir=val_dir,  # Path to the validation data directory.  # Pfad zum Verzeichnis der Validierungsdaten.
    test_dir=test_dir,  # Path to the testing data directory.  # Pfad zum Verzeichnis der Testdaten.
    batch_size=batch_size,  # Number of samples per batch.  # Anzahl der Samples pro Batch.
    hide_map_prob=hide_map_prob,  # Probability of hiding the map during training.  # Wahrscheinlichkeit, die Karte während des Trainings auszublenden.
    dropout_images_prob=dropout_images_prob,  # Dropout probabilities for images.  # Dropout-Wahrscheinlichkeiten für Bilder.
    control_mode=model.control_mode,  # Control mode determined by the model.  # Kontrollmodus, der durch das Modell festgelegt wird.
    num_workers=dataloader_num_workers,  # Number of workers for data loading.  # Anzahl der Worker für das Laden der Daten.
    token_mask_prob=mask_prob,  # Probability of masking tokens during training.  # Wahrscheinlichkeit, Token während des Trainings zu maskieren.
    transformer_nheads=None if model.encoder_type == "lstm" else model.nhead,  # Number of attention heads, if the model is not LSTM.  # Anzahl der Attention-Köpfe, falls das Modell kein LSTM ist.
    sequence_length=model.sequence_size,  # Sequence length for training.  # Sequenzlänge für das Training.
)

experiment_name = os.path.basename(  # Extract the experiment name from the output directory path.  # Experimentname aus dem Pfad des Ausgabe-Verzeichnisses extrahieren.
    output_dir if output_dir[-1] != "/" else output_dir[:-1]  # Handle trailing slash in the directory path.  # Umgang mit einem abschließenden Schrägstrich im Verzeichnispfad.
)

if report_to == "tensorboard":  # Check if logging should use TensorBoard.  # Überprüfen, ob Logging mit TensorBoard erfolgen soll.
    logger = pl_loggers.TensorBoardLogger(  # Initialize a TensorBoard logger.  # Einen TensorBoard-Logger initialisieren.
        save_dir=output_dir,  # Directory to save logs.  # Verzeichnis zum Speichern der Logs.
        name=experiment_name,  # Name of the experiment.  # Name des Experiments.
    )
elif report_to == "wandb":  # Check if logging should use Weights & Biases (wandb).  # Überprüfen, ob Logging mit Weights & Biases (wandb) erfolgen soll.
    logger = pl_loggers.WandbLogger(  # Initialize a wandb logger.  # Einen wandb-Logger initialisieren.
        name=experiment_name,  # Name of the experiment.  # Name des Experiments.
        # id=experiment_name,  # Optional experiment ID (commented).  # Optionale Experiment-ID (auskommentiert).
        # resume="allow",  # Optional resume setting (commented).  # Optionale Fortsetzungs-Einstellung (auskommentiert).
        project="TEDD1104",  # Name of the wandb project.  # Name des wandb-Projekts.
        save_dir=output_dir,  # Directory to save wandb logs.  # Verzeichnis zum Speichern der wandb-Logs.
    )
else:  # Raise an error for unsupported logger types.  # Fehler auslösen für nicht unterstützte Logger-Typen.
    raise ValueError(  # Raise an exception with an error message.  # Ausnahme mit einer Fehlermeldung auslösen.
        f"Unknown logger: {report_to}. Please use 'tensorboard' or 'wandb'."  # Error message for unsupported logger.  # Fehlermeldung für nicht unterstützten Logger.
    )

lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")  # Create a learning rate monitor to log at each step.  # Lernraten-Monitor erstellen, der bei jedem Schritt loggt.

checkpoint_callback = pl.callbacks.ModelCheckpoint(  # Create a checkpoint callback to save model states.  # Checkpoint-Callback erstellen, um Modellzustände zu speichern.
    dirpath=output_dir,  # Directory to save checkpoints.  # Verzeichnis zum Speichern der Checkpoints.
    monitor="Validation/acc_k@1_macro",  # Metric to monitor for saving checkpoints.  # Zu überwachende Metrik für das Speichern der Checkpoints.
    mode="max",  # Save checkpoints with the maximum metric value.  # Checkpoints mit maximalem Metrikwert speichern.
    save_last=True,  # Always save the last checkpoint.  # Immer den letzten Checkpoint speichern.
)
checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"  # Define a custom name for the last checkpoint.  # Benutzerdefinierten Namen für den letzten Checkpoint festlegen.

trainer = pl.Trainer(  # Initialize the PyTorch Lightning Trainer.  # PyTorch Lightning Trainer initialisieren.
    devices=devices,  # Devices to use (e.g., CPU or GPU).  # Zu verwendende Geräte (z. B. CPU oder GPU).
    accelerator=accelerator,  # Type of accelerator (e.g., 'gpu' or 'cpu').  # Art des Beschleunigers (z. B. 'gpu' oder 'cpu').
    precision=precision if precision == "bf16" else int(precision),  # Numerical precision (e.g., 16-bit or 32-bit).  # Numerische Genauigkeit (z. B. 16-Bit oder 32-Bit).
    strategy=strategy,  # Training strategy (e.g., distributed training).  # Trainingsstrategie (z. B. verteiltes Training).
    val_check_interval=val_check_interval,  # Frequency of validation checks.  # Häufigkeit der Validierungsüberprüfungen.
    accumulate_grad_batches=accumulation_steps,  # Accumulate gradients over multiple batches.  # Gradienten über mehrere Batches akkumulieren.
    max_epochs=max_epochs,  # Maximum number of training epochs.  # Maximale Anzahl der Trainingsepochen.
    logger=logger,  # Logger instance for experiment tracking.  # Logger-Instanz zur Nachverfolgung des Experiments.
    callbacks=[  # List of callbacks to use during training.  # Liste der während des Trainings zu verwendenden Callbacks.
        pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),  # Apply Stochastic Weight Averaging (SWA).  # Stochastic Weight Averaging (SWA) anwenden.
        checkpoint_callback,  # Add the checkpoint callback.  # Checkpoint-Callback hinzufügen.
        lr_monitor,  # Add the learning rate monitor.  # Lernraten-Monitor hinzufügen.
    ],
    gradient_clip_val=1.0,  # Clip gradients to avoid exploding gradients.  # Gradienten begrenzen, um explodierende Gradienten zu vermeiden.
    log_every_n_steps=100,  # Log metrics every 100 steps.  # Metriken alle 100 Schritte loggen.
)

trainer.fit(  # Start the training process.  # Den Trainingsprozess starten.
    ckpt_path=checkpoint_path,  # Path to the checkpoint for resuming training.  # Pfad zum Checkpoint, um das Training fortzusetzen.
    model=model,  # Model instance to train.  # Zu trainierendes Modell.
    datamodule=data,  # Data module instance for loading data.  # Datenmodul-Instanz zum Laden der Daten.
)

if test_dir:  # Check if a test directory is specified.  # Überprüfen, ob ein Testverzeichnis angegeben ist.
    trainer.test(datamodule=data, ckpt_path="best")  # Test the model using the best checkpoint.  # Das Modell mit dem besten Checkpoint testen.



if __name__ == "__main__":  # Ensures this script runs only when executed directly.  # Stellt sicher, dass dieses Skript nur ausgeführt wird, wenn es direkt ausgeführt wird.

    parser = argparse.ArgumentParser(  # Initializes an argument parser for command-line inputs.  # Initialisiert einen Argumentparser für Kommandozeileneingaben.
        description="Train a T.E.D.D. 1104 model in the supervised self-driving task."  # Provides a description for the parser.  # Gibt eine Beschreibung für den Parser an.
    )

    group = parser.add_mutually_exclusive_group(required=True)  # Creates a group of mutually exclusive arguments, one of which must be provided.  # Erstellt eine Gruppe von sich gegenseitig ausschließenden Argumenten, von denen eines angegeben werden muss.

    group.add_argument(  # Adds the first argument to the mutually exclusive group.  # Fügt der Gruppe das erste Argument hinzu.
        "--train_new",  # Flag for training a new model.  # Flag zum Trainieren eines neuen Modells.
        action="store_true",  # Stores True if this flag is present.  # Speichert True, wenn dieses Flag gesetzt ist.
        help="Train a new model",  # Describes the purpose of this flag.  # Beschreibt die Funktion dieses Flags.
    )

    group.add_argument(  # Adds the second argument to the mutually exclusive group.  # Fügt der Gruppe das zweite Argument hinzu.
        "--continue_training",  # Flag for continuing training from a checkpoint.  # Flag zum Fortsetzen des Trainings von einem Checkpoint.
        action="store_true",  # Stores True if this flag is present.  # Speichert True, wenn dieses Flag gesetzt ist.
        help="Continues training a model from a checkpoint.",  # Describes the purpose of this flag.  # Beschreibt die Funktion dieses Flags.
    )

    parser.add_argument(  # Adds a required argument for the training data directory.  # Fügt ein erforderliches Argument für das Trainingsdatenverzeichnis hinzu.
        "--train_dir",  # Specifies the training data directory.  # Gibt das Trainingsdatenverzeichnis an.
        type=str,  # Expects a string input.  # Erwartet eine Eingabe vom Typ String.
        required=True,  # This argument is mandatory.  # Dieses Argument ist verpflichtend.
        help="The directory containing the training data.",  # Describes the purpose of this argument.  # Beschreibt die Funktion dieses Arguments.
    )

    parser.add_argument(  # Adds a required argument for the validation data directory.  # Fügt ein erforderliches Argument für das Validierungsdatenverzeichnis hinzu.
        "--val_dir",  # Specifies the validation data directory.  # Gibt das Validierungsdatenverzeichnis an.
        type=str,  # Expects a string input.  # Erwartet eine Eingabe vom Typ String.
        required=True,  # This argument is mandatory.  # Dieses Argument ist verpflichtend.
        help="The directory containing the validation data.",  # Describes the purpose of this argument.  # Beschreibt die Funktion dieses Arguments.
    )

    parser.add_argument(  # Adds an optional argument for the test data directory.  # Fügt ein optionales Argument für das Testdatenverzeichnis hinzu.
        "--test_dir",  # Specifies the test data directory.  # Gibt das Testdatenverzeichnis an.
        type=str,  # Expects a string input.  # Erwartet eine Eingabe vom Typ String.
        default=None,  # Default value is None if not provided.  # Standardwert ist None, wenn nicht angegeben.
        help="The directory containing the test data.",  # Describes the purpose of this argument.  # Beschreibt die Funktion dieses Arguments.
    )

    parser.add_argument(  # Adds a required argument for the output directory.  # Fügt ein erforderliches Argument für das Ausgabe-Verzeichnis hinzu.
        "--output_dir",  # Specifies where to save the model.  # Gibt an, wo das Modell gespeichert werden soll.
        type=str,  # Expects a string input.  # Erwartet eine Eingabe vom Typ String.
        required=True,  # This argument is mandatory.  # Dieses Argument ist verpflichtend.
        help="The directory to save the model to.",  # Describes the purpose of this argument.  # Beschreibt die Funktion dieses Arguments.
    )

    parser.add_argument(  # Adds an argument to choose the encoder type.  # Fügt ein Argument zur Auswahl des Encoder-Typs hinzu.
        "--encoder_type",  # Specifies the encoder type to use.  # Gibt den zu verwendenden Encoder-Typ an.
        type=str,  # Expects a string input.  # Erwartet eine Eingabe vom Typ String.
        choices=["lstm", "transformer"],  # Limits options to "lstm" or "transformer".  # Beschränkt die Optionen auf "lstm" oder "transformer".
        default="transformer",  # Default encoder is "transformer".  # Der Standardencoder ist "transformer".
        help="The Encoder type to use: transformer or lstm",  # Describes the purpose of this argument.  # Beschreibt die Funktion dieses Arguments.
    )

    parser.add_argument(  # Adds a required argument for the batch size.  # Fügt ein erforderliches Argument für die Batchgröße hinzu.
        "--batch_size",  # Specifies the batch size.  # Gibt die Batchgröße an.
        type=int,  # Expects an integer input.  # Erwartet eine Eingabe vom Typ Integer.
        required=True,  # This argument is mandatory.  # Dieses Argument ist verpflichtend.
        help="The batch size for training and eval.",  # Describes the purpose of this argument.  # Beschreibt die Funktion dieses Arguments.
    )

    parser.add_argument(  # Adds an optional argument for gradient accumulation steps.  # Fügt ein optionales Argument für die Anzahl der Gradientakkumulationsschritte hinzu.
        "--accumulation_steps",  # Specifies the number of steps to accumulate gradients.  # Gibt die Anzahl der Schritte zur Akkumulation von Gradienten an.
        type=int,  # Expects an integer input.  # Erwartet eine Eingabe vom Typ Integer.
        default=1,  # Default value is 1.  # Der Standardwert ist 1.
        help="The number of steps to accumulate gradients.",  # Describes the purpose of this argument.  # Beschreibt die Funktion dieses Arguments.
    )

    parser.add_argument(  # Adds a required argument for the maximum number of epochs.  # Fügt ein erforderliches Argument für die maximale Anzahl von Epochen hinzu.
        "--max_epochs",  # Specifies the maximum number of epochs.  # Gibt die maximale Anzahl von Epochen an.
        type=int,  # Expects an integer input.  # Erwartet eine Eingabe vom Typ Integer.
        required=True,  # This argument is mandatory.  # Dieses Argument ist verpflichtend.
        help="The maximum number of epochs to train for.",  # Describes the purpose of this argument.  # Beschreibt die Funktion dieses Arguments.
    )


parser.add_argument(
    "--dataloader_num_workers",  # Number of CPU workers for the Data Loaders.  # Anzahl der CPU-Worker für die Data Loader.
    type=int,  # Specifies the type as integer.  # Gibt den Typ als Ganzzahl an.
    default=os.cpu_count(),  # Uses the number of available CPUs as the default.  # Verwendet die Anzahl verfügbarer CPUs als Standard.
    help="Number of CPU workers for the Data Loaders",  # Description of the parameter.  # Beschreibung des Parameters.
)

parser.add_argument(
    "--hide_map_prob",  # Probability of hiding the minimap in the sequence (0<=hide_map_prob<=1).  # Wahrscheinlichkeit, die Minikarte in der Sequenz auszublenden (0<=hide_map_prob<=1).
    type=float,  # Specifies the type as float.  # Gibt den Typ als Gleitkommazahl an.
    default=0.0,  # Default probability is set to 0.  # Standardwahrscheinlichkeit ist auf 0 gesetzt.
    help="Probability of hiding the minimap in the sequence (0<=hide_map_prob<=1)",  # Description of the parameter.  # Beschreibung des Parameters.
)

parser.add_argument(
    "--dropout_images_prob",  # Probability of dropping each image in the sequence (0<=dropout_images_prob<=1).  # Wahrscheinlichkeit, jedes Bild in der Sequenz zu löschen (0<=dropout_images_prob<=1).
    type=float,  # Specifies the type as float.  # Gibt den Typ als Gleitkommazahl an.
    nargs=5,  # Expects 5 float values.  # Erwartet 5 Gleitkommazahlen.
    default=[0.0, 0.0, 0.0, 0.0, 0.0],  # Default probabilities are all set to 0.  # Standardwahrscheinlichkeiten sind alle auf 0 gesetzt.
    help="Probability of dropping each image in the sequence (0<=dropout_images_prob<=1)",  # Description of the parameter.  # Beschreibung des Parameters.
)

parser.add_argument(
    "--variable_weights",  # List of weights for the loss function depending on control_mode.  # Liste der Gewichte für die Verlustfunktion je nach control_mode.
    type=float,  # Specifies the type as float.  # Gibt den Typ als Gleitkommazahl an.
    nargs="+",  # Accepts one or more values.  # Akzeptiert einen oder mehrere Werte.
    default=None,  # Default is None.  # Standard ist None.
    help="List of weights for the loss function [9] if control_mode == 'keyboard' or [2] if control_mode == 'controller'",  # Description of the parameter.  # Beschreibung des Parameters.
)

parser.add_argument(
    "--val_check_interval",  # The interval in epochs between validation checks.  # Das Intervall in Epochen zwischen Validierungsprüfungen.
    type=float,  # Specifies the type as float.  # Gibt den Typ als Gleitkommazahl an.
    default=1.0,  # Default interval is 1 epoch.  # Standardintervall ist 1 Epoche.
    help="The interval in epochs between validation checks.",  # Description of the parameter.  # Beschreibung des Parameters.
)

parser.add_argument(
    "--learning_rate",  # The learning rate for the optimizer.  # Die Lernrate für den Optimierer.
    type=float,  # Specifies the type as float.  # Gibt den Typ als Gleitkommazahl an.
    default=3e-5,  # Default learning rate is 0.00003.  # Standard-Lernrate ist 0.00003.
    help="[NEW MODEL] The learning rate for the optimizer.",  # Description of the parameter.  # Beschreibung des Parameters.
)

parser.add_argument(
    "--weight_decay",  # AdamW weight decay.  # AdamW-Gewichtsabnahme.
    type=float,  # Specifies the type as float.  # Gibt den Typ als Gleitkommazahl an.
    default=1e-4,  # Default weight decay is 0.0001.  # Standard-Gewichtsabnahme ist 0.0001.
    help="[NEW MODEL]] AdamW Weight Decay",  # Description of the parameter.  # Beschreibung des Parameters.
)

parser.add_argument(
    "--optimizer_name",  # The optimizer to use: adamw or adafactor.  # Der zu verwendende Optimierer: adamw oder adafactor.
    type=str,  # Specifies the type as string.  # Gibt den Typ als Zeichenkette an.
    default="adamw",  # Default optimizer is adamw.  # Standardoptimierer ist adamw.
    choices=["adamw", "adafactor"],  # Possible choices are adamw and adafactor.  # Mögliche Optionen sind adamw und adafactor.
    help="[NEW MODEL] The optimizer to use: adamw or adafactor. Adafactor requires fairseq to be installed. pip install fairseq",  # Description and prerequisites.  # Beschreibung und Voraussetzungen.
)

parser.add_argument(
    "--scheduler_name",  # The scheduler to use: linear, polynomial, cosine, plateau.  # Der zu verwendende Scheduler: linear, polynomial, cosine, plateau.
    type=str,  # Specifies the type as string.  # Gibt den Typ als Zeichenkette an.
    default="linear",  # Default scheduler is linear.  # Standardscheduler ist linear.
    choices=["linear", "plateau", "polynomial", "cosine"],  # Possible choices.  # Mögliche Optionen.
    help="[NEW MODEL] The scheduler to use: linear, polynomial, cosine, plateau.",  # Description of the parameter.  # Beschreibung des Parameters.
)

parser.add_argument(
    "--warmup_factor",  # Percentage of training steps for warmup (0<=warmup_factor<=1).  # Prozentsatz der Trainingsschritte für das Aufwärmen (0<=warmup_factor<=1).
    type=float,  # Specifies the type as float.  # Gibt den Typ als Gleitkommazahl an.
    default=0.05,  # Default warmup factor is 0.05.  # Standard-Aufwärmfaktor ist 0.05.
    help="[NEW MODEL] Percentage of the total training steps that we will use for the warmup (0<=warmup_factor<=1)",  # Description of the parameter.  # Beschreibung des Parameters.
)




**************** continue














    









    
    parser.add_argument(
        "--cnn_model_name",
        type=str,
        default="efficientnet_b4",
        help="[NEW MODEL] CNN model name from torchvision models, see https://pytorch.org/vision/stable/models.html "
        "for a list of available models.",
    )

    parser.add_argument(
        "--do_not_load_pretrained_cnn",
        action="store_true",
        help="[NEW MODEL] Do not load the pretrained weights for the cnn model",
    )

    parser.add_argument(
        "--embedded_size",
        type=int,
        default=512,
        help="[NEW MODEL] The size of the embedding for the encoder.",
    )

    parser.add_argument(
        "--lstm_hidden_size",
        type=int,
        default=512,
        help="[NEW MODEL LSTM] The size of the hidden state for the LSTM.",
    )

    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="[NEW MODEL Transformers] Number of heads in the multi-head attention",
    )

    parser.add_argument(
        "--num_layers_encoder",
        type=int,
        default=4,
        help="[NEW MODEL] Number of transformer layers in the encoder",
    )

    parser.add_argument(
        "--bidirectional_lstm",
        action="store_true",
        help="[NEW MODEL LSTM] Forward or bidirectional LSTM",
    )

    parser.add_argument(
        "--dropout_cnn_out",
        type=float,
        default=0.3,
        help="[NEW MODEL] Dropout rate for the output of the CNN",
    )

    parser.add_argument(
        "--positional_embeddings_dropout",
        type=float,
        default=0.1,
        help="[NEW MODEL Transformer] Dropout rate for the positional embeddings",
    )

    parser.add_argument(
        "--dropout_encoder",
        type=float,
        default=0.1,
        help="[NEW MODEL] Dropout rate for the encoder",
    )

    parser.add_argument(
        "--dropout_encoder_features",
        type=float,
        default=0.3,
        help="[NEW MODEL] Dropout probability of the encoder output",
    )

    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.2,
        help="[TRANSFORMER] Probability of masking each input vector in the transformer encoder",
    )

    parser.add_argument(
        "--sequence_size",
        type=int,
        default=5,
        help="[NEW MODEL] Length of the input sequence. Placeholder for the future, only 5 supported",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="If new_model is True, the path to the checkpoint to a pretrained model in the image reordering task. "
        "If continue_training is True, the path to the checkpoint to continue training from.",
    )

    parser.add_argument(
        "--control_mode",
        type=str,
        default="keyboard",
        choices=["keyboard", "controller"],
        help="Model output format: keyboard (Classification task: 9 classes) "
        "or controller (Regression task: 2 variables)",
    )

    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="[NEW MODEL] Label smoothing in the CrossEntropyLoss "
        "if we are in the classification task (control_mode == 'keyboard')",
    )

    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs/TPUs to use. ",
    )

    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "tpu", "gpu", "cpu", "ipu"],
        help="Accelerator to use. If 'auto', tries to automatically detect TPU, GPU, CPU or IPU system",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="16",
        choices=["bf16", "16", "32", "64"],
        help=" Double precision (64), full precision (32), "
        "half precision (16) or bfloat16 precision (bf16). "
        "Can be used on CPU, GPU or TPUs.",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Supports passing different training strategies with aliases (ddp, ddp_spawn, etc)",
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["wandb", "tensorboard"],
        help="Report to wandb or tensorboard",
    )

    parser.add_argument(
        "--find_lr",
        action="store_true",
        help="Find the optimal learning rate for the model. We will use Pytorch Lightning's find_lr function. "
        "See: "
        "https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#learning-rate-finder",
    )

    args = parser.parse_args()

    if args.train_new:
        train_new_model(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            cnn_model_name=args.cnn_model_name,
            accumulation_steps=args.accumulation_steps,
            hide_map_prob=args.hide_map_prob,
            dropout_images_prob=args.dropout_images_prob,
            variable_weights=args.variable_weights,
            control_mode=args.control_mode,
            val_check_interval=args.val_check_interval,
            dataloader_num_workers=args.dataloader_num_workers,
            pretrained_cnn=not args.do_not_load_pretrained_cnn,
            embedded_size=args.embedded_size,
            nhead=args.nhead,
            num_layers_encoder=args.num_layers_encoder,
            lstm_hidden_size=args.lstm_hidden_size,
            dropout_cnn_out=args.dropout_cnn_out,
            dropout_encoder_features=args.dropout_encoder_features,
            positional_embeddings_dropout=args.positional_embeddings_dropout,
            dropout_encoder=args.dropout_encoder,
            mask_prob=args.mask_prob,
            sequence_size=args.sequence_size,
            encoder_type=args.encoder_type,
            bidirectional_lstm=args.bidirectional_lstm,
            checkpoint_path=args.checkpoint_path,
            label_smoothing=args.label_smoothing,
            devices=args.devices,
            accelerator=args.accelerator,
            precision=args.precision,
            strategy=args.strategy,
            report_to=args.report_to,
            find_lr=args.find_lr,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            optimizer_name=args.optimizer_name,
            scheduler_name=args.scheduler_name,
            warmup_factor=args.warmup_factor,
        )

    else:
        continue_training(
            checkpoint_path=args.checkpoint_path,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            max_epochs=args.max_epochs,
            mask_prob=args.mask_prob,
            hide_map_prob=args.hide_map_prob,
            dropout_images_prob=args.dropout_images_prob,
            dataloader_num_workers=args.dataloader_num_workers,
            devices=args.devices,
            accelerator=args.accelerator,
            precision=args.precision,
            strategy=args.strategy,
            report_to=args.report_to,
        )
