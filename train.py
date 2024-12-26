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





///////////////////// C O N T I N U E  











    if report_to == "tensorboard":
        logger = pl_loggers.TensorBoardLogger(
            save_dir=output_dir,
            name=experiment_name,
        )
    elif report_to == "wandb":
        logger = pl_loggers.WandbLogger(
            name=experiment_name,
            # id=experiment_name,
            # resume=None,
            project="TEDD1104",
            save_dir=output_dir,
        )
    else:
        raise ValueError(
            f"Unknown logger: {report_to}. Please use 'tensorboard' or 'wandb'."
        )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        monitor="Validation/acc_k@1_macro",
        mode="max",
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"

    model.accelerator = accelerator

    trainer = pl.Trainer(
        devices=devices,
        accelerator=accelerator,
        precision=precision if precision == "bf16" else int(precision),
        strategy=strategy,
        val_check_interval=val_check_interval,
        accumulate_grad_batches=accumulation_steps,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[
            # pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),
            checkpoint_callback,
            lr_monitor,
        ],
        gradient_clip_val=1.0 if optimizer_name.lower() != "adafactor" else 0.0,
        log_every_n_steps=100,
        auto_lr_find=find_lr,
    )


















    if find_lr:
        print(f"We will try to find the optimal learning rate.")
        lr_finder = trainer.tuner.lr_find(model, datamodule=data)
        print(lr_finder.results)
        fig = lr_finder.plot(suggest=True)
        fig.savefig(os.path.join(output_dir, "lr_finder.png"))
        new_lr = lr_finder.suggestion()
        print(f"We will train with the suggested learning rate: {new_lr}")
        model.hparams.learning_rate = new_lr

    trainer.fit(model, datamodule=data)

    print(f"Best model path: {checkpoint_callback.best_model_path}")
    if test_dir:
        trainer.test(datamodule=data, ckpt_path="best")


def continue_training(
    checkpoint_path: str,
    train_dir: str,
    val_dir: str,
    batch_size: int,
    max_epochs: int,
    output_dir,
    accumulation_steps,
    devices: int = 1,
    accelerator: str = "auto",
    precision: str = "16",
    strategy=None,
    test_dir: str = None,
    mask_prob: float = 0.0,
    hide_map_prob: float = 0.0,
    dropout_images_prob=None,
    dataloader_num_workers=os.cpu_count(),
    val_check_interval: float = 0.25,
    report_to: str = "wandb",
):

    """
    Continues training a model from a checkpoint.

    :param str checkpoint_path: Path to the checkpoint to continue training from
    :param str train_dir: The directory containing the training data.
    :param str val_dir: The directory containing the validation data.
    :param str output_dir: The directory to save the model to.
    :param int batch_size: The batch size.
    :param int accumulation_steps: The number of steps to accumulate gradients.
    :param int devices: Number of devices to use.
    :param str accelerator: Accelerator to use. If 'auto', tries to automatically detect TPU, GPU, CPU or IPU system.
    :param str precision: Precision to use. Double precision (64), full precision (32), half precision (16) or bfloat16
                          precision (bf16). Can be used on CPU, GPU or TPUs.
    :param str strategy: Strategy to use for data parallelism. "None" for no data parallelism,
                         ddp_find_unused_parameters_false for DDP.
    :param str report_to: Where to report the results. "tensorboard" for TensorBoard, "wandb" for W&B.
    :param int max_epochs: The maximum number of epochs to train for.
    :param bool hide_map_prob: Probability of hiding the minimap (0<=hide_map_prob<=1)
    :param float mask_prob: probability of masking each input vector in the transformer
    :param float dropout_images_prob: Probability of dropping an image (0<=dropout_images_prob<=1)
    :param str test_dir: The directory containing the test data.
    :param int dataloader_num_workers: The number of workers to use for the dataloaders.
    :param float val_check_interval: The interval in epochs to check the validation accuracy.
    """

    if dropout_images_prob is None:
        dropout_images_prob = [0.0, 0.0, 0.0, 0.0, 0.0]

    print(f"Restoring checkpoint: {checkpoint_path}")

    model = Tedd1104ModelPL.load_from_checkpoint(checkpoint_path=checkpoint_path)

    print("Done! Preparing to continue training...")

    data = Tedd1104DataModule(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=batch_size,
        hide_map_prob=hide_map_prob,
        dropout_images_prob=dropout_images_prob,
        control_mode=model.control_mode,
        num_workers=dataloader_num_workers,
        token_mask_prob=mask_prob,
        transformer_nheads=None if model.encoder_type == "lstm" else model.nhead,
        sequence_length=model.sequence_size,
    )

    experiment_name = os.path.basename(
        output_dir if output_dir[-1] != "/" else output_dir[:-1]
    )
    if report_to == "tensorboard":
        logger = pl_loggers.TensorBoardLogger(
            save_dir=output_dir,
            name=experiment_name,
        )
    elif report_to == "wandb":
        logger = pl_loggers.WandbLogger(
            name=experiment_name,
            # id=experiment_name,
            # resume="allow",
            project="TEDD1104",
            save_dir=output_dir,
        )
    else:
        raise ValueError(
            f"Unknown logger: {report_to}. Please use 'tensorboard' or 'wandb'."
        )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        monitor="Validation/acc_k@1_macro",
        mode="max",
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"

    trainer = pl.Trainer(
        devices=devices,
        accelerator=accelerator,
        precision=precision if precision == "bf16" else int(precision),
        strategy=strategy,
        val_check_interval=val_check_interval,
        accumulate_grad_batches=accumulation_steps,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[
            pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),
            checkpoint_callback,
            lr_monitor,
        ],
        gradient_clip_val=1.0,
        log_every_n_steps=100,
    )

    trainer.fit(
        ckpt_path=checkpoint_path,
        model=model,
        datamodule=data,
    )

    # print(f"Best model path: {checkpoint_callback.best_model_path}")

    if test_dir:
        trainer.test(datamodule=data, ckpt_path="best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a T.E.D.D. 1104 model in the supervised self-driving task."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train_new",
        action="store_true",
        help="Train a new model",
    )

    group.add_argument(
        "--continue_training",
        action="store_true",
        help="Continues training a model from a checkpoint.",
    )

    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="The directory containing the training data.",
    )

    parser.add_argument(
        "--val_dir",
        type=str,
        required=True,
        help="The directory containing the validation data.",
    )

    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="The directory containing the test data.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory to save the model to.",
    )

    parser.add_argument(
        "--encoder_type",
        type=str,
        choices=["lstm", "transformer"],
        default="transformer",
        help="The Encoder type to use: transformer or lstm",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="The batch size for training and eval.",
    )

    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="The number of steps to accumulate gradients.",
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        required=True,
        help="The maximum number of epochs to train for.",
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of CPU workers for the Data Loaders",
    )

    parser.add_argument(
        "--hide_map_prob",
        type=float,
        default=0.0,
        help="Probability of hiding the minimap in the sequence (0<=hide_map_prob<=1)",
    )

    parser.add_argument(
        "--dropout_images_prob",
        type=float,
        nargs=5,
        default=[0.0, 0.0, 0.0, 0.0, 0.0],
        help="Probability of dropping each image in the sequence (0<=dropout_images_prob<=1)",
    )

    parser.add_argument(
        "--variable_weights",
        type=float,
        nargs="+",
        default=None,
        help="List of weights for the loss function [9] if control_mode == 'keyboard' "
        "or [2] if control_mode == 'controller'",
    )

    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=1.0,
        help="The interval in epochs between validation checks.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="[NEW MODEL] The learning rate for the optimizer.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="[NEW MODEL]] AdamW Weight Decay",
    )

    parser.add_argument(
        "--optimizer_name",
        type=str,
        default="adamw",
        choices=["adamw", "adafactor"],
        help="[NEW MODEL] The optimizer to use: adamw or adafactor. Adafactor requires fairseq to be installed. "
        "pip install fairseq",
    )

    parser.add_argument(
        "--scheduler_name",
        type=str,
        default="linear",
        choices=["linear", "plateau", "polynomial", "cosine"],
        help="[NEW MODEL] The scheduler to use: linear, polynomial, cosine, plateau.",
    )

    parser.add_argument(
        "--warmup_factor",
        type=float,
        default=0.05,
        help="[NEW MODEL] Percentage of the total training steps that we will use for the warmup (0<=warmup_factor<=1)",
    )

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
