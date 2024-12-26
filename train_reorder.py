from model import Tedd1104ModelPLForImageReordering  # Import the image reordering model.  # Modell für die Neuanordnung von Bildern importieren.
from dataset import count_examples  # Import function to count dataset examples.  # Funktion zum Zählen der Datensatzbeispiele importieren.
import argparse  # Import library to handle command-line arguments.  # Bibliothek zum Verarbeiten von Befehlszeilenargumenten importieren.
from dataset_image_reordering import Tedd1104ataModuleForImageReordering  # Import data module for image reordering.  # Datenmodul für die Neuanordnung von Bildern importieren.
import os  # Import library to interact with the operating system.  # Bibliothek zur Interaktion mit dem Betriebssystem importieren.
from pytorch_lightning import loggers as pl_loggers  # Import PyTorch Lightning loggers.  # PyTorch Lightning-Logger importieren.
import pytorch_lightning as pl  # Import PyTorch Lightning for training pipelines.  # PyTorch Lightning für Trainingspipelines importieren.
import math  # Import math library for mathematical operations.  # Mathebibliothek für mathematische Operationen importieren.

def train_new_model(  # Define a function to train a new model.  # Funktion zum Trainieren eines neuen Modells definieren.
    train_dir: str,  # Directory containing training data.  # Verzeichnis mit Trainingsdaten.
    val_dir: str,  # Directory containing validation data.  # Verzeichnis mit Validierungsdaten.
    output_dir: str,  # Directory to save model outputs.  # Verzeichnis zum Speichern der Modellausgaben.
    batch_size: int,  # Batch size for training.  # Batchgröße für das Training.
    max_epochs: int,  # Maximum number of training epochs.  # Maximale Anzahl der Trainingsepochen.
    cnn_model_name: str,  # Name of the CNN model to use.  # Name des zu verwendenden CNN-Modells.
    devices: int = 1,  # Number of devices to use for training.  # Anzahl der Geräte für das Training.
    accelerator: str = "auto",  # Hardware accelerator (e.g., GPU/TPU).  # Hardwarebeschleuniger (z. B. GPU/TPU).
    precision: str = "bf16",  # Precision type (e.g., 16-bit, bf16).  # Genauigkeitstyp (z. B. 16-Bit, bf16).
    strategy=None,  # Training strategy (e.g., distributed training).  # Trainingsstrategie (z. B. verteiltes Training).
    accumulation_steps: int = 1,  # Gradient accumulation steps.  # Schritte zur Gradientenakkumulation.
    hide_map_prob: float = 0.0,  # Probability to hide the map in training.  # Wahrscheinlichkeit, die Karte im Training zu verstecken.
    test_dir: str = None,  # Directory containing test data.  # Verzeichnis mit Testdaten.
    dropout_images_prob=None,  # Probability to drop out images during training.  # Wahrscheinlichkeit, Bilder während des Trainings auszuschließen.
    val_check_interval: float = 0.25,  # Interval to check validation metrics.  # Intervall zur Überprüfung der Validierungsmetriken.
    dataloader_num_workers=os.cpu_count(),  # Number of workers for data loading.  # Anzahl der Arbeiter für das Laden von Daten.
    pretrained_cnn: bool = True,  # Use a pretrained CNN.  # Verwendung eines vortrainierten CNN.
    embedded_size: int = 512,  # Size of the embedded vectors.  # Größe der eingebetteten Vektoren.
    nhead: int = 8,  # Number of attention heads.  # Anzahl der Attention-Köpfe.
    num_layers_encoder: int = 1,  # Number of encoder layers.  # Anzahl der Encoder-Schichten.
    dropout_cnn_out: float = 0.1,  # Dropout probability for CNN outputs.  # Dropout-Wahrscheinlichkeit für CNN-Ausgaben.
    positional_embeddings_dropout: float = 0.1,  # Dropout for positional embeddings.  # Dropout für Positions-Embeddings.
    dropout_encoder: float = 0.1,  # Dropout for the encoder.  # Dropout für den Encoder.
    dropout_encoder_features: float = 0.8,  # Dropout for encoder features.  # Dropout für Encoder-Features.
    mask_prob: float = 0.0,  # Probability of masking input features.  # Wahrscheinlichkeit des Maskierens von Eingabefeatures.
    sequence_size: int = 5,  # Size of input sequence.  # Größe der Eingabesequenz.
    report_to: str = "wandb",  # Reporting tool for logs (e.g., Weights & Biases).  # Reporting-Tool für Logs (z. B. Weights & Biases).
    find_lr: bool = False,  # Whether to find the optimal learning rate.  # Optimale Lernrate suchen oder nicht.
    optimizer_name: str = "adamw",  # Name of the optimizer.  # Name des Optimierers.
    scheduler_name: str = "linear",  # Name of the learning rate scheduler.  # Name des Lernratenplaners.
    learning_rate: float = 1e-5,  # Initial learning rate.  # Anfangslernrate.
    weight_decay: float = 1e-3,  # Weight decay for regularization.  # Gewichtsverfall zur Regularisierung.
    warmup_factor: float = 0.05,  # Warm-up factor for learning rate.  # Warm-up-Faktor für die Lernrate.
):


"""
Train a new model.

:param str train_dir: The directory containing the training data.  # Directory für Trainingsdaten.  
:param str val_dir: The directory containing the validation data.  # Verzeichnis mit Validierungsdaten.
:param str output_dir: The directory to save the model to.  # Verzeichnis, in dem das Modell gespeichert wird.
:param int batch_size: The batch size.  # Batch-Größe.
:param int accumulation_steps: The number of steps to accumulate gradients.  # Anzahl der Schritte zum Gradientenakkumulation.
:param int max_epochs: The maximum number of epochs to train for.  # Maximale Anzahl an Epochen für das Training.
:param bool hide_map_prob: Probability of hiding the minimap (0<=hide_map_prob<=1)  # Wahrscheinlichkeit, die Minimap auszublenden (0<=hide_map_prob<=1).
:param float dropout_images_prob: Probability of dropping an image (0<=dropout_images_prob<=1)  # Wahrscheinlichkeit, ein Bild zu entfernen (0<=dropout_images_prob<=1).
:param str test_dir: The directory containing the test data.  # Verzeichnis mit Testdaten.
:param float val_check_interval: The interval to check the validation accuracy.  # Intervall zur Überprüfung der Validierungsgenauigkeit.
:param int devices: Number of devices to use.  # Anzahl der zu verwendenden Geräte.
:param str accelerator: Accelerator to use. If 'auto', tries to automatically detect TPU, GPU, CPU or IPU system.  # Accelerator, der verwendet wird. Bei 'auto' wird versucht, TPU, GPU, CPU oder IPU-System automatisch zu erkennen.
:param str precision: Precision to use. Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16). Can be used on CPU, GPU or TPUs.  # Verwendete Genauigkeit. Doppelte Genauigkeit (64), volle Genauigkeit (32), halbe Genauigkeit (16) oder bfloat16 (bf16). Kann auf CPU, GPU oder TPUs verwendet werden.
:param str strategy: Strategy to use for data parallelism. "None" for no data parallelism, ddp_find_unused_parameters_false for DDP.  # Strategie für Datenparallelismus. "None" für keine Parallelisierung, ddp_find_unused_parameters_false für DDP.
:param str report_to: Where to report the results. "tensorboard" for TensorBoard, "wandb" for W&B.  # Wo die Ergebnisse gemeldet werden. "tensorboard" für TensorBoard, "wandb" für W&B.
:param int dataloader_num_workers: The number of workers to use for the dataloader.  # Anzahl der Worker für den Dataloader.
:param int embedded_size: Size of the output embedding  # Größe des Ausgabeeinbettungsvektors.
:param float dropout_cnn_out: Dropout rate for the output of the CNN  # Dropout-Rate für die Ausgabe der CNN.
:param str cnn_model_name: Name of the CNN model from torchvision.models  # Name des CNN-Modells aus torchvision.models.
:param bool pretrained_cnn: If True, the model will be loaded with pretrained weights  # Falls True, wird das Modell mit vortrainierten Gewichten geladen.
:param int embedded_size: Size of the input feature vectors  # Größe der Eingabe-Feature-Vektoren.
:param int nhead: Number of heads in the multi-head attention  # Anzahl der Köpfe in der Multi-Head-Attention.
:param int num_layers_encoder: Number of transformer layers in the encoder  # Anzahl der Transformer-Schichten im Encoder.
:param float mask_prob: Probability of masking each input vector in the transformer  # Wahrscheinlichkeit, jeden Eingabevektor im Transformer zu maskieren.
:param float positional_embeddings_dropout: Dropout rate for the positional embeddings  # Dropout-Rate für die Positions-Einbettungen.
:param int sequence_size: Length of the input sequence  # Länge der Eingabesequenz.
:param float dropout_encoder: Dropout rate for the encoder  # Dropout-Rate für den Encoder.
:param float dropout_encoder_features: Dropout probability of the encoder output  # Dropout-Wahrscheinlichkeit der Encoder-Ausgabe.
:param str optimizer_name: Optimizer to use: adamw or adafactor  # Optimierer, der verwendet wird: adamw oder adafactor.
:param str scheduler_name: Scheduler to use: linear or plateau  # Scheduler, der verwendet wird: linear oder plateau.
:param float learning_rate: Learning rate  # Lernrate.
:param float weight_decay: Weight decay  # Gewichtsdämpfung.
:param bool find_lr: Whether to find the learning rate. We will use PytorchLightning's find_lr function. See: https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#learning-rate-finder  # Ob die Lernrate gefunden werden soll. Wir verwenden die find_lr-Funktion von PytorchLightning. Siehe: https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#learning-rate-finder.
"""

if dropout_images_prob is None:  # Checks if the dropout probability for images is set to None. # Überprüft, ob die Dropout-Wahrscheinlichkeit für Bilder auf None gesetzt ist.
    dropout_images_prob = [0.0, 0.0, 0.0, 0.0, 0.0]  # If None, set default values for the dropout probability. # Falls None, wird der Standardwert für die Dropout-Wahrscheinlichkeit gesetzt.

num_examples = count_examples(dataset_dir=train_dir)  # Counts the number of training examples in the dataset directory. # Zählt die Anzahl der Trainingsbeispiele im Datensatzverzeichnis.

num_update_steps_per_epoch = math.ceil(  # Calculates the number of update steps per epoch, rounding up to the nearest integer. # Berechnet die Anzahl der Update-Schritte pro Epoche und rundet auf die nächste ganze Zahl.
    math.ceil(math.ceil(num_examples / batch_size) / accumulation_steps / devices)  # Divides the number of examples by batch size, then by accumulation steps and devices. # Teilt die Anzahl der Beispiele durch die Batch-Größe, dann durch die Akkumulationsschritte und Geräte.
)

max_train_steps = max_epochs * num_update_steps_per_epoch  # Calculates the maximum number of training steps based on the number of epochs and update steps per epoch. # Berechnet die maximale Anzahl der Trainingsschritte basierend auf der Anzahl der Epochen und der Update-Schritte pro Epoche.
num_warmup_steps = int(max_train_steps * warmup_factor)  # Calculates the number of warmup steps based on a warmup factor. # Berechnet die Anzahl der Warmup-Schritte basierend auf einem Warmup-Faktor.

print(  # Prints training details for the user to review. # Gibt Trainingsdetails aus, damit der Benutzer sie überprüfen kann.
    f"\n*** Training info ***\n"
    f"Number of training examples: {num_examples}\n"  # Outputs the number of training examples. # Gibt die Anzahl der Trainingsbeispiele aus.
    f"Number of update steps per epoch: {num_update_steps_per_epoch}\n"  # Outputs the number of update steps per epoch. # Gibt die Anzahl der Update-Schritte pro Epoche aus.
    f"Max training steps: {max_train_steps}\n"  # Outputs the maximum number of training steps. # Gibt die maximale Anzahl der Trainingsschritte aus.
    f"Number of warmup steps: {num_warmup_steps}\n"  # Outputs the number of warmup steps. # Gibt die Anzahl der Warmup-Schritte aus.
    f"Optimizer: {optimizer_name}\n"  # Outputs the name of the optimizer. # Gibt den Namen des Optimierers aus.
    f"Scheduler: {scheduler_name}\n"  # Outputs the name of the scheduler. # Gibt den Namen des Schedulers aus.
    f"Learning rate: {learning_rate}\n"  # Outputs the learning rate. # Gibt die Lernrate aus.
)

model: Tedd1104ModelPLForImageReordering = Tedd1104ModelPLForImageReordering(  # Initializes the model for image reordering. # Initialisiert das Modell zur Bildumordnung.
    cnn_model_name=cnn_model_name,  # Name of the CNN model to use. # Name des zu verwendenden CNN-Modells.
    pretrained_cnn=pretrained_cnn,  # Whether to use a pre-trained CNN model. # Ob ein vortrainiertes CNN-Modell verwendet werden soll.
    embedded_size=embedded_size,  # Size of the embedded vector. # Größe des eingebetteten Vektors.
    nhead=nhead,  # Number of attention heads in the model. # Anzahl der Aufmerksamkeitsköpfe im Modell.
    num_layers_encoder=num_layers_encoder,  # Number of layers in the encoder. # Anzahl der Schichten im Encoder.
    dropout_cnn_out=dropout_cnn_out,  # Dropout probability for CNN output. # Dropout-Wahrscheinlichkeit für CNN-Ausgaben.
    positional_embeddings_dropout=positional_embeddings_dropout,  # Dropout probability for positional embeddings. # Dropout-Wahrscheinlichkeit für Positionsembeddings.
    dropout_encoder=dropout_encoder,  # Dropout probability for encoder layers. # Dropout-Wahrscheinlichkeit für Encoder-Schichten.
    dropout_encoder_features=dropout_encoder_features,  # Dropout probability for encoder features. # Dropout-Wahrscheinlichkeit für Encoder-Features.
    sequence_size=sequence_size,  # Size of the input sequence. # Größe der Eingabesequenz.
    accelerator=accelerator,  # Hardware accelerator (e.g., GPU or TPU). # Hardwarebeschleuniger (z.B. GPU oder TPU).
    learning_rate=learning_rate,  # Learning rate for optimization. # Lernrate für die Optimierung.
    weight_decay=weight_decay,  # Weight decay for regularization. # Gewichtung der Regularisierung für das Abklingen der Gewichte.
    optimizer_name=optimizer_name,  # Name of the optimizer. # Name des Optimierers.
    scheduler_name=scheduler_name,  # Name of the learning rate scheduler. # Name des Lernraten-Schedulers.
    num_warmup_steps=num_warmup_steps,  # Number of warmup steps. # Anzahl der Warmup-Schritte.
    num_training_steps=max_train_steps,  # Maximum number of training steps. # Maximale Anzahl der Trainingsschritte.
)

if not os.path.exists(output_dir):  # Checks if the output directory exists. # Überprüft, ob das Ausgabeverzeichnis existiert.
    print(f"{output_dir} does not exits. We will create it.")  # If it doesn't exist, notify the user. # Falls es nicht existiert, wird der Benutzer benachrichtigt.
    os.makedirs(output_dir)  # Creates the output directory. # Erstellt das Ausgabeverzeichnis.

data = Tedd1104ataModuleForImageReordering(  # Initializes the data module for image reordering. # Initialisiert das Datenmodul zur Bildumordnung.
    train_dir=train_dir,  # Directory for training data. # Verzeichnis für Trainingsdaten.
    val_dir=val_dir,  # Directory for validation data. # Verzeichnis für Validierungsdaten.
    test_dir=test_dir,  # Directory for test data. # Verzeichnis für Testdaten.
    batch_size=batch_size,  # Size of each batch of data. # Größe jedes Batchs von Daten.
    hide_map_prob=hide_map_prob,  # Probability of hiding map during training. # Wahrscheinlichkeit, dass die Karte während des Trainings ausgeblendet wird.
    dropout_images_prob=dropout_images_prob,  # Dropout probability for images during training. # Dropout-Wahrscheinlichkeit für Bilder während des Trainings.
    num_workers=dataloader_num_workers,  # Number of workers for data loading. # Anzahl der Arbeiter für das Laden der Daten.
    token_mask_prob=mask_prob,  # Probability of masking tokens during training. # Wahrscheinlichkeit, dass Tokens während des Trainings maskiert werden.
    transformer_nheads=None if model.encoder_type == "lstm" else model.nhead,  # Sets the number of transformer heads, except for LSTM models. # Legt die Anzahl der Transformer-Köpfe fest, außer bei LSTM-Modellen.
    sequence_length=model.sequence_size,  # Sequence length for input data. # Sequenzlänge für Eingabedaten.
)

experiment_name = os.path.basename(  # Extracts the experiment name from the output directory. # Extrahiert den Experimentnamen aus dem Ausgabeverzeichnis.
    output_dir if output_dir[-1] != "/" else output_dir[:-1]  # Ensures the output directory name does not end with a slash. # Stellt sicher, dass der Ausgabeverzeichnisname nicht mit einem Schrägstrich endet.
)

if report_to == "tensorboard":  # Checks if logging is set to TensorBoard. # Überprüft, ob das Logging auf TensorBoard eingestellt ist.
    logger = pl_loggers.TensorBoardLogger(  # Initializes TensorBoard logger. # Initialisiert den TensorBoard-Logger.
        save_dir=output_dir,  # Directory to save TensorBoard logs. # Verzeichnis zum Speichern der TensorBoard-Protokolle.
        name=experiment_name,  # Name of the experiment. # Name des Experiments.
    )
elif report_to == "wandb":  # Checks if logging is set to Wandb. # Überprüft, ob das Logging auf Wandb eingestellt ist.
    logger = pl_loggers.WandbLogger(  # Initializes Wandb logger. # Initialisiert den Wandb-Logger.
        name=experiment_name,  # Name of the experiment. # Name des Experiments.
        project="TEDD1104_reorder",  # Project name on Wandb. # Projektname auf Wandb.
        save_dir=output_dir,  # Directory to save Wandb logs. # Verzeichnis zum Speichern der Wandb-Protokolle.
    )
else:  # If neither TensorBoard nor Wandb is selected, raise an error. # Falls weder TensorBoard noch Wandb ausgewählt ist, wird ein Fehler ausgelöst.
    raise ValueError(  
        f"Unknown logger: {report_to}. Please use 'tensorboard' or 'wandb'."  # Outputs an error if an unknown logger is specified. # Gibt einen Fehler aus, wenn ein unbekannter Logger angegeben ist.
    )

lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")  # Initializes the learning rate monitor for logging. # Initialisiert den Lernraten-Monitor für das Logging.
checkpoint_callback = pl.callbacks.ModelCheckpoint(  # Initializes model checkpointing. # Initialisiert das Speichern von Modellzuständen.
    dirpath=output_dir,  # Directory to save model checkpoints. # Verzeichnis zum Speichern von Modell-Snapshots.
    monitor="Validation/acc",  # Metric to monitor during training. # Metrik, die während des Trainings überwacht wird.
    mode="max",  # Mode for monitoring (maximizing the metric). # Modus für die Überwachung (maximieren der Metrik).
    save_last=True  # Saves the last model checkpoint. # Speichert den letzten Modell-Snapshot.
)
checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"  # Sets the filename format for the last checkpoint. # Legt das Dateinamensformat für den letzten Snapshot fest.

trainer = pl.Trainer(  # Initializes the PyTorch Lightning trainer. # Initialisiert den PyTorch Lightning Trainer.
    devices=devices,  # Number of devices (GPUs/TPUs) to use for training. # Anzahl der Geräte (GPUs/TPUs) für das Training.
    accelerator=accelerator,  # Type of accelerator to use (e.g., GPU). # Art des zu verwendenden Beschleunigers (z.B. GPU).
    precision=precision if precision == "bf16" else int(precision),  # Sets the precision for training (e.g., bf16 or 32-bit). # Setzt die Präzision für das Training (z.B. bf16 oder 32-Bit).
    strategy=strategy,  # Defines the parallelization strategy. # Definiert die Parallelisierungsstrategie.
    val_check_interval=val_check_interval,  # Interval for validation checks during training. # Intervall für Validierungsprüfungen während des Trainings.
    accumulate_grad_batches=accumulation_steps,  # Number of batches to accumulate gradients over. # Anzahl der Batches, über die Gradienten akkumuliert werden.
    max_epochs=max_epochs,  # Maximum number of epochs to train for. # Maximale Anzahl der Epochen für das Training.
    logger=logger,  # Logger to use for tracking the training process. # Logger, der für das Verfolgen des Trainingsprozesses verwendet wird.
    callbacks=[  # List of callbacks to use during training. # Liste der Callback-Funktionen, die während des Trainings verwendet werden.
        pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),  # Callback for stochastic weight averaging. # Callback für stochastisches Gewichtsmittel.
        checkpoint_callback,  # Include the checkpoint callback. # Füge den Snapshot-Callback hinzu.
        lr_monitor,  # Include the learning rate monitor callback. # Füge den Lernraten-Monitor-Callback hinzu.
    ],
    gradient_clip_val=1.0 if optimizer_name.lower() != "adafactor" else 0.0,  # Clip gradients to avoid exploding gradients, except for Adafactor optimizer. # Beschneidet Gradienten, um explodierende Gradienten zu vermeiden, außer für den Adafactor-Optimierer.
    log_every_n_steps=100,  # Logs training progress every 100 steps. # Protokolliert den Trainingsfortschritt alle 100 Schritte.
    auto_lr_find=find_lr,  # Automatically finds the optimal learning rate. # Findet automatisch die optimale Lernrate.
)


if find_lr:  # If we are looking for the optimal learning rate.  # Wenn wir den optimalen Lernratenwert suchen.
    print(f"We will try to find the optimal learning rate.")  # Print message about finding learning rate.  # Ausgabe der Nachricht, dass der Lernratenwert gefunden wird.
    lr_finder = trainer.tuner.lr_find(model, datamodule=data)  # Use the trainer's tuner to find the learning rate for the model.  # Verwenden des Tuners des Trainers, um den Lernratenwert für das Modell zu finden.
    print(lr_finder.results)  # Print the results of the learning rate search.  # Ausgabe der Ergebnisse der Lernraten-Suche.
    fig = lr_finder.plot(suggest=True)  # Plot the learning rate graph, with a suggested rate.  # Plot der Lernraten-Grafik mit einem empfohlenen Wert.
    fig.savefig(os.path.join(output_dir, "lr_finder.png"))  # Save the plot as an image in the output directory.  # Speichern des Plots als Bild im Ausgabeverzeichnis.
    new_lr = lr_finder.suggestion()  # Get the suggested learning rate.  # Abrufen der empfohlenen Lernrate.
    print(f"We will train with the suggested learning rate: {new_lr}")  # Print the new suggested learning rate.  # Ausgabe der neuen empfohlenen Lernrate.
    model.hparams.learning_rate = new_lr  # Set the model's learning rate to the suggested value.  # Setzen der Lernrate des Modells auf den empfohlenen Wert.

trainer.fit(model, datamodule=data)  # Train the model with the provided data.  # Trainiere das Modell mit den bereitgestellten Daten.

print(f"Best model path: {checkpoint_callback.best_model_path}")  # Print the path to the best saved model.  # Ausgabe des Pfads zum besten gespeicherten Modell.
if test_dir:  # If a test directory is provided, run the test.  # Wenn ein Testverzeichnis angegeben ist, führe den Test aus.
    trainer.test(datamodule=data, ckpt_path="best")  # Test the model using the best checkpoint.  # Teste das Modell mit dem besten gespeicherten Checkpoint.

def continue_training(  # Define a function to continue training from a checkpoint.  # Definiere eine Funktion, um das Training von einem Checkpoint fortzusetzen.
    checkpoint_path: str,  # Path to the checkpoint to resume from.  # Pfad zum Checkpoint, von dem aus das Training fortgesetzt wird.
    train_dir: str,  # Path to the training data directory.  # Pfad zum Trainingsdatenverzeichnis.
    val_dir: str,  # Path to the validation data directory.  # Pfad zum Validierungsdatenverzeichnis.
    batch_size: int,  # Batch size for training.  # Batch-Größe für das Training.
    max_epochs: int,  # Maximum number of training epochs.  # Maximale Anzahl von Trainings-Epochen.
    output_dir,  # Directory to store output, such as logs and models.  # Verzeichnis zur Speicherung von Ausgaben, wie Logs und Modellen.
    accumulation_steps,  # Number of gradient accumulation steps.  # Anzahl der Gradientenakkumulations-Schritte.
    devices: int = 1,  # Number of devices to use for training (e.g., GPUs).  # Anzahl der Geräte, die für das Training verwendet werden (z. B. GPUs).
    accelerator: str = "auto",  # Type of accelerator (e.g., "auto", "cpu", "gpu").  # Art des Beschleunigers (z. B. "auto", "cpu", "gpu").
    precision: str = "16",  # Precision for training (e.g., "16" for half precision).  # Präzision für das Training (z. B. "16" für halbe Präzision).
    strategy=None,  # Strategy for distributed training (e.g., "dp", "ddp").  # Strategie für verteiltes Training (z. B. "dp", "ddp").
    test_dir: str = None,  # Optional directory for testing after training.  # Optionales Verzeichnis für den Test nach dem Training.
    mask_prob: float = 0.2,  # Probability of applying a mask during training.  # Wahrscheinlichkeit, dass ein Maskierungsverfahren während des Trainings angewendet wird.
    hide_map_prob: float = 0.0,  # Probability of hiding the map during training.  # Wahrscheinlichkeit, dass die Karte während des Trainings ausgeblendet wird.
    dropout_images_prob=None,  # Probability of applying dropout to images during training.  # Wahrscheinlichkeit, dass während des Trainings Dropout auf Bilder angewendet wird.
    dataloader_num_workers=os.cpu_count(),  # Number of workers for data loading.  # Anzahl der Arbeiter für das Laden von Daten.
    val_check_interval: float = 0.25,  # Interval for validation checks during training.  # Intervall für Validierungsüberprüfungen während des Trainings.
    report_to: str = "wandb",  # The service to report results to (e.g., "wandb").  # Der Dienst, an den Ergebnisse gemeldet werden (z. B. "wandb").
):

    """
    Continues training a model from a checkpoint.  # Description: This function resumes training from a saved checkpoint.  # Beschreibung: Diese Funktion setzt das Training von einem gespeicherten Checkpoint fort.

    :param str checkpoint_path: Path to the checkpoint to continue training from  # Description: The path to the saved model checkpoint.  # Beschreibung: Der Pfad zum gespeicherten Modell-Checkpoint.
    :param str train_dir: The directory containing the training data.  # Description: The directory where training data is stored.  # Beschreibung: Das Verzeichnis, in dem die Trainingsdaten gespeichert sind.
    :param str val_dir: The directory containing the validation data.  # Description: The directory where validation data is stored.  # Beschreibung: Das Verzeichnis, in dem die Validierungsdaten gespeichert sind.
    :param str output_dir: The directory to save the model to.  # Description: The location where the trained model will be saved.  # Beschreibung: Der Ort, an dem das trainierte Modell gespeichert wird.
    :param int batch_size: The batch size.  # Description: The number of samples processed in one batch.  # Beschreibung: Die Anzahl der Proben, die in einem Batch verarbeitet werden.
    :param int accumulation_steps: The number of steps to accumulate gradients.  # Description: The number of steps for which gradients are accumulated before updating the model.  # Beschreibung: Die Anzahl der Schritte, für die Gradienten akkumuliert werden, bevor das Modell aktualisiert wird.
    :param int devices: Number of devices to use.  # Description: The number of devices (e.g., GPUs) used for training.  # Beschreibung: Die Anzahl der Geräte (z. B. GPUs), die für das Training verwendet werden.
    :param str accelerator: Accelerator to use. If 'auto', tries to automatically detect TPU, GPU, CPU or IPU system.  # Description: The hardware accelerator to use (e.g., GPU, CPU, etc.).  # Beschreibung: Der Hardwarebeschleuniger, der verwendet werden soll (z. B. GPU, CPU, etc.).
    :param str precision: Precision to use. Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16). Can be used on CPU, GPU or TPUs.  # Description: Specifies the precision used for calculations (e.g., 32-bit, 64-bit, etc.).  # Beschreibung: Gibt die Genauigkeit an, die für Berechnungen verwendet wird (z. B. 32-Bit, 64-Bit, etc.).
    :param str strategy: Strategy to use for data parallelism. "None" for no data parallelism, ddp_find_unused_parameters_false for DDP.  # Description: Strategy for parallel processing across multiple devices.  # Beschreibung: Strategie für parallele Verarbeitung auf mehreren Geräten.
    :param str report_to: Where to report the results. "tensorboard" for TensorBoard, "wandb" for W&B.  # Description: Where to report training results (e.g., TensorBoard or W&B).  # Beschreibung: Wo die Trainingsergebnisse gemeldet werden sollen (z. B. TensorBoard oder W&B).
    :param int max_epochs: The maximum number of epochs to train for.  # Description: The maximum number of times the model will be trained on the full dataset.  # Beschreibung: Die maximale Anzahl von Epochen, für die das Modell auf dem vollständigen Datensatz trainiert wird.
    :param bool hide_map_prob: Probability of hiding the minimap (0<=hide_map_prob<=1)  # Description: Probability of hiding the minimap during training.  # Beschreibung: Die Wahrscheinlichkeit, dass die Minikarte während des Trainings ausgeblendet wird.
    :param float mask_prob: Probability of masking each input vector in the transformer  # Description: The probability of masking parts of the input data for transformers.  # Beschreibung: Die Wahrscheinlichkeit, Teile der Eingabedaten für Transformatoren zu maskieren.
    :param float dropout_images_prob: Probability of dropping an image (0<=dropout_images_prob<=1)  # Description: The probability that an image will be ignored during training.  # Beschreibung: Die Wahrscheinlichkeit, dass ein Bild während des Trainings ignoriert wird.
    :param str test_dir: The directory containing the test data.  # Description: The directory where test data is stored.  # Beschreibung: Das Verzeichnis, in dem die Testdaten gespeichert sind.
    :param int dataloader_num_workers: The number of workers to use for the dataloaders.  # Description: The number of workers to load the data concurrently.  # Beschreibung: Die Anzahl der Arbeiter, die die Daten gleichzeitig laden.
    :param float val_check_interval: The interval in epochs to check the validation accuracy.  # Description: How often to check validation accuracy during training.  # Beschreibung: Wie oft während des Trainings die Validierungsgenauigkeit überprüft werden soll.
    """

    if dropout_images_prob is None:  # Check if dropout_images_prob is not defined.  # Überprüft, ob dropout_images_prob nicht definiert ist.
    dropout_images_prob = [0.0, 0.0, 0.0, 0.0, 0.0]  # If not defined, set a default value.  # Falls nicht definiert, wird ein Standardwert gesetzt.

print(f"Restoring checkpoint: {checkpoint_path}")  # Prints a message indicating the restoration of the checkpoint.  # Gibt eine Nachricht aus, die die Wiederherstellung des Checkpoints anzeigt.

model = Tedd1104ModelPLForImageReordering.load_from_checkpoint(  # Load the model from the checkpoint path.  # Lädt das Modell vom Checkpoint-Pfad.
    checkpoint_path=checkpoint_path  # Path to the checkpoint file.  # Pfad zur Checkpoint-Datei.
)

print("Done! Preparing to continue training...")  # Prints a message indicating that the model has been loaded and is preparing for training.  # Gibt eine Nachricht aus, dass das Modell geladen wurde und die Vorbereitung für das Training erfolgt.

data = Tedd1104ataModuleForImageReordering(  # Initializes the data module for training.  # Initialisiert das Datenmodul für das Training.
    train_dir=train_dir,  # Directory containing the training data.  # Verzeichnis mit Trainingsdaten.
    val_dir=val_dir,  # Directory containing the validation data.  # Verzeichnis mit Validierungsdaten.
    test_dir=test_dir,  # Directory containing the test data.  # Verzeichnis mit Testdaten.
    batch_size=batch_size,  # The batch size for training.  # Die Batch-Größe für das Training.
    hide_map_prob=hide_map_prob,  # Probability to hide the map in the training process.  # Wahrscheinlichkeit, die Karte im Trainingsprozess zu verbergen.
    dropout_images_prob=dropout_images_prob,  # Probability for image dropout.  # Wahrscheinlichkeit für das Dropout von Bildern.
    num_workers=dataloader_num_workers,  # Number of workers to load the data.  # Anzahl der Arbeiter zum Laden der Daten.
    token_mask_prob=mask_prob,  # Probability for token masking.  # Wahrscheinlichkeit für das Maskieren von Tokens.
    transformer_nheads=None if model.encoder_type == "lstm" else model.nhead,  # Set the number of transformer heads based on the model encoder type.  # Setzt die Anzahl der Transformer-Köpfe basierend auf dem Modell-Typ.
    sequence_length=model.sequence_size,  # Sequence length for the model.  # Sequenzlänge für das Modell.
)

print(f"Restoring checkpoint: {checkpoint_path}")  # Prints the checkpoint restoration message again.  # Gibt die Wiederherstellung des Checkpoints erneut aus.

experiment_name = os.path.basename(  # Extracts the experiment name from the output directory path.  # Extrahiert den Experimentnamen aus dem Ausgabe-Verzeichnis-Pfad.
    output_dir if output_dir[-1] != "/" else output_dir[:-1]  # Removes trailing slash from the directory path if present.  # Entfernt den abschließenden Schrägstrich, wenn vorhanden.
)

if report_to == "tensorboard":  # Checks if the logging is for TensorBoard.  # Überprüft, ob das Logging für TensorBoard vorgesehen ist.
    logger = pl_loggers.TensorBoardLogger(  # Initializes the TensorBoard logger.  # Initialisiert den TensorBoard-Logger.
        save_dir=output_dir,  # Directory where logs will be saved.  # Verzeichnis, in dem die Logs gespeichert werden.
        name=experiment_name,  # The name of the experiment.  # Der Name des Experiments.
    )
elif report_to == "wandb":  # Checks if the logging is for Wandb.  # Überprüft, ob das Logging für Wandb vorgesehen ist.
    logger = pl_loggers.WandbLogger(  # Initializes the Wandb logger.  # Initialisiert den Wandb-Logger.
        resume="allow",  # Resumes the experiment if already started.  # Setzt das Experiment fort, falls es bereits gestartet wurde.
        project="TEDD1104_reorder",  # Name of the project on Wandb.  # Name des Projekts auf Wandb.
        save_dir=output_dir,  # Directory where logs will be saved.  # Verzeichnis, in dem die Logs gespeichert werden.
    )
else:  # If neither TensorBoard nor Wandb is selected.  # Falls weder TensorBoard noch Wandb ausgewählt ist.
    raise ValueError(  # Raises an error if an unknown logger is specified.  # Wirft einen Fehler, wenn ein unbekannter Logger angegeben wird.
        f"Unknown logger: {report_to}. Please use 'tensorboard' or 'wandb'."  # Error message specifying allowed loggers.  # Fehlermeldung mit den erlaubten Loggern.
    )

lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")  # Initializes a learning rate monitor to track the learning rate during training.  # Initialisiert einen Learning Rate Monitor, um die Lernrate während des Trainings zu verfolgen.

checkpoint_callback = pl.callbacks.ModelCheckpoint(  # Initializes the checkpoint callback to save the model during training.  # Initialisiert den Checkpoint-Callback, um das Modell während des Trainings zu speichern.
    dirpath=output_dir,  # Directory to save the model checkpoints.  # Verzeichnis, um die Modell-Checkpoints zu speichern.
    monitor="Validation/acc",  # Metric to monitor during training for checkpoint saving.  # Metrik, die während des Trainings zur Überwachung für das Speichern von Checkpoints verwendet wird.
    mode="max",  # Saves the model when the monitored metric reaches its maximum.  # Speichert das Modell, wenn die überwachte Metrik ihr Maximum erreicht.
    save_last=True  # Saves the last checkpoint even if it doesn't improve the monitored metric.  # Speichert den letzten Checkpoint, auch wenn er die überwachte Metrik nicht verbessert.
)

checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"  # Sets the name format for the last checkpoint file.  # Legt das Namensformat für die letzte Checkpoint-Datei fest.

model.accelerator = accelerator  # Sets the accelerator for the model (e.g., GPU or TPU).  # Setzt den Accelerator für das Modell (z.B. GPU oder TPU).



trainer = pl.Trainer(  # Initialize the PyTorch Lightning trainer.  # Initialisiere den PyTorch Lightning Trainer.
    devices=devices,  # Specifies which devices (e.g., GPU/CPU) to use for training.  # Gibt an, welche Geräte (z. B. GPU/CPU) für das Training verwendet werden sollen.
    accelerator=accelerator,  # Defines the accelerator type (e.g., GPU, TPU, etc.) for the training process.  # Definiert den Beschleuniger (z. B. GPU, TPU usw.) für den Trainingsprozess.
    precision=precision if precision == "bf16" else int(precision),  # Sets the precision for training (e.g., float16, float32).  # Legt die Präzision für das Training fest (z. B. float16, float32).
    strategy=strategy,  # Defines the strategy for distributed training (e.g., DataParallel, DDP).  # Definiert die Strategie für das verteilte Training (z. B. DataParallel, DDP).
    val_check_interval=val_check_interval,  # Specifies how often to run validation during training.  # Gibt an, wie oft während des Trainings eine Validierung durchgeführt werden soll.
    accumulate_grad_batches=accumulation_steps,  # Number of batches to accumulate gradients before updating the model.  # Anzahl der Batches, bei denen Gradienten akkumuliert werden, bevor das Modell aktualisiert wird.
    max_epochs=max_epochs,  # Sets the maximum number of epochs for training.  # Legt die maximale Anzahl an Epochen für das Training fest.
    logger=logger,  # Defines the logger to track training progress and metrics.  # Definiert den Logger, um den Trainingsfortschritt und Metriken zu verfolgen.
    callbacks=[  # List of callbacks to monitor and adjust training.  # Liste der Rückrufe (Callbacks), um das Training zu überwachen und anzupassen.
        # pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),  # (Optional) Callback for stochastic weight averaging.  # (Optional) Rückruf für stochastische Gewichtsmittelung.
        checkpoint_callback,  # Callback to save the best model checkpoint during training.  # Rückruf zum Speichern des besten Modell-Checkpoints während des Trainings.
        lr_monitor,  # Callback to monitor the learning rate during training.  # Rückruf zum Überwachen der Lernrate während des Trainings.
    ],
    gradient_clip_val=1.0,  # Clips gradients to avoid exploding gradients during training.  # Schneidet Gradienten, um explodierende Gradienten während des Trainings zu vermeiden.
    log_every_n_steps=100,  # Logs training progress every 100 steps.  # Protokolliert den Trainingsfortschritt alle 100 Schritte.
)

trainer.fit(  # Starts the training process.  # Beginnt den Trainingsprozess.
    ckpt_path=checkpoint_path,  # Specifies the checkpoint to resume from, if any.  # Gibt den Checkpoint an, von dem aus das Training fortgesetzt werden soll, falls vorhanden.
    model=model,  # The model to be trained.  # Das Modell, das trainiert werden soll.
    datamodule=data,  # The data module that provides the training and validation data.  # Das Datenmodul, das Trainings- und Validierungsdaten bereitstellt.
)

# print(f"Best model path: {checkpoint_callback.best_model_path}")  # (Optional) Prints the path to the best model checkpoint found.  # (Optional) Gibt den Pfad zum besten gefundenen Modell-Checkpoint aus.

if test_dir:  # If a test directory is provided, run the test phase.  # Wenn ein Testverzeichnis angegeben ist, wird die Testphase ausgeführt.
    trainer.test(datamodule=data, ckpt_path="best")  # Run testing on the best model.  # Führen Sie den Test mit dem besten Modell durch.



if __name__ == "__main__":  # Check if the script is being run directly, not imported as a module.  # Prüfen, ob das Skript direkt ausgeführt wird und nicht als Modul importiert wird.

    parser = argparse.ArgumentParser()  # Create an argument parser object to handle command-line arguments.  # Erstelle ein Argument-Parser-Objekt, um Kommandozeilenargumente zu verarbeiten.

    group = parser.add_mutually_exclusive_group(required=True)  # Create a mutually exclusive group of arguments, meaning only one can be chosen.  # Erstelle eine Gruppe von sich gegenseitig ausschließenden Argumenten, wobei nur eines ausgewählt werden kann.

    group.add_argument(  # Add an argument to the mutually exclusive group for training a new model.  # Füge ein Argument zur exklusiven Gruppe hinzu, um ein neues Modell zu trainieren.
        "--train_new",  # Command-line flag to start training a new model.  # Kommandozeilen-Flag zum Starten des Trainings eines neuen Modells.
        action="store_true",  # When the flag is specified, it stores a boolean value `True`.  # Wenn das Flag gesetzt ist, wird der boolesche Wert `True` gespeichert.
        help="Train a new model",  # Help message for this argument, explaining its purpose.  # Hilfsnachricht für dieses Argument, das den Zweck erklärt.
    )

    group.add_argument(  # Add another argument to the mutually exclusive group to continue training from a checkpoint.  # Füge ein weiteres Argument zur exklusiven Gruppe hinzu, um das Training von einem Checkpoint fortzusetzen.
        "--continue_training",  # Command-line flag to continue training from an existing checkpoint.  # Kommandozeilen-Flag, um das Training von einem bestehenden Checkpoint fortzusetzen.
        action="store_true",  # When the flag is specified, it stores a boolean value `True`.  # Wenn das Flag gesetzt ist, wird der boolesche Wert `True` gespeichert.
        help="Continues training a model from a checkpoint.",  # Help message explaining the purpose of continuing training.  # Hilfsnachricht, die den Zweck des Fortsetzens des Trainings erklärt.
    )

    parser.add_argument(  # Add an argument for specifying the directory containing the training data.  # Füge ein Argument hinzu, um das Verzeichnis mit den Trainingsdaten anzugeben.
        "--train_dir",  # Command-line argument for the training data directory.  # Kommandozeilen-Argument für das Trainingsdaten-Verzeichnis.
        type=str,  # Argument expects a string type value.  # Das Argument erwartet einen Wert vom Typ String.
        required=True,  # This argument is mandatory.  # Dieses Argument ist erforderlich.
        help="The directory containing the training data.",  # Help message explaining what this argument is for.  # Hilfsnachricht, die erklärt, wofür dieses Argument dient.
    )

    parser.add_argument(  # Add an argument for specifying the directory containing the validation data.  # Füge ein Argument hinzu, um das Verzeichnis mit den Validierungsdaten anzugeben.
        "--val_dir",  # Command-line argument for the validation data directory.  # Kommandozeilen-Argument für das Validierungsdaten-Verzeichnis.
        type=str,  # Argument expects a string type value.  # Das Argument erwartet einen Wert vom Typ String.
        required=True,  # This argument is mandatory.  # Dieses Argument ist erforderlich.
        help="The directory containing the validation data.",  # Help message explaining the purpose of this argument.  # Hilfsnachricht, die den Zweck dieses Arguments erklärt.
    )

    parser.add_argument(  # Add an argument for specifying the directory containing the test data.  # Füge ein Argument hinzu, um das Verzeichnis mit den Testdaten anzugeben.
        "--test_dir",  # Command-line argument for the test data directory.  # Kommandozeilen-Argument für das Testdaten-Verzeichnis.
        type=str,  # Argument expects a string type value.  # Das Argument erwartet einen Wert vom Typ String.
        default=None,  # This argument is optional, with a default value of None.  # Dieses Argument ist optional und hat einen Standardwert von None.
        help="The directory containing the test data.",  # Help message explaining what this argument is for.  # Hilfsnachricht, die erklärt, wofür dieses Argument dient.
    )

    parser.add_argument(  # Add an argument to specify where to save the model.  # Füge ein Argument hinzu, um anzugeben, wo das Modell gespeichert werden soll.
        "--output_dir",  # Command-line argument for the output directory where the model will be saved.  # Kommandozeilen-Argument für das Ausgabeverzeichnis, in dem das Modell gespeichert wird.
        type=str,  # Argument expects a string type value.  # Das Argument erwartet einen Wert vom Typ String.
        required=True,  # This argument is mandatory.  # Dieses Argument ist erforderlich.
        help="The directory to save the model to.",  # Help message explaining the purpose of this argument.  # Hilfsnachricht, die erklärt, wofür dieses Argument dient.
    )

    parser.add_argument(  # Add an argument for batch size, used during training and evaluation.  # Füge ein Argument für die Batch-Größe hinzu, die während des Trainings und der Auswertung verwendet wird.
        "--batch_size",  # Command-line argument for the batch size.  # Kommandozeilen-Argument für die Batch-Größe.
        type=int,  # Argument expects an integer type value.  # Das Argument erwartet einen Wert vom Typ Integer.
        required=True,  # This argument is mandatory.  # Dieses Argument ist erforderlich.
        help="The batch size for training and eval.",  # Help message explaining the purpose of this argument.  # Hilfsnachricht, die den Zweck dieses Arguments erklärt.
    )

parser.add_argument(  # Adding a new argument to the argument parser.  # Fügt dem Argumentparser ein neues Argument hinzu.
    "--accumulation_steps",  # The name of the argument.  # Der Name des Arguments.
    type=int,  # The type of the argument, here it's an integer.  # Der Typ des Arguments, hier eine Ganzzahl.
    default=1,  # Default value of 1 if not provided by the user.  # Standardwert von 1, wenn der Benutzer nichts angibt.
    help="The number of steps to accumulate gradients.",  # Description of what the argument does.  # Beschreibung dessen, was das Argument tut.
)

parser.add_argument(  # Adding another argument to the parser.  # Fügt dem Argumentparser ein weiteres Argument hinzu.
    "--max_epochs",  # The name of the argument.  # Der Name des Arguments.
    type=int,  # The argument type, an integer.  # Der Argumenttyp, eine Ganzzahl.
    required=True,  # Marks this argument as mandatory.  # Kennzeichnet dieses Argument als erforderlich.
    help="The maximum number of epochs to train for.",  # Description of the argument.  # Beschreibung des Arguments.
)

parser.add_argument(  # Adding another argument for dataloader workers.  # Fügt ein weiteres Argument für Dataloader-Worker hinzu.
    "--dataloader_num_workers",  # The name of the argument.  # Der Name des Arguments.
    type=int,  # Integer type for the argument.  # Ganzzahltyp für das Argument.
    default=os.cpu_count(),  # Default value set to the number of CPU cores available.  # Standardwert auf die Anzahl der verfügbaren CPU-Kerne gesetzt.
    help="Number of CPU workers for the Data Loaders",  # Description of what the argument does.  # Beschreibung dessen, was das Argument tut.
)

parser.add_argument(  # Adding an argument for hiding the minimap probability.  # Fügt ein Argument für die Wahrscheinlichkeit des Verbergens der Minimap hinzu.
    "--hide_map_prob",  # The name of the argument.  # Der Name des Arguments.
    type=float,  # The argument type is float.  # Der Argumenttyp ist eine Fließkommazahl.
    default=1.0,  # Default value of 1.0.  # Standardwert von 1.0.
    help="Probability of hiding the minimap in the sequence (0<=hide_map_prob<=1)",  # Explanation of the range for the value.  # Erklärung des Wertebereichs für das Argument.
)

parser.add_argument(  # Adding an argument for dropout probability for images.  # Fügt ein Argument für die Dropout-Wahrscheinlichkeit von Bildern hinzu.
    "--dropout_images_prob",  # The name of the argument.  # Der Name des Arguments.
    type=float,  # The argument type is float.  # Der Argumenttyp ist eine Fließkommazahl.
    nargs=5,  # The argument expects 5 values.  # Das Argument erwartet 5 Werte.
    default=[0.0, 0.0, 0.0, 0.0, 0.0],  # Default value is a list of 5 values of 0.0.  # Der Standardwert ist eine Liste mit 5 Werten von 0.0.
    help="Probability of dropping each image in the sequence (0<=dropout_images_prob<=1)",  # Explanation of the range for the value.  # Erklärung des Wertebereichs für das Argument.
)

parser.add_argument(  # Adding an argument for validation check interval.  # Fügt ein Argument für das Validierungsprüfintervall hinzu.
    "--val_check_interval",  # The name of the argument.  # Der Name des Arguments.
    type=float,  # Argument type is float.  # Der Argumenttyp ist eine Fließkommazahl.
    default=1.0,  # Default value is 1.0.  # Der Standardwert ist 1.0.
    help="The interval in epochs between validation checks.",  # Description of the argument's function.  # Beschreibung der Funktion des Arguments.
)

parser.add_argument(  # Adding an argument for the learning rate.  # Fügt ein Argument für die Lernrate hinzu.
    "--learning_rate",  # The name of the argument.  # Der Name des Arguments.
    type=float,  # The argument type is float.  # Der Argumenttyp ist eine Fließkommazahl.
    default=3e-5,  # Default value for the learning rate.  # Standardwert für die Lernrate.
    help="[NEW MODEL] The learning rate for the optimizer.",  # Explanation of the argument.  # Erklärung des Arguments.
)

parser.add_argument(  # Adding an argument for weight decay.  # Fügt ein Argument für den Gewichtszusatz hinzu.
    "--weight_decay",  # The name of the argument.  # Der Name des Arguments.
    type=float,  # The argument type is float.  # Der Argumenttyp ist eine Fließkommazahl.
    default=1e-4,  # Default value for weight decay.  # Standardwert für Gewichtszusatz.
    help="[NEW MODEL]] AdamW Weight Decay",  # Description of the weight decay method.  # Beschreibung der Gewichtszusatzmethode.
)

parser.add_argument(  # Adding an argument for optimizer choice.  # Fügt ein Argument für die Wahl des Optimierers hinzu.
    "--optimizer_name",  # The name of the argument.  # Der Name des Arguments.
    type=str,  # The argument type is a string.  # Der Argumenttyp ist eine Zeichenkette.
    default="adamw",  # Default optimizer is "adamw".  # Der Standard-Optimizer ist "adamw".
    choices=["adamw", "adafactor"],  # The user can choose between adamw and adafactor optimizers.  # Der Benutzer kann zwischen den Optimierern "adamw" und "adafactor" wählen.
    help="[NEW MODEL] The optimizer to use: adamw or adafactor. Adafactor requires fairseq to be installed. "
    "pip install fairseq",  # Description of the optimizers and installation instructions for adafactor.  # Beschreibung der Optimierer und Installationsanweisungen für Adafactor.
)


parser.add_argument(
    "--scheduler_name",  # The argument for specifying the scheduler name.  # Der Parameter zur Angabe des Namens des Schedulers.
    type=str,  # Specifies the type of the input as a string.  # Gibt den Typ der Eingabe als String an.
    default="linear",  # Default value is "linear" if no value is provided.  # Der Standardwert ist "linear", falls kein Wert angegeben wird.
    choices=["linear", "plateau"],  # The available options for the scheduler are "linear" or "plateau".  # Die verfügbaren Optionen für den Scheduler sind "linear" oder "plateau".
    help="[NEW MODEL] The scheduler to use: linear or plateau.",  # Help description for the argument.  # Hilfebeschreibung für das Argument.
)

parser.add_argument(
    "--warmup_factor",  # The argument for specifying the warmup factor.  # Der Parameter zur Angabe des Warmup-Faktors.
    type=float,  # Specifies the type of the input as a float.  # Gibt den Typ der Eingabe als Float an.
    default=0.05,  # Default value is 0.05 for the warmup factor.  # Der Standardwert für den Warmup-Faktor ist 0,05.
    help="[NEW MODEL] Percentage of the total training steps that we will use for the warmup (0<=warmup_factor<=1)",  # Help description for the warmup factor.  # Hilfebeschreibung für den Warmup-Faktor.
)

parser.add_argument(
    "--cnn_model_name",  # The argument for specifying the CNN model name.  # Der Parameter zur Angabe des CNN-Modellnamens.
    type=str,  # Specifies the type of the input as a string.  # Gibt den Typ der Eingabe als String an.
    default="efficientnet_b4",  # Default value is "efficientnet_b4".  # Der Standardwert ist "efficientnet_b4".
    help="[NEW MODEL] CNN model name from torchvision models, see https://pytorch.org/vision/stable/models.html "
    "for a list of available models.",  # Help description for the CNN model name.  # Hilfebeschreibung für den CNN-Modellnamen.
)

parser.add_argument(
    "--do_not_load_pretrained_cnn",  # The argument to control whether to load pretrained CNN weights.  # Der Parameter zur Steuerung, ob vortrainierte CNN-Gewichte geladen werden sollen.
    action="store_true",  # A flag to indicate the action should be taken when specified.  # Ein Schalter, der angibt, dass die Aktion beim Angeben ausgeführt werden soll.
    help="[NEW MODEL] Do not load the pretrained weights for the cnn model",  # Help description for the flag.  # Hilfebeschreibung für den Schalter.
)

parser.add_argument(
    "--embedded_size",  # The argument for specifying the embedding size.  # Der Parameter zur Angabe der Einbettungsgröße.
    type=int,  # Specifies the type of the input as an integer.  # Gibt den Typ der Eingabe als Ganzzahl an.
    default=512,  # Default value is 512 for the embedding size.  # Der Standardwert für die Einbettungsgröße ist 512.
    help="[NEW MODEL] The size of the embedding for the encoder.",  # Help description for the embedding size.  # Hilfebeschreibung für die Einbettungsgröße.
)

parser.add_argument(
    "--nhead",  # The argument for specifying the number of heads in the multi-head attention.  # Der Parameter zur Angabe der Anzahl der Köpfe in der Multi-Head-Attention.
    type=int,  # Specifies the type of the input as an integer.  # Gibt den Typ der Eingabe als Ganzzahl an.
    default=8,  # Default value is 8 for the number of heads.  # Der Standardwert für die Anzahl der Köpfe ist 8.
    help="[NEW MODEL Transformers] Number of heads in the multi-head attention",  # Help description for the number of heads.  # Hilfebeschreibung für die Anzahl der Köpfe.
)

parser.add_argument(
    "--num_layers_encoder",  # The argument for specifying the number of transformer layers in the encoder.  # Der Parameter zur Angabe der Anzahl der Transformer-Schichten im Encoder.
    type=int,  # Specifies the type of the input as an integer.  # Gibt den Typ der Eingabe als Ganzzahl an.
    default=4,  # Default value is 4 for the number of layers.  # Der Standardwert für die Anzahl der Schichten ist 4.
    help="[NEW MODEL] Number of transformer layers in the encoder",  # Help description for the number of transformer layers.  # Hilfebeschreibung für die Anzahl der Transformer-Schichten.
)

parser.add_argument(
    "--dropout_cnn_out",  # The argument for specifying the dropout rate for the output of the CNN.  # Der Parameter zur Angabe der Dropout-Rate für die Ausgabe des CNN.
    type=float,  # Specifies the type of the input as a float.  # Gibt den Typ der Eingabe als Float an.
    default=0.3,  # Default value is 0.3 for the dropout rate.  # Der Standardwert für die Dropout-Rate ist 0,3.
    help="[NEW MODEL] Dropout rate for the output of the CNN",  # Help description for the dropout rate.  # Hilfebeschreibung für die Dropout-Rate.
)



parser.add_argument(
    "--positional_embeddings_dropout",  # Argument name for dropout in positional embeddings.  # Name des Arguments für Dropout in den Positions-Embeddings.
    type=float,  # Specifies the argument type is a floating-point number.  # Gibt an, dass der Argumenttyp eine Fließkommazahl ist.
    default=0.1,  # Default value for this argument, set to 0.1.  # Der Standardwert für dieses Argument ist auf 0,1 gesetzt.
    help="[NEW MODEL Transformer] Dropout rate for the positional embeddings",  # Help message explaining the purpose of this argument.  # Hilfsnachricht, die den Zweck dieses Arguments erklärt.
)

parser.add_argument(
    "--dropout_encoder",  # Argument name for dropout in the encoder.  # Name des Arguments für Dropout im Encoder.
    type=float,  # Argument is a floating-point number.  # Das Argument ist eine Fließkommazahl.
    default=0.1,  # Default dropout rate for the encoder set to 0.1.  # Standard-Rate für Dropout im Encoder ist auf 0,1 gesetzt.
    help="[NEW MODEL] Dropout rate for the encoder",  # Help message explaining the dropout rate in the encoder.  # Hilfsnachricht zur Erklärung der Dropout-Rate im Encoder.
)

parser.add_argument(
    "--dropout_encoder_features",  # Argument for dropout rate of encoder output features.  # Argument für die Dropout-Rate der Encoder-Ausgabefeatures.
    type=float,  # Argument is a floating-point number.  # Das Argument ist eine Fließkommazahl.
    default=0.3,  # Default dropout rate for encoder features is set to 0.3.  # Der Standardwert für Dropout der Encoder-Features ist auf 0,3 gesetzt.
    help="[NEW MODEL] Dropout probability of the encoder output",  # Help message explaining the dropout rate for the encoder output.  # Hilfsnachricht, die die Dropout-Rate für die Encoder-Ausgabe erklärt.
)

parser.add_argument(
    "--mask_prob",  # Argument name for the probability of masking input vectors.  # Name des Arguments für die Wahrscheinlichkeit des Maskierens von Eingabvektoren.
    type=float,  # Specifies the argument type as a floating-point number.  # Gibt an, dass der Argumenttyp eine Fließkommazahl ist.
    default=0.2,  # Default masking probability is set to 0.2.  # Die Standardwahrscheinlichkeit für das Maskieren ist auf 0,2 gesetzt.
    help="[NEW MODEL Transformers] Probability of masking each input vector in the transformer encoder",  # Help message describing the masking probability for input vectors.  # Hilfsnachricht, die die Maskierungswahrscheinlichkeit für Eingabvektoren erklärt.
)

parser.add_argument(
    "--sequence_size",  # Argument for defining the length of the input sequence.  # Argument zur Festlegung der Länge der Eingabesequenz.
    type=int,  # Specifies the argument type as an integer.  # Gibt an, dass der Argumenttyp eine ganze Zahl ist.
    default=5,  # Default sequence length is set to 5.  # Die Standardsequenzlänge ist auf 5 gesetzt.
    help="[NEW MODEL] Length of the input sequence. Placeholder for the future, only 5 supported",  # Help message explaining the sequence length.  # Hilfsnachricht, die die Sequenzlänge erklärt. Platzhalter für die Zukunft, derzeit nur 5 unterstützt.
)

parser.add_argument(
    "--checkpoint_path",  # Argument for specifying the checkpoint path.  # Argument zur Angabe des Checkpoint-Pfads.
    type=str,  # Specifies the argument type as a string.  # Gibt an, dass der Argumenttyp ein String ist.
    default=None,  # Default is set to None, meaning no checkpoint path.  # Der Standardwert ist None, was bedeutet, dass kein Checkpoint-Pfad angegeben ist.
    help="If new_model is True, the path to the checkpoint to a pretrained model in the image reordering task. "
    "If continue_training is True, the path to the checkpoint to continue training from.",  # Help message explaining the checkpoint path.  # Hilfsnachricht zur Erklärung des Checkpoint-Pfads.
)

parser.add_argument(
    "--devices",  # Argument for specifying the number of devices (GPUs/TPUs).  # Argument zur Angabe der Anzahl der Geräte (GPUs/TPUs).
    type=int,  # Specifies the argument type as an integer.  # Gibt an, dass der Argumenttyp eine ganze Zahl ist.
    default=1,  # Default number of devices is set to 1.  # Die Standardanzahl der Geräte ist auf 1 gesetzt.
    help="Number of GPUs/TPUs to use. ",  # Help message describing how many GPUs or TPUs to use.  # Hilfsnachricht zur Angabe der Anzahl der zu verwendenden GPUs oder TPUs.
)

parser.add_argument(
    "--accelerator",  # Argument for specifying the accelerator type.  # Argument zur Angabe des Accelerator-Typs.
    type=str,  # Specifies the argument type as a string.  # Gibt an, dass der Argumenttyp ein String ist.
    default="auto",  # Default accelerator type is set to "auto".  # Der Standardwert für den Accelerator ist auf "auto" gesetzt.
    choices=["auto", "tpu", "gpu", "cpu", "ipu"],  # The argument can be one of these choices: auto, tpu, gpu, cpu, or ipu.  # Das Argument kann einer dieser Werte sein: auto, tpu, gpu, cpu oder ipu.
    help="Accelerator to use. If 'auto', tries to automatically detect TPU, GPU, CPU or IPU system",  # Help message describing the available accelerators.  # Hilfsnachricht, die die verfügbaren Accelerator-Typen beschreibt.
)

parser.add_argument(
    "--precision",  # Argument for setting the precision of computations.  # Argument zur Festlegung der Genauigkeit der Berechnungen.
    type=str,  # Specifies that the input type is a string.  # Gibt an, dass der Eingabetyp ein String ist.
    default="16",  # Default value for precision is set to 16-bit.  # Der Standardwert für die Genauigkeit ist auf 16-Bit festgelegt.
    choices=["bf16", "16", "32", "64"],  # Defines possible precision values (bfloat16, 16-bit, 32-bit, 64-bit).  # Definiert mögliche Genauigkeitswerte (bfloat16, 16-Bit, 32-Bit, 64-Bit).
    help=" Double precision (64), full precision (32), "  # Help description for the precision argument.  # Hilfebenachrichtigung für das Genauigkeitsargument.
    "half precision (16) or bfloat16 precision (bf16). "  # More detail about precision choices.  # Weitere Details zu den Genauigkeitsauswahlmöglichkeiten.
    "Can be used on CPU, GPU or TPUs.",  # Specifies where the precision can be applied.  # Gibt an, wo die Genauigkeit angewendet werden kann (CPU, GPU, TPU).
)

parser.add_argument(
    "--strategy",  # Argument to specify the training strategy.  # Argument zur Festlegung der Trainingsstrategie.
    type=str,  # Specifies that the input type is a string.  # Gibt an, dass der Eingabetyp ein String ist.
    default=None,  # Default value is None if not provided.  # Der Standardwert ist None, wenn nichts angegeben wird.
    help="Supports passing different training strategies with aliases (ddp, ddp_spawn, etc)",  # Help description for training strategy argument.  # Hilfebenachrichtigung für das Trainingsstrategieargument.
)

parser.add_argument(
    "--report_to",  # Argument to specify the reporting tool.  # Argument zur Festlegung des Reporting-Tools.
    type=str,  # Specifies that the input type is a string.  # Gibt an, dass der Eingabetyp ein String ist.
    default="wandb",  # Default reporting tool is set to wandb.  # Das Standard-Reporting-Tool ist auf wandb festgelegt.
    choices=["wandb", "tensorboard"],  # Defines the valid reporting tools (wandb or tensorboard).  # Definiert die gültigen Reporting-Tools (wandb oder tensorboard).
    help="Report to wandb or tensorboard",  # Help description for the report_to argument.  # Hilfebenachrichtigung für das report_to-Argument.
)

parser.add_argument(
    "--find_lr",  # Argument to enable learning rate finder.  # Argument zur Aktivierung des Lernratenfinders.
    action="store_true",  # If the flag is set, the learning rate finder will be used.  # Wenn das Flag gesetzt ist, wird der Lernratenfinder verwendet.
    help="Find the optimal learning rate for the model. We will use Pytorch Lightning's find_lr function. "  # Help description for the find_lr argument.  # Hilfebenachrichtigung für das find_lr-Argument.
    "See: "  # Provides a link for further reading on learning rate finder.  # Bietet einen Link für weitere Informationen zum Lernratenfinder.
    "https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#learning-rate-finder",  # URL for further documentation on learning rate finder.  # URL für weiterführende Dokumentation zum Lernratenfinder.
)

args = parser.parse_args()  # Parses the command line arguments and stores them in 'args'.  # Parsen der Befehlszeilenargumente und Speichern in 'args'.

if args.train_new:  # If the argument 'train_new' is set to True, start training a new model.  # Wenn das Argument 'train_new' auf True gesetzt ist, beginnt das Training eines neuen Modells.
    train_new_model(  # Calls the function to train a new model.  # Ruft die Funktion auf, um ein neues Modell zu trainieren.
        train_dir=args.train_dir,  # Directory where training data is located.  # Verzeichnis, in dem die Trainingsdaten gespeichert sind.
        val_dir=args.val_dir,  # Directory where validation data is located.  # Verzeichnis, in dem die Validierungsdaten gespeichert sind.
        test_dir=args.test_dir,  # Directory where test data is located.  # Verzeichnis, in dem die Testdaten gespeichert sind.
        output_dir=args.output_dir,  # Directory where output should be saved.  # Verzeichnis, in dem Ausgaben gespeichert werden.
        batch_size=args.batch_size,  # Batch size for training.  # Batch-Größe für das Training.
        max_epochs=args.max_epochs,  # Maximum number of epochs for training.  # Maximale Anzahl an Epochen für das Training.
        cnn_model_name=args.cnn_model_name,  # Name of the CNN model to use.  # Name des zu verwendenden CNN-Modells.
        accumulation_steps=args.accumulation_steps,  # Number of steps for gradient accumulation.  # Anzahl der Schritte für die Gradientenakkumulation.
        hide_map_prob=args.hide_map_prob,  # Probability to hide parts of the map.  # Wahrscheinlichkeit, Teile der Karte zu verbergen.
        dropout_images_prob=args.dropout_images_prob,  # Probability of dropout for image data.  # Wahrscheinlichkeit für Dropout von Bilddaten.
        val_check_interval=args.val_check_interval,  # Interval for validation checks during training.  # Intervall für Validierungsprüfungen während des Trainings.
        dataloader_num_workers=args.dataloader_num_workers,  # Number of workers for data loading.  # Anzahl der Arbeiter für das Laden von Daten.
        pretrained_cnn=not args.do_not_load_pretrained_cnn,  # Whether to use a pretrained CNN model or not.  # Ob ein vortrainiertes CNN-Modell verwendet werden soll oder nicht.
        embedded_size=args.embedded_size,  # Size of the embedding layer.  # Größe der Einbettungsschicht.
        nhead=args.nhead,  # Number of heads in the attention mechanism.  # Anzahl der Köpfe im Aufmerksamkeitsmechanismus.
        num_layers_encoder=args.num_layers_encoder,  # Number of layers in the encoder.  # Anzahl der Schichten im Encoder.
        dropout_cnn_out=args.dropout_cnn_out,  # Dropout rate for CNN output.  # Dropout-Rate für die CNN-Ausgabe.
        dropout_encoder_features=args.dropout_encoder_features,  # Dropout rate for encoder features.  # Dropout-Rate für Encoder-Features.
        positional_embeddings_dropout=args.positional_embeddings_dropout,  # Dropout rate for positional embeddings.  # Dropout-Rate für Positions-Embeddings.
        dropout_encoder=args.dropout_encoder,  # Dropout rate for the encoder.  # Dropout-Rate für den Encoder.
        mask_prob=args.mask_prob,  # Probability of masking inputs during training.  # Wahrscheinlichkeit für das Maskieren von Eingaben während des Trainings.
        sequence_size=args.sequence_size,  # Size of the input sequence.  # Größe der Eingabesequenz.
        devices=args.devices,  # Devices to use for training (CPU/GPU/TPU).  # Geräte, die für das Training verwendet werden (CPU/GPU/TPU).
        accelerator=args.accelerator,  # Accelerator to use (e.g., GPU, TPU).  # Beschleuniger, der verwendet werden soll (z.B. GPU, TPU).
        precision=args.precision,  # Precision for the model's computations.  # Genauigkeit für die Berechnungen des Modells.
        strategy=args.strategy,  # Training strategy to use.  # Trainingsstrategie, die verwendet werden soll.
        report_to=args.report_to,  # Reporting tool (e.g., wandb, tensorboard).  # Reporting-Tool (z.B. wandb, tensorboard).
        find_lr=args.find_lr,  # Whether to find the optimal learning rate.  # Ob die optimale Lernrate gefunden werden soll.
        learning_rate=args.learning_rate,  # Learning rate for training.  # Lernrate für das Training.
        weight_decay=args.weight_decay,  # Weight decay regularization.  # Gewichtszusatz zur Regularisierung.
        optimizer_name=args.optimizer_name,  # Optimizer to use for training.  # Optimierer, der für das Training verwendet wird.
        scheduler_name=args.scheduler_name,  # Scheduler to use for learning rate scheduling.  # Scheduler, der für die Lernratenplanung verwendet wird.
        warmup_factor=args.warmup_factor,  # Warmup factor for learning rate.  # Aufwärmfaktor für die Lernrate.
    )

else:  # If 'train_new' is not set, continue training from checkpoint.  # Wenn 'train_new' nicht gesetzt ist, wird das Training aus dem Checkpoint fortgesetzt.
    continue_training(  # Calls the function to continue training.  # Ruft die Funktion auf, um das Training fortzusetzen.
        checkpoint_path=args.checkpoint_path,  # Path to the model checkpoint.  # Pfad zum Modell-Checkpoint.
        hparams_path=args.hparams_path,  # Path to the hyperparameters file.  # Pfad zur Hyperparameters-Datei.
        train_dir=args.train_dir,  # Directory where training data is located.  # Verzeichnis, in dem die Trainingsdaten gespeichert sind.
        val_dir=args.val_dir,  # Directory where validation data is located.  # Verzeichnis, in dem die Validierungsdaten gespeichert sind.
        test_dir=args.test_dir,  # Directory where test data is located.  # Verzeichnis, in dem die Testdaten gespeichert sind.
        output_dir=args.output_dir,  # Directory where output should be saved.  # Verzeichnis, in dem Ausgaben gespeichert werden.
        batch_size=args.batch_size,  # Batch size for training.  # Batch-Größe für das Training.
        accumulation_steps=args.accumulation_steps,  # Number of steps for gradient accumulation.  # Anzahl der Schritte für die Gradientenakkumulation.
        max_epochs=args.max_epochs,  # Maximum number of epochs for training.  # Maximale Anzahl an Epochen für das Training.
        mask_prob=args.mask_prob,  # Probability of masking inputs during training.  # Wahrscheinlichkeit für das Maskieren von Eingaben während des Trainings.
        hide_map_prob=args.hide_map_prob,  # Probability to hide parts of the map.  # Wahrscheinlichkeit, Teile der Karte zu verbergen.
        dropout_images_prob=args.dropout_images_prob,  # Probability of dropout for image data.  # Wahrscheinlichkeit für Dropout von Bilddaten.
        dataloader_num_workers=args.dataloader_num_workers,  # Number of workers for data loading.  # Anzahl der Arbeiter für das Laden von Daten.
        devices=args.devices,  # Devices to use for training (CPU/GPU/TPU).  # Geräte, die für das Training verwendet werden (CPU/GPU/TPU).
        accelerator=args.accelerator,  # Accelerator to use (e.g., GPU, TPU).  # Beschleuniger, der verwendet werden soll (z.B. GPU, TPU).
        precision=args.precision,  # Precision for the model's computations.  # Genauigkeit für die Berechnungen des Modells.
        strategy=args.strategy,  # Training strategy to use.  # Trainingsstrategie, die verwendet werden soll.
        report_to=args.report_to,  # Reporting tool (e.g., wandb, tensorboard).  # Reporting-Tool (z.B. wandb, tensorboard).
    )
