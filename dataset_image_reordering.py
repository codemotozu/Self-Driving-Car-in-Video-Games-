from __future__ import print_function, division  # Ensures compatibility with Python 2 for print and division (future behavior in Python 3) / Sichert die Kompatibilität mit Python 2 für print und Division (zukünftiges Verhalten in Python 3)
import os  # Imports the os module for interacting with the operating system / Importiert das os-Modul für die Interaktion mit dem Betriebssystem
import torch  # Imports PyTorch library for tensor operations / Importiert die PyTorch-Bibliothek für Tensor-Operationen
import torchvision  # Imports torchvision for computer vision tasks like image handling / Importiert torchvision für Computer-Vision-Aufgaben wie Bildverarbeitung
from torch.utils.data import Dataset, DataLoader  # Imports Dataset and DataLoader classes for handling datasets and loading them in batches / Importiert die Klassen Dataset und DataLoader für die Handhabung von Datensätzen und das Laden in Batches
from torchvision import transforms  # Imports image transformations from torchvision for data preprocessing / Importiert Bildtransformationen aus torchvision für die Datenvorverarbeitung
import glob  # Imports glob for file name pattern matching / Importiert glob für die Dateimustererkennung
from typing import List, Optional, Dict  # Imports typing utilities for type hints / Importiert Typ-Hinweise für List, Optional und Dict
from utils import IOHandler, get_mask  # Imports utility functions for IO handling and mask generation / Importiert Hilfsfunktionen für IO-Handling und Masken-Generierung
import pytorch_lightning as pl  # Imports PyTorch Lightning for simplifying deep learning model training / Importiert PyTorch Lightning zum Vereinfachen des Trainings von Deep-Learning-Modellen
from dataset import (  # Imports dataset-related functions / Importiert datensatzbezogene Funktionen
    RemoveMinimap,  # Removes minimap from images / Entfernt die Mini-Karten aus Bildern
    RemoveImage,  # Removes specific images from a sequence / Entfernt spezifische Bilder aus einer Sequenz
    SplitImages,  # Splits image sequence into separate images / Teilt die Bildsequenz in separate Bilder
    Normalize,  # Normalizes image data / Normalisiert die Bilddaten
    SequenceColorJitter,  # Applies random color jitter to the sequence / Wendet zufällige Farbänderungen auf die Sequenz an
    collate_fn,  # Defines how to combine samples in a batch / Definiert, wie Proben in einem Batch kombiniert werden
    set_worker_sharing_strategy,  # Sets worker strategy for multi-worker loading / Legt die Worker-Strategie für das Laden mit mehreren Arbeitern fest
)

class ReOrderImages(object):  # Defines a class for reordering images based on a tensor of positions / Definiert eine Klasse zum Neuanordnen von Bildern basierend auf einem Tensor von Positionen
    """Reorders the image given a tensor of positions"""  # Explanation of the class functionality / Erklärung der Funktionsweise der Klasse

    def __call__(self, sample: Dict[str, torch.tensor]) -> (torch.tensor, torch.tensor):  # Calls the transformation for reordering the images / Ruft die Transformation zum Neuanordnen der Bilder auf
        """
        Applies the transformation to the sequence of images.  # Applies reordering logic to the given image sequence / Wendet die Neuanordnungslogik auf die gegebene Bildsequenz an

        :param Dict[str, torch.tensor] sample: Sequence of images  # Expects a dictionary with image data as tensors / Erwartet ein Dictionary mit Bilddaten als Tensoren
        :return: Dict[str, torch.tensor]- Reordered sequence of images  # Returns reordered images and the original data / Gibt die neu angeordneten Bilder und die Originaldaten zurück
        """
        images, y = sample  # Extracts images and their order from the input sample / Extrahiert Bilder und deren Reihenfolge aus dem Eingabebeispiel

        return images[y], sample  # Reorders the images based on the indices in y and returns / Ordnet die Bilder basierend auf den Indizes in y neu und gibt sie zurück


class Tedd1104Dataset(Dataset):  # Defines a custom Dataset class for the TEDD1104 dataset / Definiert eine benutzerdefinierte Dataset-Klasse für den TEDD1104-Datensatz
    """TEDD1104 Reordering dataset."""  # Description of the class functionality / Beschreibung der Funktionsweise der Klasse

    def __init__(self,  # Constructor for initializing the dataset object with various parameters / Konstruktor zum Initialisieren des Dataset-Objekts mit verschiedenen Parametern
        dataset_dir: str,  # Directory where the dataset is stored / Verzeichnis, in dem der Datensatz gespeichert ist
        hide_map_prob: float,  # Probability of hiding the minimap from images / Wahrscheinlichkeit, dass die Mini-Karte aus den Bildern entfernt wird
        token_mask_prob: float,  # Probability of masking tokens in the transformer model / Wahrscheinlichkeit, dass Tokens im Transformer-Modell maskiert werden
        transformer_nheads: int = None,  # Number of heads in the transformer model (None if LSTM is used) / Anzahl der Köpfe im Transformer-Modell (None, wenn LSTM verwendet wird)
        dropout_images_prob: List[float] = 0.0,  # Probability of dropping specific images from the sequence / Wahrscheinlichkeit, bestimmte Bilder aus der Sequenz zu entfernen
        sequence_length: int = 5,  # Length of the image sequence to process / Länge der Bildsequenz, die verarbeitet werden soll
        train: bool = False,  # Whether the dataset is used for training or testing / Ob der Datensatz für das Training oder den Test verwendet wird
    ):
        """
        INIT  # Initialization function for the dataset / Initialisierungsfunktion für den Datensatz

        :param str dataset_dir: The directory of the dataset.  # Directory containing the dataset / Verzeichnis, das den Datensatz enthält
        :param bool hide_map_prob: Probability of hiding the minimap (0<=hide_map_prob<=1)  # Probability range for hiding the minimap / Wahrscheinlichkeitsbereich für das Entfernen der Mini-Karte
        :param bool token_mask_prob: Probability of masking a token in the transformer model (0<=token_mask_prob<=1)  # Probability range for token masking / Wahrscheinlichkeitsbereich für das Maskieren von Tokens
        :param int transformer_nheads: Number of heads in the transformer model, None if LSTM is used  # Number of transformer heads or None for LSTM model / Anzahl der Transformer-Köpfe oder None für das LSTM-Modell
        :param List[float] dropout_images_prob: Probability of dropping an image (0<=dropout_images_prob<=1)  # List of probabilities for each image in the sequence / Liste von Wahrscheinlichkeiten für jedes Bild in der Sequenz
        :param int sequence_length: Length of the image sequence  # Length of the sequence to process / Länge der zu verarbeitenden Sequenz
        :param bool train: If True, the dataset is used for training.  # Indicates whether the dataset is for training / Gibt an, ob der Datensatz für das Training verwendet wird
        """
        
        # Initializing variables with input arguments / Initialisiert Variablen mit den Eingabeargumenten
        self.dataset_dir = dataset_dir  # Directory of the dataset / Verzeichnis des Datensatzes
        self.hide_map_prob = hide_map_prob  # Probability of hiding the minimap / Wahrscheinlichkeit, die Mini-Karte zu verstecken
        self.dropout_images_prob = dropout_images_prob  # Probability list for dropping images / Liste von Wahrscheinlichkeiten zum Entfernen von Bildern
        self.sequence_length = sequence_length  # Length of the image sequence / Länge der Bildsequenz
        self.token_mask_prob = token_mask_prob  # Probability for token masking / Wahrscheinlichkeit für das Maskieren von Tokens
        self.transformer_nheads = transformer_nheads  # Number of transformer heads / Anzahl der Transformer-Köpfe
        self.train = train  # Flag indicating if the dataset is for training / Flag, das angibt, ob der Datensatz für das Training ist

        # Assert statements to check valid input ranges / Überprüfungsanweisungen, um gültige Eingabebereiche sicherzustellen
        assert 0 <= hide_map_prob <= 1.0, (  # Assert statement for hide_map_prob / Überprüfungsanweisung für hide_map_prob
            f"hide_map_prob not in 0 <= hide_map_prob <= 1.0 range. "
            f"hide_map_prob: {hide_map_prob}"
        )

        # Further assertions for dropout_images_prob and token_mask_prob / Weitere Überprüfungen für dropout_images_prob und token_mask_prob
        assert len(dropout_images_prob) == 5, (
            f"dropout_images_prob must have 5 probabilities, one for each image in the sequence. "
            f"dropout_images_prob len: {len(dropout_images_prob)}"
        )
        
        # Check that probabilities for dropping images are valid / Überprüft, dass die Wahrscheinlichkeiten zum Entfernen von Bildern gültig sind
        for dropout_image_prob in dropout_images_prob:
            assert 0 <= dropout_image_prob < 1.0, (
                f"All probabilities in dropout_image_prob must be in the range 0 <= dropout_image_prob < 1.0. "
                f"dropout_images_prob: {dropout_images_prob}"
            )

        # Check the range of token_mask_prob / Überprüft den Bereich von token_mask_prob
        assert 0 <= token_mask_prob < 1.0, (
            f"token_mask_prob not in 0 <= token_mask_prob < 1.0 range. "
            f"token_mask_prob: {token_mask_prob}"
        )

        # Setting up image transformations based on training or testing phase / Einrichten der Bildtransformationen je nach Trainings- oder Testphase
        if train:  # If in training mode, apply transformations for data augmentation / Wenn im Trainingsmodus, wende Transformationen für die Datenaugmentation an
            self.transform = transforms.Compose(
                [
                    RemoveMinimap(hide_map_prob=hide_map_prob),  # Remove minimap with given probability / Entfernt die Mini-Karte mit der angegebenen Wahrscheinlichkeit
                    RemoveImage(dropout_images_prob=dropout_images_prob),  # Drop images based on probabilities / Entfernt Bilder basierend auf den Wahrscheinlichkeiten
                    SplitImages(),  # Split images into separate components / Teilt die Bilder in separate Komponenten
                    SequenceColorJitter(),  # Apply random color jitter to images / Wendet zufällige Farbänderungen auf die Bilder an
                    Normalize(),  # Normalize image values / Normalisiert die Bildwerte
                    ReOrderImages(),  # Reorder the images according to the tensor of positions / Ordnet die Bilder entsprechend dem Tensor von Positionen neu
                ]
            )
        else:  # If in testing mode, skip certain augmentations / Wenn im Testmodus, überspringe bestimmte Augmentationen
            self.transform = transforms.Compose(
                [
                    RemoveMinimap(hide_map_prob=hide_map_prob),  # Remove minimap / Entfernt die Mini-Karte
                    # RemoveImage(dropout_images_prob=dropout_images_prob),  # Skip removing images / Überspringt das Entfernen von Bildern
                    SplitImages(),  # Split images / Teilt die Bilder
                    # SequenceColorJitter(),  # Skip color jitter / Überspringt Farbänderungen
                    Normalize(),  # Normalize images / Normalisiert die Bilder
                    ReOrderImages(),  # Reorder images / Ordnet die Bilder neu
                ]
            )
        
        self.dataset_files = glob.glob(os.path.join(dataset_dir, "*.jpeg"))  # Loads all image files with .jpeg extension / Lädt alle Bilddateien mit der Erweiterung .jpeg
        self.IOHandler = IOHandler()  # Initializes the IOHandler for managing dataset input/output / Initialisiert den IOHandler zum Verwalten von Datensatz-Eingabe/Ausgabe

    def __len__(self):  # Returns the number of images in the dataset / Gibt die Anzahl der Bilder im Datensatz zurück
        """
        Returns the length of the dataset.  # Length of the dataset in terms of image count / Länge des Datensatzes in Bezug auf die Bildanzahl

        :return: int - Length of the dataset.  # Rückgabe der Länge des Datensatzes / Returns the dataset length
        """
        return len(self.dataset_files)  # Returns the number of image files in the dataset / Gibt die Anzahl der Bilddateien im Datensatz zurück

    def __getitem__(self, idx):  # Fetches a sample (image) from the dataset by index / Holt ein Beispiel (Bild) aus dem Datensatz nach Index
        """
        Returns a sample from the dataset.  # Gibt ein Beispiel aus dem Datensatz zurück

        :param int idx: Index of the sample.  # Index des Beispiels / Index of the sample
        :return: Dict[str, torch.tensor]- Transformed sequence of images  # Transformed image sequence / Transformierte Bildsequenz
        """
        if torch.is_tensor(idx):  # Checks if idx is a tensor, and converts to integer if true / Überprüft, ob idx ein Tensor ist, und konvertiert ihn in eine Ganzzahl, falls wahr
            idx = int(idx)  # Converts tensor index to integer / Konvertiert den Tensor-Index in eine Ganzzahl

        img_name = self.dataset_files[idx]  # Gets the image name based on index / Holt den Bildnamen basierend auf dem Index
        image = None  # Initializes image variable / Initialisiert die Bild-Variable
        while image is None:  # Loop until a valid image is found / Schleife, bis ein gültiges Bild gefunden wird
            try:
                image = torchvision.io.read_image(img_name)  # Reads image using torchvision / Liest das Bild mit torchvision
            except (ValueError, FileNotFoundError) as err:  # Catches exceptions if the image is corrupt or missing / Fängt Ausnahmen ein, wenn das Bild beschädigt oder fehlend ist
                error_message = str(err).split("\n")[-1]  # Extracts the error message / Extrahiert die Fehlermeldung
                print(  # Prints error message and loads a random image instead / Gibt eine Fehlermeldung aus und lädt stattdessen ein zufälliges Bild
                    f"Error reading image: {img_name} probably a corrupted file.\n"
                    f"Exception: {error_message}\n"
                    f"We will load a random image instead."
                )
                img_name = self.dataset_files[
                    int(len(self.dataset_files) * torch.rand(1))  # Chooses a random image from the dataset if error occurs / Wählt ein zufälliges Bild aus dem Datensatz, wenn ein Fehler auftritt
                ]

        y = torch.randperm(5)  # Generates a random permutation of numbers 0-4 for reordering / Erzeugt eine zufällige Permutation der Zahlen 0-4 für die Neuanordnung

        image, y = self.transform((image, y))  # Applies the transformations to the image and reordering sequence / Wendet die Transformationen auf das Bild und die Neuanordnungssequenz an

        mask = get_mask(  # Generates a mask based on the specified parameters / Erzeugt eine Maske basierend auf den angegebenen Parametern
            train=self.train,  # Whether it's for training or not / Ob es für das Training oder nicht ist
            nheads=self.transformer_nheads,  # Number of transformer heads / Anzahl der Transformer-Köpfe
            mask_prob=self.token_mask_prob,  # Probability of masking tokens / Wahrscheinlichkeit für das Maskieren von Tokens
            sequence_length=self.sequence_length,  # Length of the image sequence / Länge der Bildsequenz
        )

        return image, mask, y  # Returns the transformed image, mask, and reordering sequence / Gibt das transformierte Bild, die Maske und die Neuanordnungssequenz zurück


class Tedd1104ataModuleForImageReordering(pl.LightningDataModule):  # Defines the DataModule for handling TEDD1104 dataset with PyTorch Lightning / Definiert das DataModule zur Handhabung des TEDD1104-Datensatzes mit PyTorch Lightning
    """
    Tedd1104DataModule is a PyTorch Lightning DataModule for the Tedd1104 dataset.  # DataModule for TEDD1104 dataset / DataModule für den TEDD1104-Datensatz
    """


    def __init__(self,  # Constructor for initializing the DataModule with necessary arguments / Konstruktor zum Initialisieren des DataModules mit erforderlichen Argumenten
        dataset_dir: str,  # Directory where the dataset is stored / Verzeichnis, in dem der Datensatz gespeichert ist
        hide_map_prob: float,  # Probability of hiding the minimap from images / Wahrscheinlichkeit, dass die Mini-Karte aus den Bildern entfernt wird
        token_mask_prob: float,  # Probability of masking tokens in the transformer model / Wahrscheinlichkeit, dass Tokens im Transformer-Modell maskiert werden
        transformer_nheads: int = None,  # Number of heads in the transformer model, None if LSTM is used / Anzahl der Köpfe im Transformer-Modell, None wenn LSTM verwendet wird
        dropout_images_prob: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0],  # Probability list for dropping specific images in the sequence / Liste von Wahrscheinlichkeiten, bestimmte Bilder in der Sequenz zu entfernen
        sequence_length: int = 5,  # Length of the image sequence to process / Länge der Bildsequenz, die verarbeitet werden soll
        batch_size: int = 32,  # Batch size for loading the dataset / Batch-Größe für das Laden des Datensatzes
        train: bool = True,  # Whether the dataset is for training or testing / Ob der Datensatz für das Training oder den Test verwendet wird
    ):
        """
        Initializes the TEDD1104DataModule for loading and processing the dataset. / Initialisiert das TEDD1104DataModule zum Laden und Verarbeiten des Datensatzes.

        :param dataset_dir: The directory containing the TEDD1104 dataset images. / Verzeichnis, das die TEDD1104-Datensatzbilder enthält.
        :param hide_map_prob: Probability of hiding the minimap. / Wahrscheinlichkeit, die Mini-Karte zu verstecken.
        :param token_mask_prob: Probability of masking tokens in the transformer model. / Wahrscheinlichkeit, Tokens im Transformer-Modell zu maskieren.
        :param transformer_nheads: Number of transformer heads. / Anzahl der Transformer-Köpfe.
        :param dropout_images_prob: Probabilities for dropping images in the sequence. / Wahrscheinlichkeiten für das Entfernen von Bildern in der Sequenz.
        :param sequence_length: Length of the image sequence. / Länge der Bildsequenz.
        :param batch_size: The batch size for loading the data. / Die Batch-Größe zum Laden der Daten.
        :param train: If True, it loads the training set, otherwise the test set. / Wenn True, wird der Trainingsdatensatz geladen, andernfalls der Testdatensatz.
        """
        super().__init__()

        self.dataset_dir = dataset_dir  # Directory for dataset / Verzeichnis für den Datensatz
        self.hide_map_prob = hide_map_prob  # Probability for hiding minimap / Wahrscheinlichkeit für das Entfernen der Mini-Karte
        self.token_mask_prob = token_mask_prob  # Probability for token masking / Wahrscheinlichkeit für das Maskieren von Tokens
        self.transformer_nheads = transformer_nheads  # Number of transformer heads / Anzahl der Transformer-Köpfe
        self.dropout_images_prob = dropout_images_prob  # Probabilities for dropping images / Wahrscheinlichkeiten für das Entfernen von Bildern
        self.sequence_length = sequence_length  # Length of the sequence / Länge der Sequenz
        self.batch_size = batch_size  # Batch size for data loading / Batch-Größe zum Laden der Daten
        self.train = train  # Flag indicating training mode / Flag, das den Trainingsmodus angibt

    def setup(self, stage: Optional[str] = None):  # Setup method for preparing the dataset for training or testing / Setup-Methode zur Vorbereitung des Datensatzes für Training oder Test
        """
        Set up the dataset, split into training and testing sets based on the stage. / Setzt den Datensatz auf, teilt ihn in Trainings- und Testdatensätze je nach Phase.

        :param stage: The stage for which to setup (e.g., "fit", "test"). / Die Phase, für die das Setup durchgeführt werden soll (z.B. "fit", "test").
        """
        if stage == 'fit' or stage is None:  # If the stage is for fitting (training), setup the training dataset / Wenn die Phase für das Training (fit) ist, richte den Trainingsdatensatz ein
            self.train_dataset = Tedd1104Dataset(
                dataset_dir=self.dataset_dir,  # Directory of the dataset / Verzeichnis des Datensatzes
                hide_map_prob=self.hide_map_prob,  # Probability of hiding minimap / Wahrscheinlichkeit des Entfernens der Mini-Karte
                token_mask_prob=self.token_mask_prob,  # Probability of token masking / Wahrscheinlichkeit des Maskierens von Tokens
                transformer_nheads=self.transformer_nheads,  # Number of transformer heads / Anzahl der Transformer-Köpfe
                dropout_images_prob=self.dropout_images_prob,  # Probability of dropping images / Wahrscheinlichkeit, bestimmte Bilder zu entfernen
                sequence_length=self.sequence_length,  # Length of image sequence / Länge der Bildsequenz
                train=True,  # Set for training / Für das Training
            )

        if stage == 'test' or stage is None:  # If the stage is for testing, setup the test dataset / Wenn die Phase für den Test (test) ist, richte den Testdatensatz ein
            self.test_dataset = Tedd1104Dataset(
                dataset_dir=self.dataset_dir,  # Directory of the dataset / Verzeichnis des Datensatzes
                hide_map_prob=self.hide_map_prob,  # Probability of hiding minimap / Wahrscheinlichkeit des Entfernens der Mini-Karte
                token_mask_prob=self.token_mask_prob,  # Probability of token masking / Wahrscheinlichkeit des Maskierens von Tokens
                transformer_nheads=self.transformer_nheads,  # Number of transformer heads / Anzahl der Transformer-Köpfe
                dropout_images_prob=self.dropout_images_prob,  # Probability of dropping images / Wahrscheinlichkeit des Entfernens von Bildern
                sequence_length=self.sequence_length,  # Length of image sequence / Länge der Bildsequenz
                train=False,  # Set for testing / Für den Test
            )

    def train_dataloader(self):  # Method to get the DataLoader for training / Methode, um den DataLoader für das Training zu bekommen
        """
        Returns the DataLoader for training dataset. / Gibt den DataLoader für den Trainingsdatensatz zurück.
        """
        return DataLoader(
            self.train_dataset,  # Dataset for training / Datensatz für das Training
            batch_size=self.batch_size,  # Batch size / Batch-Größe
            shuffle=True,  # Shuffle the dataset / Shufflet den Datensatz
            num_workers=4,  # Number of workers for data loading / Anzahl der Arbeiter für das Laden der Daten
            collate_fn=collate_fn,  # Function for collating the batch samples / Funktion zum Zusammenstellen der Batch-Proben
        )

    def val_dataloader(self):  # Method to get the DataLoader for validation dataset / Methode, um den DataLoader für den Validierungsdatensatz zu bekommen
        """
        Returns the DataLoader for validation dataset. / Gibt den DataLoader für den Validierungsdatensatz zurück.
        """
        return DataLoader(
            self.test_dataset,  # Dataset for validation / Datensatz für die Validierung
            batch_size=self.batch_size,  # Batch size / Batch-Größe
            shuffle=False,  # Do not shuffle for validation / Nicht shuffeln für die Validierung
            num_workers=4,  # Number of workers for data loading / Anzahl der Arbeiter für das Laden der Daten
            collate_fn=collate_fn,  # Function for collating the batch samples / Funktion zum Zusammenstellen der Batch-Proben
        )

    def test_dataloader(self):  # Method to get the DataLoader for testing dataset / Methode, um den DataLoader für den Testdatensatz zu bekommen
        """
        Returns the DataLoader for testing dataset. / Gibt den DataLoader für den Testdatensatz zurück.
        """
        return DataLoader(
            self.test_dataset,  # Dataset for testing / Datensatz für den Test
            batch_size=self.batch_size,  # Batch size / Batch-Größe
            shuffle=False,  # Do not shuffle for testing / Nicht shuffeln für den Test
            num_workers=4,  # Number of workers for data loading / Anzahl der Arbeiter für das Laden der Daten
            collate_fn=collate_fn,  # Function for collating the batch samples / Funktion zum Zusammenstellen der Batch-Proben
        )
