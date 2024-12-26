from __future__ import print_function, division  # Ensures compatibility with older Python versions (e.g., Python 2 for print and division behavior) / Sichert die Kompatibilität mit älteren Python-Versionen (z.B. Python 2 für Print- und Divisionsverhalten)
import os  # Imports the os module for interacting with the operating system / Importiert das os-Modul, um mit dem Betriebssystem zu interagieren
import torch  # Imports the torch library for working with tensors and neural networks / Importiert die torch-Bibliothek für die Arbeit mit Tensoren und neuronalen Netzwerken
import torchvision.io  # Imports torchvision.io for image input/output operations / Importiert torchvision.io für Bild-Ein- und Ausgabefunktionen
from torch.utils.data import Dataset, DataLoader  # Imports Dataset and DataLoader for handling datasets efficiently / Importiert Dataset und DataLoader für effizientes Handling von Datensätzen
from torchvision import transforms  # Imports transforms for image transformation utilities / Importiert transforms für Bildtransformationshilfsfunktionen
import glob  # Imports glob for file pattern matching / Importiert glob für die Dateimustererkennung
from typing import List, Optional, Dict  # Imports typing utilities for type hinting / Importiert Typisierungs-Hilfsmittel für Type-Hinting
from utils import IOHandler, get_mask  # Imports custom utility functions for input/output handling and mask retrieval / Importiert benutzerdefinierte Hilfsfunktionen für Ein-/Ausgabeverwaltung und Maskenabfrage
import pytorch_lightning as pl  # Imports PyTorch Lightning for organizing and simplifying the training process / Importiert PyTorch Lightning zur Organisation und Vereinfachung des Trainingsprozesses


def count_examples(dataset_dir: str) -> int:  # Function to count the number of JPEG images in the given directory / Funktion zur Zählung der Anzahl von JPEG-Bildern im angegebenen Verzeichnis
    return len(glob.glob(os.path.join(dataset_dir, "*.jpeg")))  # Uses glob to find all JPEG files and return their count / Verwendet glob, um alle JPEG-Dateien zu finden und deren Anzahl zurückzugeben


class RemoveMinimap(object):  # Class to remove a minimap (black square) from all images in a sequence / Klasse zum Entfernen einer Minikarte (schwarzes Quadrat) aus allen Bildern einer Sequenz

    def __init__(self, hide_map_prob: float):  # Constructor to initialize the class with the probability of hiding the minimap / Konstruktor, um die Klasse mit der Wahrscheinlichkeit zu initialisieren, die Minikarte zu verbergen
        """
        INIT

        :param float hide_map_prob: Probability of hiding the minimap (0<=hide_map_prob<=1) / Wahrscheinlichkeit, die Minikarte zu verbergen (0<=hide_map_prob<=1)
        """

        self.hide_map_prob = hide_map_prob  # Sets the probability of hiding the minimap / Setzt die Wahrscheinlichkeit, die Minikarte zu verbergen

    def __call__(self, sample: Dict[str, torch.tensor]) -> (torch.tensor, torch.tensor):  # Applies the transformation to a sequence of images / Wendet die Transformation auf eine Sequenz von Bildern an
        """
        Applies the transformation to the sequence of images.

        :param Dict[str, np.ndarray] sample: Sequence of images / Sequenz von Bildern
        :return: Dict[str, np.ndarray] - Transformed sequence of images / Transformierte Sequenz von Bildern
        """

        image, y = sample  # Unpacks the image and label from the sample / Entpackt das Bild und das Label aus dem Sample

        width: int = int(image.size(2) / 5)  # Calculates the width of each image section in the sequence / Berechnet die Breite jedes Bildteils in der Sequenz

        if self.hide_map_prob > 0:  # Checks if there's any probability to hide the minimap / Überprüft, ob eine Wahrscheinlichkeit besteht, die Minikarte zu verbergen
            if torch.rand(1)[0] <= self.hide_map_prob:  # Generates a random number and compares it with the hide_map_prob / Generiert eine Zufallszahl und vergleicht sie mit der hide_map_prob
                for j in range(0, 5):  # Loops through 5 sections of the image sequence / Schleift durch 5 Abschnitte der Bildsequenz
                    image[:, 215:, j * width : (j * width) + 80] = torch.zeros(  # Sets a part of the image to black (zeros) / Setzt einen Teil des Bildes auf schwarz (Nullen)
                        (3, 55, 80), dtype=image.dtype  # Specifies the dimensions of the blacked-out part / Gibt die Dimensionen des ausgeblendeten Teils an
                    )

        return image, y  # Returns the transformed image and label / Gibt das transformierte Bild und das Label zurück


class RemoveImage(object):  # Class to remove random images (blackout) from the sequence / Klasse zum Entfernen zufälliger Bilder (Blackout) aus der Sequenz

    def __init__(self, dropout_images_prob: List[float]):  # Initializes with a list of probabilities for dropping each image / Initialisiert mit einer Liste von Wahrscheinlichkeiten, um jedes Bild zu entfernen
        """
        INIT

        :param List[float] dropout_images_prob: Probability of dropping each image (0<=dropout_images_prob<=1) / Wahrscheinlichkeit, jedes Bild zu entfernen (0<=dropout_images_prob<=1)
        """
        self.dropout_images_prob = dropout_images_prob  # Sets the probability list for image dropout / Setzt die Wahrscheinlichkeitsliste für den Bild-Ausfall

    def __call__(self, sample: Dict[str, torch.tensor]) -> (torch.tensor, torch.tensor):  # Applies the transformation to a sequence of images / Wendet die Transformation auf eine Sequenz von Bildern an
        """
        Applies the transformation to the sequence of images.

        :param Dict[str, np.ndarray] sample: Sequence of images / Sequenz von Bildern
        :return: Dict[str, np.ndarray] - Transformed sequence of images / Transformierte Sequenz von Bildern
        """
        image, y = sample  # Unpacks the image and label from the sample / Entpackt das Bild und das Label aus dem Sample

        width: int = int(image.size(2) / 5)  # Calculates the width of each image section in the sequence / Berechnet die Breite jedes Bildteils in der Sequenz

        for j in range(0, 5):  # Loops through the 5 sections of the image sequence / Schleift durch die 5 Abschnitte der Bildsequenz
            if self.dropout_images_prob[j] > 0:  # Checks if the dropout probability for the current image is greater than 0 / Überprüft, ob die Ausfallwahrscheinlichkeit für das aktuelle Bild größer als 0 ist
                if torch.rand(1)[0] <= self.dropout_images_prob[j]:  # Generates a random number and compares it to the dropout probability / Generiert eine Zufallszahl und vergleicht sie mit der Ausfallwahrscheinlichkeit
                    image[:, :, j * width : (j + 1) * width] = torch.zeros(  # Sets the image section to black (zeros) if it's dropped / Setzt den Bildteil auf schwarz (Nullen), wenn er entfernt wird
                        (image.shape[0], image.shape[1], width), dtype=image.dtype  # Specifies the size of the blacked-out section / Gibt die Größe des ausgeblendeten Teils an
                    )

        return image, y  # Returns the transformed image and label / Gibt das transformierte Bild und das Label zurück


class SplitImages(object):  # Class to split a sequence of images into 5 parts / Klasse zum Aufteilen einer Bildsequenz in 5 Teile

    def __call__(self, sample: torch.tensor) -> (torch.tensor, torch.tensor):  # Applies the transformation to the sequence of images / Wendet die Transformation auf die Bildsequenz an
        """
        Applies the transformation to the sequence of images.

        :param np.ndarray sample: Sequence image / Sequenzbild
        :return: Dict[str, np.ndarray] - Transformed sequence of images / Transformierte Sequenz von Bildern
        """
        image, y = sample  # Unpacks the image and label from the sample / Entpackt das Bild und das Label aus dem Sample
        width: int = int(image.size(2) / 5)  # Calculates the width of each image section in the sequence / Berechnet die Breite jedes Bildteils in der Sequenz
        image1 = image[:, :, 0:width]  # Splits the image into 5 sections based on the calculated width / Teilt das Bild in 5 Abschnitte basierend auf der berechneten Breite
        image2 = image[:, :, width : width * 2]
        image3 = image[:, :, width * 2 : width * 3]
        image4 = image[:, :, width * 3 : width * 4]
        image5 = image[:, :, width * 4 : width * 5]
        return torch.stack([image1, image2, image3, image4, image5]), torch.tensor(y)  # Stacks the 5 image sections and returns them with the label / Stapelt die 5 Bildabschnitte und gibt sie zusammen mit dem Label zurück


class SequenceColorJitter(object):
    """
    Randomly change the brightness, contrast, and saturation of a sequence of images
    """
    # Klasse für die zufällige Änderung der Helligkeit, des Kontrasts und der Sättigung einer Bildsequenz

    def __init__(self, brightness=0.5, contrast=0.1, saturation=0.1, hue=0.5):
        """
        INIT

        :param float brightness: Probability of changing brightness (0<=brightness<=1)
        :param float contrast: Probability of changing contrast (0<=contrast<=1)
        :param float saturation: Probability of changing saturation (0<=saturation<=1)
        :param float hue: Probability of changing hue (0<=hue<=1)
        """
        # Initialisierung der Parameter für die Helligkeit, den Kontrast, die Sättigung und den Farbton
        self.jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, sample: Dict[str, torch.tensor]) -> (torch.tensor, torch.tensor):
        """
        Applies the transformation to the sequence of images.

        :param Dict[str, torch.tensor] sample: Sequence of images
        :return: Dict[str, torch.tensor]- Transformed sequence of images
        """
        # Wendet die Transformation auf eine Bildsequenz an
        images, y = sample
        images = self.jitter(images)
        return images, y


class Normalize(object):
    """
    Normalize a tensor image with mean and standard deviation.
    """
    # Normalisiert ein Bild mit Mittelwert und Standardabweichung

    transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    def __call__(self, sample: (torch.tensor, torch.tensor)) -> (torch.tensor, torch.tensor):
        """
        Applies the transformation to the sequence of images.

        :param Dict[str, torch.tensor] sample: Sequence of images
        :return: Dict[str, torch.tensor]- Transformed sequence of images
        """
        # Wendet die Normalisierung auf eine Bildsequenz an
        images, y = sample
        return (
            torch.stack(
                [
                    self.transform(images[0] / 255.0),
                    self.transform(images[1] / 255.0),
                    self.transform(images[2] / 255.0),
                    self.transform(images[3] / 255.0),
                    self.transform(images[4] / 255.0),
                ]
            ),
            y,
        )


def collate_fn(batch):
    """
    Collate function for the dataloader.

    :param batch: List of samples
    :return: Dict[str, torch.tensor]- Transformed sequence of images
    """
    # Kombiniert die Stichproben aus dem Dataloader
    return_dict: Dict[str, torch.tensor] = {
        "images": torch.cat([b[0] for b in batch], dim=0),
        "attention_mask": torch.cat([b[1] for b in batch], dim=0),
        "y": torch.stack([b[2] for b in batch]),
    }
    return_dict["attention_mask"].requires_grad = False
    return_dict["y"].requires_grad = False
    return return_dict


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    # Setzt die Strategie für das Teilen von Daten zwischen Prozessen auf Dateisystem


class Tedd1104Dataset(Dataset):
    """TEDD1104 dataset."""
    # TEDD1104-Datensatz

    def __init__(
        self,
        dataset_dir: str,
        hide_map_prob: float,
        token_mask_prob: float,
        transformer_nheads: int = None,
        dropout_images_prob: List[float] = None,
        sequence_length: int = 5,
        control_mode: str = "keyboard",
        train: bool = False,
    ):
        """
        INIT

        :param str dataset_dir: The directory of the dataset.
        :param bool hide_map_prob: Probability of hiding the minimap (0<=hide_map_prob<=1)
        :param bool token_mask_prob: Probability of masking a token in the transformer model (0<=token_mask_prob<=1)
        :param int transformer_nheads: Number of heads in the transformer model, None if LSTM is used
        :param List[float] dropout_images_prob: Probability of dropping an image (0<=dropout_images_prob<=1)
        :param int sequence_length: Length of the image sequence
        :param str control_mode: Type of the user input: "keyboard" or "controller"
        :param bool train: If True, the dataset is used for training.
        """
        # Initialisiert den Datensatz mit den gegebenen Parametern
        self.dataset_dir = dataset_dir
        self.hide_map_prob = hide_map_prob
        self.dropout_images_prob = (
            dropout_images_prob if dropout_images_prob else [0.0] * sequence_length
        )
        self.control_mode = control_mode.lower()
        self.sequence_length = sequence_length
        self.token_mask_prob = token_mask_prob
        self.transformer_nheads = transformer_nheads
        self.train = train

        assert self.control_mode in [
            "keyboard",
            "controller",
        ], f"{self.control_mode} control mode not supported. Supported dataset types: [keyboard, controller].  "
        # Stellt sicher, dass der Kontrollmodus entweder 'keyboard' oder 'controller' ist

        assert 0 <= self.hide_map_prob <= 1.0, (
            f"hide_map_prob not in 0 <= hide_map_prob <= 1.0 range. "
            f"hide_map_prob: {self.hide_map_prob}"
        )
        # Überprüft, ob die Wahrscheinlichkeit für das Ausblenden der Minimap im gültigen Bereich liegt

        assert len(self.dropout_images_prob) == 5, (
            f"dropout_images_prob must have 5 probabilities, one for each image in the sequence. "
            f"dropout_images_prob len: {len(dropout_images_prob)}"
        )
        # Stellt sicher, dass es 5 Wahrscheinlichkeiten für das Entfernen von Bildern gibt

        for dropout_image_prob in self.dropout_images_prob:
            assert 0 <= dropout_image_prob < 1.0, (
                f"All probabilities in dropout_image_prob must be in the range 0 <= dropout_image_prob < 1.0. "
                f"dropout_images_prob: {self.dropout_images_prob}"
            )
        # Überprüft, ob alle Wahrscheinlichkeiten im Bereich [0, 1) liegen

        assert 0 <= self.token_mask_prob < 1.0, (
            f"token_mask_prob not in 0 <= token_mask_prob < 1.0 range. "
            f"token_mask_prob: {self.token_mask_prob}"
        )
        # Überprüft, ob die Wahrscheinlichkeit zum Maskieren von Token im gültigen Bereich liegt

        if train:
            self.transform = transforms.Compose(
                [
                    RemoveMinimap(hide_map_prob=hide_map_prob),
                    RemoveImage(dropout_images_prob=dropout_images_prob),
                    SplitImages(),
                    SequenceColorJitter(),
                    Normalize(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    # RemoveMinimap(hide_map_prob=hide_map_prob),
                    # RemoveImage(dropout_images_prob=dropout_images_prob),
                    SplitImages(),
                    # SequenceColorJitter(),
                    Normalize(),
                ]
            )
        # Definiert die Transformationspipeline, abhängig davon, ob es sich um das Training handelt oder nicht

        self.dataset_files = glob.glob(os.path.join(dataset_dir, "*.jpeg"))
        # Lädt alle .jpeg-Dateien im angegebenen Verzeichnis

        self.IOHandler = IOHandler()
        # Initialisiert den IOHandler

    def __len__(self):
        """
        Returns the length of the dataset.

        :return: int - Length of the dataset.
        """
        # Gibt die Länge des Datensatzes zurück
        return len(self.dataset_files)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.

        :param int idx: Index of the sample.
        :return: Dict[str, torch.tensor]- Transformed sequence of images
        """
        # Gibt eine Stichprobe aus dem Datensatz zurück
        if torch.is_tensor(idx):
            idx = int(idx)

        img_name = self.dataset_files[idx]
        image = None
        while image is None:
            try:
                image = torchvision.io.read_image(img_name)
            except (ValueError, FileNotFoundError) as err:
                error_message = str(err).split("\n")[-1]
                print(
                    f"Error reading image: {img_name} probably a corrupted file.\n"
                    f"Exception: {error_message}\n"
                    f"We will load a random image instead."
                )
                img_name = self.dataset_files[
                    int(len(self.dataset_files) * torch.rand(1))
                ]

        y = self.IOHandler.imagename_input_conversion(
            image_name=img_name,
            output_type=self.control_mode,
        )
        # Liest das Bild und konvertiert den zugehörigen Input in das gewünschte Format

        image, y = self.transform((image, y))
        # Wendet die Transformation auf das Bild und den zugehörigen Wert an

        mask = get_mask(
            train=self.train,
            nheads=self.transformer_nheads,
            mask_prob=self.token_mask_prob,
            sequence_length=self.sequence_length,
        )
        # Erzeugt eine Maske für das Transformer-Modell

        return image, mask, y
        # Gibt das transformierte Bild, die Maske und das zugehörige Ziel zurück

# Define a class for the Tedd1104 dataset, inheriting from PyTorch Lightning's DataModule class.
class Tedd1104DataModule(pl.LightningDataModule):  # Definiert eine Klasse für das Tedd1104-Datenset, das von PyTorch Lightning's DataModule erbt.

    """
    Tedd1104DataModule is a PyTorch Lightning DataModule for the Tedd1104 dataset.
    """  # Tedd1104DataModule ist ein PyTorch Lightning DataModule für das Tedd1104-Datenset.

    def __init__(  # Konstruktor der Klasse.
        self,
        batch_size: int,  # Batch-Größe für das Dataset.
        train_dir: str = None,  # Verzeichnis des Trainings-Datensatzes.
        val_dir: str = None,  # Verzeichnis des Validierungs-Datensatzes.
        test_dir: str = None,  # Verzeichnis des Test-Datensatzes.
        token_mask_prob: float = 0.0,  # Wahrscheinlichkeit der Maskierung eines Tokens im Transformer-Modell (0<=token_mask_prob<=1).
        transformer_nheads: int = None,  # Anzahl der Köpfe im Transformer-Modell, None, wenn LSTM verwendet wird.
        sequence_length: int = 5,  # Länge der Bildsequenz.
        hide_map_prob: float = 0.0,  # Wahrscheinlichkeit, die Minimap zu verbergen (0<=hide_map_prob<=1).
        dropout_images_prob: List[float] = None,  # Wahrscheinlichkeit, ein Bild zu verwerfen (0<=dropout_images_prob<=1).
        control_mode: str = "keyboard",  # Erfasst die Eingabe vom "keyboard" oder "controller".
        num_workers: int = os.cpu_count(),  # Anzahl der Worker, die zum Laden des Datasets verwendet werden.
    ):
        """
        Initializes the Tedd1104DataModule.  # Initialisiert das Tedd1104DataModule.
        """
        super().__init__()  # Ruft den Konstruktor der Basisklasse auf.
        self.train_dir = train_dir  # Setzt das Verzeichnis des Trainings-Datensatzes.
        self.val_dir = val_dir  # Setzt das Verzeichnis des Validierungs-Datensatzes.
        self.test_dir = test_dir  # Setzt das Verzeichnis des Test-Datensatzes.
        self.batch_size = batch_size  # Setzt die Batch-Größe.
        self.token_mask_prob = token_mask_prob  # Setzt die Wahrscheinlichkeit für das Maskieren von Tokens.
        self.transformer_nheads = transformer_nheads  # Setzt die Anzahl der Köpfe im Transformer-Modell.
        self.sequence_length = sequence_length  # Setzt die Länge der Bildsequenz.
        self.hide_map_prob = hide_map_prob  # Setzt die Wahrscheinlichkeit für das Verbergen der Minimap.
        self.dropout_images_prob = (
            dropout_images_prob if dropout_images_prob else [0.0, 0.0, 0.0, 0.0, 0.0]  # Setzt die Wahrscheinlichkeit für das Verwerfen von Bildern, falls None, wird eine Liste mit Null-Werten verwendet.
        )
        self.control_mode = control_mode  # Setzt den Kontrollmodus ("keyboard" oder "controller").

        # Warnt, falls die Anzahl der Worker größer als 32 ist, was zu Speicherproblemen führen kann.
        if num_workers > 32:  
            print(
                "WARNING: num_workers is greater than 32, this may cause memory issues, consider using a smaller value."
                "Go ahead if you have a lot of RAM."  # Warnung vor möglichen Speicherproblemen bei mehr als 32 Arbeitern.
            )

        self.num_workers = num_workers  # Setzt die Anzahl der Worker.

    def setup(self, stage: Optional[str] = None) -> None:  # Bereitet das Dataset vor.
        """
        Sets up the dataset.  # Bereitet das Dataset vor.
        :param str stage: Stage of the setup.  # Die Phase der Vorbereitung (Trainieren, Testen, etc.).
        """
        if stage in (None, "fit"):  # Falls die Phase das Training ist.
            # Initialisiert das Trainings-Dataset mit den entsprechenden Parametern.
            self.train_dataset = Tedd1104Dataset(
                dataset_dir=self.train_dir,
                hide_map_prob=self.hide_map_prob,
                dropout_images_prob=self.dropout_images_prob,
                control_mode=self.control_mode,
                train=True,  # Gibt an, dass es sich um Trainingsdaten handelt.
                token_mask_prob=self.token_mask_prob,
                transformer_nheads=self.transformer_nheads,
                sequence_length=self.sequence_length,
            )
            print(f"Total training samples: {len(self.train_dataset)}.")  # Gibt die Anzahl der Trainingsproben aus.

            # Initialisiert das Validierungs-Dataset mit den entsprechenden Parametern.
            self.val_dataset = Tedd1104Dataset(
                dataset_dir=self.val_dir,
                hide_map_prob=0.0,  # Keine Minimap wird im Validierungs-Dataset verborgen.
                dropout_images_prob=[0.0, 0.0, 0.0, 0.0, 0.0],  # Keine Bilder werden im Validierungs-Dataset verworfen.
                control_mode="keyboard",  # Setzt den Kontrollmodus auf "keyboard".
                token_mask_prob=0.0,  # Keine Maskierung von Tokens im Validierungs-Dataset.
                transformer_nheads=self.transformer_nheads,
                sequence_length=self.sequence_length,
            )
            print(f"Total validation samples: {len(self.val_dataset)}.")  # Gibt die Anzahl der Validierungsproben aus.

        if stage in (None, "test"):  # Falls die Phase das Testen ist.
            # Initialisiert das Test-Dataset mit den entsprechenden Parametern.
            self.test_dataset = Tedd1104Dataset(
                dataset_dir=self.test_dir,
                hide_map_prob=0.0,  # Keine Minimap im Test-Dataset verborgen.
                dropout_images_prob=[0.0, 0.0, 0.0, 0.0, 0.0],  # Keine Bilder im Test-Dataset verworfen.
                control_mode="keyboard",  # Setzt den Kontrollmodus auf "keyboard".
                token_mask_prob=0.0,  # Keine Maskierung von Tokens im Test-Dataset.
                transformer_nheads=self.transformer_nheads,
                sequence_length=self.sequence_length,
            )
            print(f"Total test samples: {len(self.test_dataset)}.")  # Gibt die Anzahl der Testproben aus.

    def train_dataloader(self) -> DataLoader:  # Gibt den Dataloader für das Training zurück.
        """
        Returns the training dataloader.  # Gibt den Trainings-Dataloader zurück.
        :return: DataLoader - Training dataloader.  # Rückgabe eines DataLoaders für das Training.
        """
        return DataLoader(
            self.train_dataset,  # Das Trainings-Dataset.
            batch_size=self.batch_size,  # Die Batch-Größe.
            num_workers=self.num_workers,  # Die Anzahl der Worker zum Laden der Daten.
            pin_memory=True,  # Hält die Daten im GPU-Speicher.
            shuffle=True,  # Mischt die Daten.
            persistent_workers=True,  # Erhält die Worker über mehrere Datenladeoperationen hinweg.
            collate_fn=collate_fn,  # Definiert eine benutzerdefinierte Funktion zum Zusammenfügen von Batches.
            worker_init_fn=set_worker_sharing_strategy,  # Setzt die Initialisierungsstrategie für die Worker.
        )

    def val_dataloader(self) -> DataLoader:  # Gibt den Dataloader für die Validierung zurück.
        """
        Returns the validation dataloader.  # Gibt den Validierungs-Dataloader zurück.
        :return: DataLoader - Validation dataloader.  # Rückgabe eines DataLoaders für die Validierung.
        """
        return DataLoader(
            self.val_dataset,  # Das Validierungs-Dataset.
            batch_size=self.batch_size,  # Die Batch-Größe.
            num_workers=self.num_workers,  # Die Anzahl der Worker.
            pin_memory=True,  # Hält die Daten im GPU-Speicher.
            shuffle=False,  # Mischt die Validierungsdaten nicht.
            persistent_workers=True,  # Erhält die Worker über mehrere Ladeoperationen hinweg.
            collate_fn=collate_fn,  # Benutzt eine benutzerdefinierte Funktion zum Zusammenfügen von Batches.
            worker_init_fn=set_worker_sharing_strategy,  # Setzt die Worker-Initialisierungsstrategie.
        )

    def test_dataloader(self) -> DataLoader:  # Gibt den Dataloader für das Testen zurück.
        """
        Returns the test dataloader.  # Gibt den Test-Dataloader zurück.
        :return: DataLoader - Test dataloader.  # Rückgabe eines DataLoaders für das Testen.
        """
        return DataLoader(
            self.test_dataset,  # Das Test-Dataset.
            batch_size=self.batch_size,  # Die Batch-Größe.
            num_workers=self.num_workers,  # Die Anzahl der Worker.
            pin_memory=True,  # Hält die Daten im GPU-Speicher.
            shuffle=False,  # Mischt die Testdaten nicht.
            persistent_workers=True,  # Erhält die Worker über mehrere Ladeoperationen hinweg.
            collate_fn=collate_fn,  # Benutzt eine benutzerdefinierte Funktion zum Zusammenfügen von Batches.
            worker_init_fn=set_worker_sharing_strategy,  # Setzt die Worker-Initialisierungsstrategie.
        )
