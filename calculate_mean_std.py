from dataset import SplitImages  # Importing the SplitImages class from the 'dataset' module for image processing. / Importiert die SplitImages-Klasse aus dem 'dataset'-Modul zur Bildverarbeitung.
import os  # Importing the 'os' module for interacting with the operating system. / Importiert das 'os'-Modul für die Interaktion mit dem Betriebssystem.
import glob  # Importing the 'glob' module for file pattern matching. / Importiert das 'glob'-Modul für die Dateimustererkennung.
import torch  # Importing the PyTorch library for tensor operations and deep learning. / Importiert die PyTorch-Bibliothek für Tensoroperationen und Deep Learning.
import torchvision  # Importing the torchvision library for vision-related utilities. / Importiert die torchvision-Bibliothek für vision-bezogene Hilfsfunktionen.
from tqdm.auto import tqdm  # Importing tqdm for displaying progress bars. / Importiert tqdm zum Anzeigen von Fortschrittsbalken.
import argparse  # Importing argparse for command-line argument parsing. / Importiert argparse zum Parsen von Befehlszeilenargumenten.
from torch.utils.data import Dataset, DataLoader  # Importing Dataset and DataLoader classes from PyTorch for handling datasets. / Importiert die Dataset- und DataLoader-Klassen aus PyTorch zur Handhabung von Datensätzen.

class Tedd1104Dataset(Dataset):  # Defining a custom dataset class inheriting from PyTorch's Dataset. / Definiert eine benutzerdefinierte Datensatzklasse, die von PyTorchs Dataset erbt.
    """TEDD1104 dataset."""  # A docstring describing the dataset. / Eine Dokumentation, die den Datensatz beschreibt.

    def __init__(self, dataset_dir: str):  # Initializing the dataset with a directory path. / Initialisiert den Datensatz mit einem Verzeichnispfad.

        self.dataset_dir = dataset_dir  # Storing the dataset directory path. / Speichert den Verzeichnispfad des Datensatzes.

        self.transform = torchvision.transforms.Compose([  # Defining a transformation pipeline using Compose. / Definiert eine Transformations-Pipeline mit Compose.
            SplitImages(),  # Applying the SplitImages transformation. / Anwenden der SplitImages-Transformation.
        ])

        self.dataset_files = glob.glob(os.path.join(dataset_dir, "*.jpeg"))  # Storing all JPEG files in the dataset directory. / Speichert alle JPEG-Dateien im Datensatzverzeichnis.

    def __len__(self):  # Defining the length of the dataset. / Definiert die Länge des Datensatzes.
        """
        Returns the length of the dataset. / Gibt die Länge des Datensatzes zurück.

        :return: int - Length of the dataset. / Rückgabe: int - Länge des Datensatzes.
        """
        return len(self.dataset_files)  # Returning the number of files in the dataset. / Gibt die Anzahl der Dateien im Datensatz zurück.

    def __getitem__(self, idx):  # Method to retrieve a sample from the dataset by index. / Methode zum Abrufen einer Stichprobe aus dem Datensatz anhand des Index.
        """
        Returns a sample from the dataset. / Gibt eine Stichprobe aus dem Datensatz zurück.

        :param int idx: Index of the sample. / Parameter int idx: Index der Stichprobe.
        :return: torch.tensor - Transformed sequence of images / Rückgabe: torch.tensor - Transformierte Bildsequenz
        """
        if torch.is_tensor(idx):  # Checking if the index is a tensor. / Überprüft, ob der Index ein Tensor ist.
            idx = int(idx)  # If so, converting it to an integer. / Wenn ja, wird es in eine Ganzzahl umgewandelt.

        image = torchvision.io.read_image(self.dataset_files[idx])  # Reading the image at the given index. / Liest das Bild am angegebenen Index.
        images, _ = self.transform((image, 0))  # Applying the transform to the image. / Wendet die Transformation auf das Bild an.
        return images  # Returning the transformed images. / Gibt die transformierten Bilder zurück.


def collate_fn(batch):  # Custom collate function for batching. / Benutzerdefinierte Collate-Funktion zum Erstellen von Batches.
    """
    Collate function for the dataloader. / Collate-Funktion für den Dataloader.

    :param batch: List of samples / Parameter batch: Liste von Stichproben
    :return: torch.tensor - Transformed sequence of images / Rückgabe: torch.tensor - Transformierte Bildsequenz
    """

    return torch.cat(batch, dim=0)  # Concatenating all batch samples into a single tensor. / Verbindet alle Stichproben des Batches zu einem einzelnen Tensor.


def calculate_mean_str(dataset_dir: str):  # Function to calculate the mean and standard deviation of the dataset. / Funktion zur Berechnung des Mittelwerts und der Standardabweichung des Datensatzes.
    dataset_files = list(glob.glob(os.path.join(dataset_dir, "*.jpeg")))  # Fetching all JPEG files in the dataset directory. / Ruft alle JPEG-Dateien im Datensatzverzeichnis ab.

    mean_sum = torch.tensor([0.0, 0.0, 0.0])  # Initializing sum tensor for mean. / Initialisiert den Summen-Tensor für den Mittelwert.
    stds_sum = torch.tensor([0.0, 0.0, 0.0])  # Initializing sum tensor for standard deviation. / Initialisiert den Summen-Tensor für die Standardabweichung.
    total = 0  # Initializing the counter for the total number of images. / Initialisiert den Zähler für die Gesamtzahl der Bilder.
    dataset = Tedd1104Dataset(dataset_dir=dataset_dir)  # Initializing the dataset object. / Initialisiert das Dataset-Objekt.
    dataloader = DataLoader(  # Creating a DataLoader to load the dataset in batches. / Erstellen eines DataLoaders, um den Datensatz in Batches zu laden.
        dataset=dataset,
        batch_size=64,  # Defining batch size. / Definiert die Batch-Größe.
        collate_fn=collate_fn,  # Using the custom collate function. / Verwendet die benutzerdefinierte Collate-Funktion.
        num_workers=os.cpu_count() // 2,  # Using half of the available CPU cores for loading the data. / Verwendet die Hälfte der verfügbaren CPU-Kerne für das Laden der Daten.
    )
    with tqdm(  # Using tqdm to display a progress bar. / Verwendet tqdm, um einen Fortschrittsbalken anzuzeigen.
        total=len(dataloader),
        desc=f"Reading images",
    ) as pbar:
        for batch in dataloader:  # Iterating through the dataset batches. / Iteriert durch die Batches des Datensatzes.
            for image in batch:  # Iterating through each image in the batch. / Iteriert durch jedes Bild im Batch.
                for dim in range(3):  # Iterating through each color channel (RGB). / Iteriert durch jeden Farbkanal (RGB).
                    channel = image[dim] / 255.0  # Normalizing the pixel values of the channel. / Normalisiert die Pixelwerte des Kanals.
                    mean_sum[dim] += torch.mean(channel)  # Adding the mean of the channel to the sum. / Addiert den Mittelwert des Kanals zur Summe.
                    stds_sum[dim] += torch.std(channel)  # Adding the standard deviation of the channel to the sum. / Addiert die Standardabweichung des Kanals zur Summe.
                total += 1  # Incrementing the total counter. / Erhöht den Gesamtzähler.
            pbar.update(1)  # Updating the progress bar. / Aktualisiert den Fortschrittsbalken.
            pbar.set_description(  # Updating the progress bar description. / Aktualisiert die Beschreibung des Fortschrittsbalkens.
                desc=f"Reading images. "
                f"Mean: [{round(mean_sum[0].item()/total,6)},{round(mean_sum[1].item()/total,6)},{round(mean_sum[2].item()/total,6)}]. "
                f"STD: [{round(stds_sum[0].item()/total,6)},{round(stds_sum[1].item()/total,6)},{round(stds_sum[2].item()/total,6)}].",
            )

    mean = mean_sum / total  # Calculating the mean by dividing the sum by total images. / Berechnet den Mittelwert, indem die Summe durch die Gesamtzahl der Bilder geteilt wird.
    std = stds_sum / total  # Calculating the standard deviation by dividing the sum by total images. / Berechnet die Standardabweichung, indem die Summe durch die Gesamtzahl der Bilder geteilt wird.

    print(f"Mean: {mean}")  # Printing the mean values. / Gibt die Mittelwertwerte aus.
    print(f"std: {std}")  # Printing the standard deviation values. / Gibt die Standardabweichungswerte aus.

    return mean, std  # Returning the mean and standard deviation. / Gibt den Mittelwert und die Standardabweichung zurück.


if __name__ == "__main__":  # Checking if the script is run directly (not imported as a module). / Überprüft, ob das Skript direkt ausgeführt wird (nicht als Modul importiert).
    parser = argparse.ArgumentParser()  # Initializing the argument parser. / Initialisiert den Argument-Parser.

    parser.add_argument(  # Defining an argument for the dataset directory. / Definiert ein Argument für das Datensatzverzeichnis.
        "--dataset_dir",  # Argument name. / Name des Arguments.
        type=str,  # Type of the argument. / Typ des Arguments.
        required=True,  # Making the argument required. / Macht das Argument erforderlich.
        help="Path to the dataset directory containing jpeg files.",  # Help text for the argument. / Hilfetext für das Argument.
    )

    args = parser.parse_args()  # Parsing the command-line arguments. / Parsen der Befehlszeilenargumente.

    mean, std = calculate_mean_str(dataset_dir=args.dataset_dir)  # Calling the function to calculate mean and standard deviation. / Ruft die Funktion auf, um Mittelwert und Standardabweichung zu berechnen.
    with open("image_metrics.txt", "w", encoding="utf8") as output_file:  # Opening a file to write the results. / Öffnet eine Datei, um die Ergebnisse zu schreiben.
        print(f"Mean: {mean.numpy()}", file=output_file)  # Writing the mean values to the file. / Schreibt die Mittelwertwerte in die Datei.
        print(f"STD: {std.numpy()}", file=output_file)  # Writing the standard deviation values to the file. / Schreibt die Standardabweichungswerte in die Datei.
