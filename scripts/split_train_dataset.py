import glob  # Importing the 'glob' module to find all file names matching a specified pattern. / Importiert das 'glob'-Modul, um alle Dateinamen zu finden, die einem angegebenen Muster entsprechen.
import os  # Importing the 'os' module for interacting with the operating system, such as file paths and directories. / Importiert das 'os'-Modul, um mit dem Betriebssystem zu interagieren, wie z.B. Dateipfade und Verzeichnisse.
from shutil import copyfile  # Importing the 'copyfile' function from the 'shutil' module to copy files. / Importiert die Funktion 'copyfile' aus dem 'shutil'-Modul, um Dateien zu kopieren.
from tqdm.auto import tqdm  # Importing the 'tqdm' function to show progress bars. / Importiert die 'tqdm'-Funktion, um Fortschrittsbalken anzuzeigen.
from shlex import quote  # Importing 'quote' from the 'shlex' module to safely escape file paths for command-line use. / Importiert 'quote' aus dem 'shlex'-Modul, um Dateipfade sicher für die Verwendung in der Kommandozeile zu maskieren.
import argparse  # Importing 'argparse' to parse command-line arguments. / Importiert 'argparse', um Kommandozeilenargumente zu parsen.
import math  # Importing the 'math' module for mathematical operations. / Importiert das 'math'-Modul für mathematische Operationen.

def split_and_compress_dataset(  # Defining the function to split and compress a dataset. / Definiert die Funktion zum Teilen und Komprimieren eines Datensatzes.
    dataset_dir: str,  # The directory where the dataset is located. / Das Verzeichnis, in dem sich der Datensatz befindet.
    output_dir: str,  # The directory where the output will be saved. / Das Verzeichnis, in dem die Ausgabedateien gespeichert werden.
    splits: int = 20,  # Number of splits to create, default is 20. / Anzahl der zu erstellenden Teile, Standard ist 20.
):
    """
    Split the dataset into "splits" subfolders of images and compress them. / Teilt den Datensatz in "splits" Unterordner mit Bildern und komprimiert sie.

    :param str dataset_dir: Path to the dataset directory. / :param str dataset_dir: Pfad zum Datensatzverzeichnis.
    :param str output_dir: Path to the output directory. / :param str output_dir: Pfad zum Ausgabeverzeichnis.
    :param int splits: Number of splits to create. / :param int splits: Anzahl der zu erstellenden Teile.
    """

    dataset_files = glob.glob(os.path.join(dataset_dir, "*.jpeg"))  # Get a list of all .jpeg files in the dataset directory. / Holt eine Liste aller .jpeg-Dateien im Datensatzverzeichnis.
    img_per_file = math.ceil(len(dataset_files) / splits)  # Calculate how many images should be in each split. / Berechnet, wie viele Bilder in jedem Teil enthalten sein sollten.

    print(  # Print the details of the splitting process. / Gibt Details zum Teilungsprozess aus.
        f"Splitting dataset into {splits} subfolders of {img_per_file} images each. Total images: {len(dataset_files)}"  # Displays the number of splits and images. / Zeigt die Anzahl der Teile und Bilder an.
    )
    print(f"Dataset directory: {dataset_dir}")  # Print the dataset directory path. / Gibt den Pfad des Datensatzverzeichnisses aus.
    print(f"Output directory: {output_dir}")  # Print the output directory path. / Gibt den Pfad des Ausgabeverzeichnisses aus.
    print("This may take a while, go grab a coffee!")  # Inform the user that the process will take time. / Informiert den Benutzer, dass der Prozess eine Weile dauern wird.
    print()  # Print a blank line for better readability. / Gibt eine Leerzeile zur besseren Lesbarkeit aus.

    # Split the dataset into multiple subfolders of img_per_file images. / Teilt den Datensatz in mehrere Unterordner mit img_per_file Bildern auf.
    for i in tqdm(  # Iterate over the dataset files in chunks. / Iteriert über die Datensatzdateien in Abschnitten.
        range(0, len(dataset_files), img_per_file), desc="Splitting dataset", position=0  # Progress bar for splitting. / Fortschrittsbalken für die Aufteilung.
    ):
        os.makedirs(os.path.join(output_dir, str(i // img_per_file)), exist_ok=True)  # Create a subfolder for each split. / Erstellt einen Unterordner für jedes Teil.
        for dataset_file in tqdm(  # Iterate over the files for this split. / Iteriert über die Dateien für dieses Teil.
            dataset_files[i : i + img_per_file], desc="Copying images", position=1  # Progress bar for copying images. / Fortschrittsbalken für das Kopieren von Bildern.
        ):
            copyfile(  # Copy each image to the respective subfolder. / Kopiert jedes Bild in den jeweiligen Unterordner.
                dataset_file,
                os.path.join(
                    output_dir,
                    str(i // img_per_file),
                    os.path.basename(dataset_file),  # Copy the image with its original filename. / Kopiert das Bild mit seinem ursprünglichen Dateinamen.
                ),
            )

        # Create zip file for this subfolder. / Erstellt eine Zip-Datei für diesen Unterordner.
        filename = f"TEDD1140_dataset_{i // img_per_file}.zip"  # Naming the zip file. / Benennt die Zip-Datei.
        os.system(  # Execute a shell command to zip the folder. / Führt einen Shell-Befehl aus, um den Ordner zu zippen.
            f"zip -r {quote(os.path.join(output_dir, filename))}.zip "
            f"{quote(os.path.join(output_dir, str(i // img_per_file)))}"
        )
        # Remove the folder after zipping. / Entfernt den Ordner nach dem Zippen.
        os.system(f"rm -rf {quote(os.path.join(output_dir, str(i // img_per_file)))}")  # Delete the folder after zipping. / Löscht den Ordner nach dem Zippen.

if __name__ == "__main__":  # If the script is being run directly (not imported). / Wenn das Skript direkt ausgeführt wird (nicht importiert).
    parser = argparse.ArgumentParser(  # Create a parser to handle command-line arguments. / Erstellt einen Parser, um Kommandozeilenargumente zu verarbeiten.
        description="Split dataset into multiple subfolders and compress them."  # Description for the script. / Beschreibung für das Skript.
    )

    parser.add_argument(  # Add argument for the dataset directory. / Fügt ein Argument für das Datensatzverzeichnis hinzu.
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset dir",  # Help message for the dataset directory argument. / Hilfenachricht für das Argument des Datensatzverzeichnisses.
    )

    parser.add_argument(  # Add argument for the output directory. / Fügt ein Argument für das Ausgabeverzeichnis hinzu.
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output dir",  # Help message for the output directory argument. / Hilfenachricht für das Argument des Ausgabeverzeichnisses.
    )

    parser.add_argument(  # Add argument for the number of splits. / Fügt ein Argument für die Anzahl der Teile hinzu.
        "--splits",
        type=int,
        default=20,
        help="Number of splits to create",  # Help message for the number of splits argument. / Hilfenachricht für das Argument der Anzahl der Teile.
    )

    args = parser.parse_args()  # Parse the command-line arguments. / Parsen der Kommandozeilenargumente.

    split_and_compress_dataset(  # Call the function to split and compress the dataset. / Ruft die Funktion auf, um den Datensatz zu teilen und zu komprimieren.
        dataset_dir=args.dataset_dir, output_dir=args.output_dir, splits=args.splits  # Pass the parsed arguments to the function. / Übergibt die geparsten Argumente an die Funktion.
    )
