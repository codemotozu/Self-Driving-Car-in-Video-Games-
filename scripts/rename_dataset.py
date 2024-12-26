import os  # Import the 'os' module for interacting with the operating system (English: This module allows access to OS-related functionality like file and directory operations. / German: Dieses Modul ermöglicht den Zugriff auf OS-bezogene Funktionen wie Datei- und Verzeichnisoperationen.)
import glob  # Import the 'glob' module to find all pathnames matching a specified pattern (English: This module is used to find files matching specific patterns in directories. / German: Dieses Modul wird verwendet, um Dateien zu finden, die bestimmten Mustern in Verzeichnissen entsprechen.)
import argparse  # Import the 'argparse' module for parsing command-line arguments (English: This module is used for handling command-line arguments passed to the script. / German: Dieses Modul wird verwendet, um Kommandozeilenargumente zu verarbeiten, die an das Skript übergeben werden.)
from tqdm import tqdm  # Import 'tqdm' for showing a progress bar (English: 'tqdm' provides a visual progress bar for loops. / German: 'tqdm' zeigt eine visuelle Fortschrittsanzeige für Schleifen.)

def rename_dataset(dataset_dir: str):  # Define a function to rename the dataset with a new naming convention (English: This function renames files in a specified directory according to a new naming pattern. / German: Diese Funktion benennt Dateien in einem angegebenen Verzeichnis nach einem neuen Namensmuster um.)
    """
    Rename legacy dataset to be consistent with the V5 naming convention.

    :param str dataset_dir: Path to the dataset directory.
    """  # A docstring explaining the purpose of the function (English: The function renames dataset files to align with the V5 naming standard. / German: Die Funktion benennt Datensatzdateien um, um mit dem V5-Namensstandard übereinzustimmen.)

    dataset_files = glob.glob(os.path.join(dataset_dir, "*.jpeg"))  # Get all .jpeg files in the specified directory (English: This line finds all files with a '.jpeg' extension in the provided dataset directory. / German: Diese Zeile findet alle Dateien mit der Erweiterung '.jpeg' im angegebenen Datensatzverzeichnis.)

    for dataset_file in tqdm(dataset_files):  # Loop over each file with a progress bar (English: This loop iterates over each found dataset file and shows the progress. / German: Diese Schleife durchläuft jede gefundene Datensatzdatei und zeigt den Fortschritt an.)
        metadata = os.path.basename(dataset_file)[:-5]  # Extract the filename without the extension (English: This removes the '.jpeg' extension and keeps the base name of the file. / German: Dies entfernt die '.jpeg'-Erweiterung und behält den Basisnamen der Datei bei.)
        imageno, key = metadata.split("_")  # Split the filename into two parts (English: This splits the filename into 'imageno' and 'key' based on an underscore. / German: Dies teilt den Dateinamen anhand eines Unterstrichs in 'imageno' und 'key'.)

        y = [[-1], [-1], [-1], [-1], [key]]  # Create a list 'y' with placeholders and the key (English: This creates a list 'y' that holds some placeholder values and the 'key'. / German: Dies erstellt eine Liste 'y', die Platzhalterwerte und den 'key' enthält.)

        new_name = (  # Construct the new filename following the naming convention
            "K"
            + str(imageno)
            + "%"
            + "_".join([",".join([str(e) for e in elem]) for elem in y])
            + ".jpeg"
        )  # The new name follows a specific pattern (English: This creates a new filename using 'imageno', placeholders, and the 'key'. / German: Dies erstellt einen neuen Dateinamen unter Verwendung von 'imageno', Platzhaltern und dem 'key'.)

        new_name = os.path.join(dataset_dir, new_name)  # Combine the new filename with the directory path (English: This combines the new filename with the original directory path. / German: Dies kombiniert den neuen Dateinamen mit dem ursprünglichen Verzeichnispfad.)

        os.rename(dataset_file, new_name)  # Rename the file with the new name (English: This renames the file from its old name to the new name. / German: Dies benennt die Datei vom alten Namen in den neuen Namen um.)

if __name__ == "__main__":  # Check if the script is being run directly (English: This checks if the script is being executed directly rather than imported as a module. / German: Dies überprüft, ob das Skript direkt ausgeführt wird, anstatt als Modul importiert zu werden.)

    parser = argparse.ArgumentParser(  # Create a command-line argument parser (English: This initializes the argument parser to handle input arguments. / German: Dies initialisiert den Argumentenparser, um Eingabeargumente zu verarbeiten.)
        description="Rename dataset to be consistent with V5 naming convention."  # Add a description for the script (English: This provides a description for the command-line help. / German: Dies fügt eine Beschreibung für die Kommandozeilenhilfe hinzu.)
    )

    parser.add_argument(  # Define a command-line argument for the dataset directory
        "--dataset_dir",  # The argument name
        type=str,  # Specify the type of the argument
        required=True,  # Make this argument mandatory
        help="Path to the dataset dir",  # Provide a help message for the argument
    )

    args = parser.parse_args()  # Parse the command-line arguments (English: This reads and processes the command-line arguments. / German: Dies liest und verarbeitet die Kommandozeilenargumente.)

    rename_dataset(args.dataset_dir)  # Call the renaming function with the provided directory (English: This calls the 'rename_dataset' function with the directory argument provided by the user. / German: Dies ruft die Funktion 'rename_dataset' mit dem vom Benutzer angegebenen Verzeichnisargument auf.)
