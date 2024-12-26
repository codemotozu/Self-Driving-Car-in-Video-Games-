"""
Shows the images from the dataset and ask the human to label them.
Requires xv image viewer. Tested on Ubuntu.
""" 
# This docstring explains that the code displays images from a dataset and prompts the human user to label them, using the 'xv' image viewer. It is tested on Ubuntu systems.
# Diese docstring erklärt, dass der Code Bilder aus einem Datensatz anzeigt und den Benutzer auffordert, diese zu kennzeichnen, unter Verwendung des 'xv' Bildbetrachters. Es wurde auf Ubuntu-Systemen getestet.

import glob
# Imports the glob module, which allows for finding all the pathnames matching a specified pattern.
# Importiert das Modul glob, das es ermöglicht, alle Pfadnamen zu finden, die einem bestimmten Muster entsprechen.

import os
# Imports the os module, which provides a way to interact with the operating system, like file and directory manipulation.
# Importiert das Modul os, das eine Möglichkeit bietet, mit dem Betriebssystem zu interagieren, wie z. B. Datei- und Verzeichnismanipulation.

from utils import IOHandler
# Imports the IOHandler class from the utils module, which is used for handling input/output operations.
# Importiert die Klasse IOHandler aus dem Modul utils, die für die Handhabung von Ein-/Ausgabeoperationen verwendet wird.

import json
# Imports the json module, which is used for reading and writing JSON data.
# Importiert das Modul json, das zum Lesen und Schreiben von JSON-Daten verwendet wird.

from tqdm import tqdm
# Imports the tqdm module, used to show a progress bar in loops.
# Importiert das Modul tqdm, das verwendet wird, um eine Fortschrittsanzeige in Schleifen zu zeigen.

import argparse
# Imports the argparse module, which is used to parse command-line arguments.
# Importiert das Modul argparse, das zum Parsen von Befehlszeilenargumenten verwendet wird.

from keyboard.getkeys import keys_to_id
# Imports the keys_to_id function from the keyboard.getkeys module, which converts keyboard input into IDs.
# Importiert die Funktion keys_to_id aus dem Modul keyboard.getkeys, die Tastatureingaben in IDs umwandelt.

def restore_dictionary(annotation_path: str):
    """
    Restores the dictionary from the annotation file
    """
    if os.path.exists(annotation_path):
        with open(annotation_path, "r") as index_file:
            return json.load(index_file)
    else:
        return {"total": 0, "correct": 0, "human_predictions": {}}
# This function checks if the annotation file exists. If it does, it loads the saved dictionary; if not, it returns a new dictionary with initial values.
# Diese Funktion überprüft, ob die Annotationsdatei existiert. Wenn ja, lädt sie das gespeicherte Wörterbuch; andernfalls gibt sie ein neues Wörterbuch mit Anfangswerten zurück.

def human_baseline(gold_dataset_dir: str, annotation_path: str):
    """
    Shows the images from the dataset and ask the human to label them.
    :param str gold_dataset_dir: The directory of the gold dataset
    :param str annotation_path: The path to the annotation file, if it exists we will resume the labeling session
    """
    files = glob.glob(os.path.join(gold_dataset_dir, "*.jpeg"))
    # Uses glob to get all jpeg files in the gold_dataset_dir.
    # Verwendet glob, um alle jpeg-Dateien im gold_dataset_dir zu erhalten.

    io_handler = IOHandler()
    # Initializes the IOHandler to manage input and output operations.
    # Initialisiert den IOHandler, um Ein- und Ausgabeoperationen zu verwalten.

    input_dictionary = restore_dictionary(annotation_path=annotation_path)
    # Calls restore_dictionary to load or initialize the input dictionary for tracking the labeling progress.
    # Ruft restore_dictionary auf, um das Eingabewörterbuch zu laden oder zu initialisieren, um den Kennzeichnungsfortschritt zu verfolgen.

    try:
        pbar_desc = (
            "-1"
            if input_dictionary["total"] == 0
            else f"Current human accuracy: {round((input_dictionary['correct']/input_dictionary['total'])*100,2)}%"
        )
        # Sets the description of the progress bar to show human labeling accuracy or -1 if no labels exist yet.
        # Legt die Beschreibung der Fortschrittsanzeige fest, um die Genauigkeit der menschlichen Kennzeichnung oder -1 anzuzeigen, wenn noch keine Kennzeichnungen existieren.

        with tqdm(total=len(files) - input_dictionary["total"], desc=pbar_desc) as pbar:
            # Initializes a progress bar to show the labeling progress, starting from the number of unlabelled images.
            # Initialisiert eine Fortschrittsanzeige, um den Kennzeichnungsfortschritt zu zeigen, beginnend mit der Anzahl der nicht gekennzeichneten Bilder.

            for image_name in files:
                metadata = os.path.basename(image_name)[:-5]
                # Extracts the metadata (image number) from the filename.
                # Extrahiert die Metadaten (Bildnummer) aus dem Dateinamen.

                header, values = metadata.split("%")
                image_no = int(header[1:])
                # Splits the metadata and converts the image number to an integer.
                # Teilt die Metadaten und wandelt die Bildnummer in eine ganze Zahl um.

                if image_no not in input_dictionary["human_predictions"]:
                    # If the image has not been labeled yet, proceed to label it.
                    # Wenn das Bild noch nicht gekennzeichnet wurde, fahre fort, es zu kennzeichnen.

                    gold_key = io_handler.imagename_input_conversion(
                        image_name=image_name, output_type="keyboard"
                    )
                    # Converts the image filename to a keyboard input format to compare with user input.
                    # Wandelt den Bilddateinamen in ein Tastatureingabeformat um, um es mit der Benutzereingabe zu vergleichen.

                    os.system(f"xv {image_name} &")
                    # Uses the 'xv' image viewer to display the image.
                    # Verwendet den 'xv' Bildbetrachter, um das Bild anzuzeigen.

                    user_key = keys_to_id(input("Push the keys: "))
                    # Prompts the user to press keys and converts the input to an ID.
                    # Fordert den Benutzer auf, Tasten zu drücken und wandelt die Eingabe in eine ID um.

                    input_dictionary["human_predictions"][image_no] = user_key
                    input_dictionary["total"] += 1
                    # Records the user's keypress for the image and increments the total count.
                    # Zeichnet die Tastenanschläge des Benutzers für das Bild auf und erhöht die Gesamtzahl.

                    if user_key == gold_key:
                        input_dictionary["correct"] += 1
                    # If the user's keypress matches the correct key, increment the correct count.
                    # Wenn der Tastenanschlag des Benutzers mit dem richtigen Schlüssel übereinstimmt, wird die korrekte Zählung erhöht.

                    pbar.update(1)
                    # Updates the progress bar to reflect one more labeled image.
                    # Aktualisiert die Fortschrittsanzeige, um ein weiteres gekennzeichnetes Bild anzuzeigen.

                    pbar.set_description(
                        f"Current human accuracy: {round((input_dictionary['correct']/input_dictionary['total'])*100,2)}%"
                    )
                    # Updates the progress bar description to show the current human labeling accuracy.
                    # Aktualisiert die Beschreibung der Fortschrittsanzeige, um die aktuelle Genauigkeit der menschlichen Kennzeichnung anzuzeigen.

                    if input_dictionary["total"] % 20 == 0:
                        with open(
                            annotation_path, "w+", encoding="utf8"
                        ) as annotation_file:
                            json.dump(input_dictionary, annotation_file)
                        # Every 20 images, save the current labeling progress to the annotation file.
                        # Alle 20 Bilder wird der aktuelle Kennzeichnungsfortschritt in der Annotationsdatei gespeichert.

    except KeyboardInterrupt:
        with open(annotation_path, "w+", encoding="utf8") as annotation_file:
            json.dump(input_dictionary, annotation_file)
        # If the process is interrupted, save the progress before exiting.
        # Wenn der Prozess unterbrochen wird, speichere den Fortschritt, bevor du den Prozess beendest.

    with open(annotation_path, "w+", encoding="utf8") as annotation_file:
        json.dump(input_dictionary, annotation_file)
    # Finally, save the labeling progress to the annotation file after finishing.
    # Speichert abschließend den Kennzeichnungsfortschritt in der Annotationsdatei nach dem Abschluss.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shows the images from the dataset and ask the human to label them."
    )
    # Initializes the argument parser for command-line arguments.
    # Initialisiert den Argumentparser für Befehlszeilenargumente.

    parser.add_argument(
        "--gold_dataset_dir",
        type=str,
        help="The directory of the gold dataset",
    )
    # Adds an argument for specifying the directory of the gold dataset.
    # Fügt ein Argument zum Spezifizieren des Verzeichnisses des Gold-Datensatzes hinzu.

    parser.add_argument(
        "--annotation_path",
        type=str,
        help=" The path to the annotation file, if it exists we will resume the labeling session",
    )
    # Adds an argument for specifying the annotation file path.
    # Fügt ein Argument zum Spezifizieren des Pfads zur Annotationsdatei hinzu.

    args = parser.parse_args()
    # Parses the command-line arguments.
    # Parst die Befehlszeilenargumente.

    human_baseline(
        gold_dataset_dir=args.gold_dataset_dir, annotation_path=args.annotation_path
    )
    # Calls the human_baseline function to start the labeling session with the specified dataset and annotation file.
    # Ruft die Funktion human_baseline auf, um die Kennzeichnungssitzung mit dem angegebenen Datensatz und der Annotationsdatei zu starten.
