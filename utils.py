import datetime  # Importing the datetime module to work with dates and times.  # Importiert das datetime-Modul, um mit Daten und Zeiten zu arbeiten
from typing import Union  # Importing Union from typing to specify types that can accept multiple possibilities.  # Importiert Union aus typing, um Typen zu definieren, die mehrere Möglichkeiten akzeptieren können
import numpy as np  # Importing NumPy for numerical operations and array handling.  # Importiert NumPy für numerische Operationen und Array-Verarbeitung
import os  # Importing os module for interacting with the operating system (e.g., file handling).  # Importiert das os-Modul für die Interaktion mit dem Betriebssystem (z. B. Dateiverwaltung)
import torch  # Importing PyTorch, a machine learning framework.  # Importiert PyTorch, ein Framework für maschinelles Lernen


def print_message(message: str) -> None:  # Defines a function to print a message with the current time.  # Definiert eine Funktion, die eine Nachricht mit der aktuellen Zeit ausgibt
    """
    Prints a message with the current time.  # Gibt eine Nachricht mit der aktuellen Zeit aus
    :param str message: Message to print  # Parameter: Nachricht zum Drucken
    """
    print(f"<{str(datetime.datetime.now()).split('.')[0]}> {message}")  # Prints the current time and message.  # Gibt die aktuelle Zeit und die Nachricht aus


def mse(image1: np.ndarray, image2: np.ndarray) -> np.float:  # Defines a function to calculate the Mean Squared Error between two images.  # Definiert eine Funktion, um den mittleren quadratischen Fehler zwischen zwei Bildern zu berechnen
    """
    Mean squared error between two images (np.ndarrays).  # Mittlerer quadratischer Fehler zwischen zwei Bildern (np.ndarrays)
    :param np.ndarray image1: First image  # Parameter: Erstes Bild
    :param np.ndarray image2: Second image  # Parameter: Zweites Bild
    :return: Float - Mean squared error  # Rückgabewert: Float - Mittlerer quadratischer Fehler
    """
    err = np.float(np.sum((np.asarray(image1) - np.asarray(image2)) ** 2))  # Calculate squared differences between the two images and sum them.  # Berechnet die quadrierten Differenzen zwischen den beiden Bildern und summiert sie
    err /= np.float(image1.shape[0] * image1.shape[1])  # Divides by the number of elements in the image (height * width).  # Teilt durch die Anzahl der Elemente im Bild (Höhe * Breite)
    return err  # Returns the mean squared error.  # Gibt den mittleren quadratischen Fehler zurück


def length_normalize(  # Defines a function to normalize the length of a matrix.  # Definiert eine Funktion, um die Länge einer Matrix zu normalisieren
    matrix: np.ndarray,  # Parameter: Matrix zur Normalisierung
) -> np.ndarray:  # Return type: Normalized matrix  # Rückgabewert: Normalisierte Matrix
    """
    Normalizes the length of a matrix.  # Normalisiert die Länge einer Matrix
    :param np.ndarray matrix: Matrix to normalize  # Parameter: Matrix zur Normalisierung
    :return: np.ndarray - Normalized matrix  # Rückgabewert: Normalisierte Matrix
    """
    norms = np.sqrt(np.sum(matrix ** 2, axis=1))  # Calculates the Euclidean norm (length) of each row in the matrix.  # Berechnet die euklidische Norm (Länge) jeder Zeile in der Matrix
    norms[norms == 0] = 1  # Prevents division by zero by setting zero norms to 1.  # Verhindert eine Division durch Null, indem Null-Normen auf 1 gesetzt werden
    return matrix / norms[:, np.newaxis]  # Divides each row by its norm to normalize the matrix.  # Teilt jede Zeile durch ihre Norm, um die Matrix zu normalisieren


class IOHandler:  # Defines a class to handle input/output operations.  # Definiert eine Klasse zur Handhabung von Ein-/Ausgabeoperationen
    """
    Class for handling input and output formats. It is used to convert between keyboard input and controller input.  # Klasse zur Handhabung von Ein-/Ausgabeformaten. Sie wird verwendet, um zwischen Tastatureingaben und Controller-Eingaben zu konvertieren.
    It also handles the saving and loading of the data.  # Sie kümmert sich auch um das Speichern und Laden von Daten
    """

 def __init__(self):  # Constructor for initializing the class instance.  # Konstruktor zur Initialisierung der Instanz der Klasse
    """
    INIT  # Initialisierung  # Beschreibung der Initialisierung
    """
    self.keys2controllerMatrix = np.array(  # Defines a matrix that maps keyboard inputs to controller inputs.  # Definiert eine Matrix, die Tastatureingaben auf Controller-Eingaben abbildet
        [  
            [0.0, 0.0],  # Represents no movement on both axes (no input).  # Keine Bewegung auf beiden Achsen (keine Eingabe)
            [-1.0, 0.0],  # Represents left movement on the x-axis (horizontal).  # Repräsentiert die Bewegung nach links auf der x-Achse (horizontal)
            [1.0, 0.0],   # Represents right movement on the x-axis (horizontal).  # Repräsentiert die Bewegung nach rechts auf der x-Achse (horizontal)
            [0.0, 1.0],   # Represents up movement on the y-axis (vertical).  # Repräsentiert die Bewegung nach oben auf der y-Achse (vertikal)
            [0.0, -1.0],  # Represents down movement on the y-axis (vertical).  # Repräsentiert die Bewegung nach unten auf der y-Achse (vertikal)
            [-1.0, 1.0],  # Represents left and up diagonal movement.  # Repräsentiert die diagonale Bewegung nach links und oben
            [-1.0, -1.0], # Represents left and down diagonal movement.  # Repräsentiert die diagonale Bewegung nach links und unten
            [1.0, 1.0],   # Represents right and up diagonal movement.  # Repräsentiert die diagonale Bewegung nach rechts und oben
            [1.0, -1.0],  # Represents right and down diagonal movement.  # Repräsentiert die diagonale Bewegung nach rechts und unten
        ]
    )  # Initializes the matrix for controlling directional movement based on keyboard input.  # Initialisiert die Matrix für die Steuerung der Richtungsbewegung basierend auf der Tastatureingabe


        # self.keys2controllerMatrix_norm = length_normalize(self.keys2controllerMatrix)  # Optionally normalize the matrix (currently commented out).  # Optional die Matrix normalisieren (derzeit auskommentiert)

    def keys2controller(self, keys: int) -> np.ndarray:  # Converts a keyboard input (key) to a controller input (vector).  # Konvertiert eine Tastatureingabe (Taste) in eine Controller-Eingabe (Vektor)
        """
        Converts a keyboard input to a controller input.  # Konvertiert eine Tastatureingabe in eine Controller-Eingabe
        :param int keys: Keyboard input  # Parameter: Tastatureingabe
        :return: np.ndarray [2] - Controller input  # Rückgabewert: np.ndarray [2] - Controller-Eingabe
        """
        return self.keys2controllerMatrix[keys]  # Returns the corresponding controller input vector.  # Gibt den entsprechenden Controller-Eingabe-Vektor zurück

    def controller2keys(self, controller_vector: np.ndarray) -> int:  # Converts a controller input (vector) to a keyboard input (key).  # Konvertiert eine Controller-Eingabe (Vektor) in eine Tastatureingabe (Taste)
        """
        Converts a controller input to a keyboard input.  # Konvertiert eine Controller-Eingabe in eine Tastatureingabe
        :param np.ndarray controller_vector: Controller input [2]  # Parameter: Controller-Eingabe [2]
        :return: int - Keyboard input  # Rückgabewert: int - Tastatureingabe
        """
        return int(  # Finds the index of the closest matching keyboard input.  # Findet den Index der am besten passenden Tastatureingabe
            np.argmin(  # Finds the index of the minimum value (smallest distance) in the sum of squared differences.  # Findet den Index des kleinsten Wertes (kleinste Distanz) in der Summe der quadrierten Differenzen
                np.sum(  # Sums the squared differences between the controller input and each row of the keyboard-controller matrix.  # Summiert die quadrierten Differenzen zwischen der Controller-Eingabe und jeder Zeile der Tastatur-Controller-Matrix
                    (
                        self.keys2controllerMatrix[np.newaxis, :]  # Adds an extra axis to the matrix for broadcasting.  # Fügt der Matrix eine zusätzliche Achse für Broadcasting hinzu
                        - controller_vector[np.newaxis, :][:, np.newaxis]  # Subtracts the controller vector from each row of the matrix.  # Subtrahiert den Controller-Vektor von jeder Zeile der Matrix
                    )
                    ** 2,  # Squares the differences.  # Quadriert die Differenzen
                    -1,  # Summing over the last axis (across each row).  # Summieren über die letzte Achse (über jede Zeile)
                )
            )
        )  # Returns the index (keyboard input) with the smallest distance to the controller input.  # Gibt den Index (Tastatureingabe) mit der kleinsten Distanz zur Controller-Eingabe zurück

def imagename_input_conversion(  # Function definition to convert image name input to a specific output type.  # Funktionsdefinition zur Umwandlung des Bildnamens in einen bestimmten Ausgabe-Typ.
    self, image_name: str, output_type: str  # Method parameters: 'image_name' is a string and 'output_type' is a string.  # Methodenparameter: 'image_name' ist ein String und 'output_type' ist ein String.
) -> Union[int, np.ndarray]:  # Return type: returns either an int or a numpy array.  # Rückgabetyp: gibt entweder einen int oder ein numpy Array zurück.
    """
    Converts an image name to an 'output_type' input  # Description of the function: it converts the image name into the specified output type.  # Beschreibung der Funktion: Sie wandelt den Bildnamen in den angegebenen Ausgabetyp um.
    :param str image_name: Image name  # 'image_name' parameter description.  # Beschreibung des Parameters 'image_name'.
    :param str output_type: Output type: keyboard or controller  # 'output_type' parameter description.  # Beschreibung des Parameters 'output_type'.
    :return: Union[int, np.ndarray] - Output in the specified format  # Return description: returns either an int or a numpy array depending on the output type.  # Rückgabe-Beschreibung: Gibt je nach Ausgabetyp entweder einen int oder ein numpy Array zurück.
    """
    metadata = os.path.basename(image_name)[:-5]  # Extract the base name of the file (without extension), removing the last 5 characters.  # Extrahiert den Basisnamen der Datei (ohne Erweiterung), entfernt die letzten 5 Zeichen.
    header, values = metadata.split("%")  # Splits metadata into header and values at the "%" character.  # Teilt die Metadaten in Header und Werte am "%" Zeichen.
    control_mode = header[0]  # First character of the header determines the control mode.  # Das erste Zeichen des Headers bestimmt den Steuerungsmodus.
    values = values.split("_")  # Splits the values part at underscores into a list.  # Teilt den Werte-Teil bei Unterstrichen in eine Liste auf.

    if control_mode == "controller":  # Checks if the control mode is "controller".  # Überprüft, ob der Steuerungsmodus "Controller" ist.
        input_value: np.ndarray = np.asarray(  # Converts the values into a numpy array.  # Wandelt die Werte in ein numpy Array um.
            [float(x) for x in values[-1].split(",")],  # Splits the last value by commas and converts to floats.  # Teilt den letzten Wert bei Kommas und wandelt ihn in Floats um.
            dtype=np.float32,  # Specifies the data type as float32.  # Gibt den Datentyp als float32 an.
        )

        input_value = np.asarray(  # Creates a new numpy array with modified values.  # Erstellt ein neues numpy Array mit modifizierten Werten.
            [input_value[0], (input_value[2] - input_value[1]) / 2]  # Computes a new value based on the first and the difference between second and third values.  # Berechnet einen neuen Wert basierend auf dem ersten und der Differenz zwischen dem zweiten und dritten Wert.
        )

        if output_type == "controller":  # Checks if the output type is "controller".  # Überprüft, ob der Ausgabetyp "Controller" ist.
            return input_value  # Returns the input value as a numpy array.  # Gibt den Eingabewert als numpy Array zurück.
        elif output_type == "keyboard":  # Checks if the output type is "keyboard".  # Überprüft, ob der Ausgabetyp "Tastatur" ist.
            return self.controller2keys(controller_vector=input_value)  # Converts the controller input to a keyboard output.  # Wandelt die Controller-Eingabe in eine Tastatur-Ausgabe um.
        else:  # If the output type is neither "controller" nor "keyboard".  # Wenn der Ausgabetyp weder "Controller" noch "Tastatur" ist.
            raise ValueError(  # Raises an error if the output type is invalid.  # Löst einen Fehler aus, wenn der Ausgabetyp ungültig ist.
                f"{output_type} output type not supported. Supported outputs: [keyboard,controller]"  # Error message.  # Fehlermeldung.
            )
    else:  # If the control mode is not "controller".  # Wenn der Steuerungsmodus nicht "Controller" ist.
        input_value: int = int(values[-1])  # Converts the last value into an integer.  # Wandelt den letzten Wert in einen Integer um.

        if output_type == "controller":  # Checks if the output type is "controller".  # Überprüft, ob der Ausgabetyp "Controller" ist.
            return self.keys2controller(input_value)  # Converts the keyboard input to a controller output.  # Wandelt die Tastatureingabe in eine Controller-Ausgabe um.
        elif output_type == "keyboard":  # Checks if the output type is "keyboard".  # Überprüft, ob der Ausgabetyp "Tastatur" ist.
            return input_value  # Returns the integer as the keyboard input.  # Gibt den Integer als Tastatureingabe zurück.
        else:  # If the output type is neither "controller" nor "keyboard".  # Wenn der Ausgabetyp weder "Controller" noch "Tastatur" ist.
            raise ValueError(  # Raises an error if the output type is invalid.  # Löst einen Fehler aus, wenn der Ausgabetyp ungültig ist.
                f"{output_type} output type not supported. Supported outputs: [keyboard,controller]"  # Error message.  # Fehlermeldung.
            )

def input_conversion(  # Function definition to convert input value based on the output type.  # Funktionsdefinition zur Umwandlung des Eingabewerts basierend auf dem Ausgabetyp.
    self, input_value: Union[int, np.ndarray], output_type: str  # Method parameters: input_value is either an int or a numpy array, and output_type is a string.  # Methodenparameter: input_value ist entweder ein int oder ein numpy Array, und output_type ist ein String.
) -> Union[int, np.ndarray]:  # Return type: returns either an int or a numpy array.  # Rückgabetyp: gibt entweder einen int oder ein numpy Array zurück.
    """
    Converts an input to an 'output_type' input  # Description of the function: it converts the input to the specified output type.  # Beschreibung der Funktion: Sie wandelt den Eingabewert in den angegebenen Ausgabetyp um.
    :param Union[int, np.ndarray] input_value: Input value  # 'input_value' parameter description.  # Beschreibung des Parameters 'input_value'.
    :param str output_type: Output type: keyboard or controller  # 'output_type' parameter description.  # Beschreibung des Parameters 'output_type'.
    :return: Union[int, np.ndarray] - Output in the specified format  # Return description: returns either an int or a numpy array depending on the output type.  # Rückgabe-Beschreibung: Gibt je nach Ausgabetyp entweder einen int oder ein numpy Array zurück.
    """
    if type(input_value) == int or input_value.size == 1:  # Checks if the input value is an integer or a single element array.  # Überprüft, ob der Eingabewert ein Integer oder ein einzelelementiges Array ist.
        if output_type == "controller":  # Checks if the output type is "controller".  # Überprüft, ob der Ausgabetyp "Controller" ist.
            return self.keys2controller(int(input_value))  # Converts the keyboard input to a controller output.  # Wandelt die Tastatureingabe in eine Controller-Ausgabe um.
        elif output_type == "keyboard":  # Checks if the output type is "keyboard".  # Überprüft, ob der Ausgabetyp "Tastatur" ist.
            return int(input_value)  # Returns the input value as an integer for keyboard input.  # Gibt den Eingabewert als Integer für Tastatureingabe zurück.
        else:  # If the output type is neither "controller" nor "keyboard".  # Wenn der Ausgabetyp weder "Controller" noch "Tastatur" ist.
            raise ValueError(  # Raises an error if the output type is invalid.  # Löst einen Fehler aus, wenn der Ausgabetyp ungültig ist.
                f"{output_type} output type not supported. Supported outputs: [keyboard,controller]"  # Error message.  # Fehlermeldung.
            )
    else:  # If the input value is a numpy array (not a single value).  # Wenn der Eingabewert ein numpy Array ist (nicht ein einzelner Wert).
        if output_type == "controller":  # Checks if the output type is "controller".  # Überprüft, ob der Ausgabetyp "Controller" ist.
            return input_value  # Returns the numpy array as controller input.  # Gibt das numpy Array als Controller-Eingabe zurück.
        elif output_type == "keyboard":  # Checks if the output type is "keyboard".  # Überprüft, ob der Ausgabetyp "Tastatur" ist.
            return self.controller2keys

(controller_vector=input_value)  # Converts the controller input to a keyboard output.  # Wandelt die Controller-Eingabe in eine Tastatur-Ausgabe um.
        else:  # If the output type is neither "controller" nor "keyboard".  # Wenn der Ausgabetyp weder "Controller" noch "Tastatur" ist.
            raise ValueError(  # Raises an error if the output type is invalid.  # Löst einen Fehler aus, wenn der Ausgabetyp ungültig ist.
                f"{output_type} output type not supported. Supported outputs: [keyboard,controller]"  # Error message.  # Fehlermeldung.
            )

def get_mask(  # Function to create a mask based on certain conditions.  # Funktion zur Erstellung einer Maske basierend auf bestimmten Bedingungen.
    train: bool,  # 'train' parameter determines if the mask is generated for training.  # Der Parameter 'train' bestimmt, ob die Maske für das Training generiert wird.
    nheads: int,  # Number of heads for the mask.  # Anzahl der Köpfe für die Maske.
    mask_prob: float = 0.0,  # Probability of masking.  # Wahrscheinlichkeit der Maskierung.
    sequence_length: int = 5,  # Length of the sequence.  # Länge der Sequenz.
) -> torch.tensor:  # Return type: returns a torch tensor representing the mask.  # Rückgabetyp: Gibt einen Torch-Tensor zurück, der die Maske darstellt.
    if train:  # Checks if the mask is for training.  # Überprüft, ob die Maske für das Training ist.
        bernolli_matrix = torch.cat(  # Concatenates a tensor with the mask probability.  # Verbindet einen Tensor mit der Maskierungswahrscheinlichkeit.
            (
                torch.tensor([0]).float(),  # Tensor with a single element 0.  # Tensor mit einem einzelnen Element 0.
                (torch.tensor([mask_prob]).float()).repeat(sequence_length),  # Repeats the mask probability for the sequence length.  # Wiederholt die Maskierungswahrscheinlichkeit für die Sequenzlänge.
            ),
            0,
        )
        bernolli_distributor = torch.distributions.Bernoulli(bernolli_matrix)  # Creates a Bernoulli distribution with the specified mask probability.  # Erstellt eine Bernoulli-Verteilung mit der angegebenen Maskierungswahrscheinlichkeit.
        sample = bernolli_distributor.sample()  # Samples from the Bernoulli distribution.  # Entnimmt Proben aus der Bernoulli-Verteilung.
        mask = sample > 0  # Creates a mask where the sampled values are greater than 0.  # Erstellt eine Maske, bei der die entnommenen Werte größer als 0 sind.
    else:  # If not for training.  # Wenn nicht für das Training.
        mask = torch.zeros(sequence_length + 1, dtype=torch.bool)  # Creates a mask of zeros for non-training cases.  # Erstellt eine Maske aus Nullen für Nicht-Trainingsfälle.

    mask = mask.repeat(nheads, sequence_length + 1, 1)  # Repeats the mask for the number of heads.  # Wiederholt die Maske für die Anzahl der Köpfe.
    mask.requires_grad = False  # Disables gradient computation for the mask.  # Deaktiviert die Gradientenberechnung für die Maske.
    return mask  # Returns the final mask.  # Gibt die endgültige Maske zurück.
