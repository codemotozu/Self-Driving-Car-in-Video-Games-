import os  # Imports the os module to interact with the operating system (file management). / Importiert das os-Modul, um mit dem Betriebssystem zu interagieren (Dateiverwaltung).
import time  # Imports the time module to handle time-related tasks. / Importiert das time-Modul zur Handhabung von zeitbezogenen Aufgaben.
import numpy as np  # Imports numpy for numerical operations, such as arrays and matrix manipulations. / Importiert numpy für numerische Operationen, wie Arrays und Matrixmanipulationen.
import argparse  # Imports argparse to handle command-line arguments. / Importiert argparse, um Kommandozeilenargumente zu verarbeiten.
from screen.screen_recorder import ImageSequencer  # Imports ImageSequencer from the screen.recorder module. / Importiert ImageSequencer aus dem Modul screen.recorder.
import cv2  # Imports OpenCV library for image processing and computer vision tasks. / Importiert die OpenCV-Bibliothek für Bildverarbeitung und Computer-Vision-Aufgaben.
from PIL import Image  # Imports the Image class from the Python Imaging Library (PIL) to handle images. / Importiert die Image-Klasse aus der Python Imaging Library (PIL) zur Bildverarbeitung.
from typing import Union  # Imports Union from typing for type hinting (allowing multiple possible input types). / Importiert Union aus dem Modul typing für Typisierung (erlaubt mehrere mögliche Eingabetypen).
from utils import IOHandler  # Imports IOHandler from the utils module to manage input/output tasks. / Importiert IOHandler aus dem Modul utils zur Verwaltung von Ein-/Ausgabeaufgaben.

class BalancedDataset:  # Defines a class to manage a dataset of balanced classes. / Definiert eine Klasse zur Verwaltung eines Datensatzes mit ausgewogenen Klassen.
    class_matrix: np.ndarray  # A matrix to track class counts. / Eine Matrix zur Verfolgung der Klassenzählungen.
    io_handler: IOHandler  # Instance of IOHandler for managing input/output. / Instanz von IOHandler zur Verwaltung von Ein-/Ausgabeoperationen.
    total: int  # Total number of examples in the dataset. / Die Gesamtzahl der Beispiele im Datensatz.

    def __init__(self):  # Constructor to initialize the dataset. / Konstruktor zur Initialisierung des Datensatzes.
        self.class_matrix = np.zeros(9, dtype=np.int32)  # Initializes a matrix with 9 zeros to track 9 classes. / Initialisiert eine Matrix mit 9 Nullen zur Verfolgung von 9 Klassen.
        self.io_handler = IOHandler()  # Initializes IOHandler for input/output management. / Initialisiert IOHandler für die Verwaltung von Ein-/Ausgabe.
        self.total = 0  # Initializes the total count of examples to 0. / Setzt die Gesamtzahl der Beispiele auf 0.

    def balance_dataset(self, input_value: Union[np.ndarray, int]) -> bool:  # Balances the dataset based on the input value. / Balanciert den Datensatz basierend auf dem Eingabewert.
        example_class = self.io_handler.input_conversion(  # Converts the input value to a class. / Konvertiert den Eingabewert in eine Klasse.
            input_value=input_value, output_type="keyboard"  # Converts input into a keyboard command (can be modified for controller). / Wandelt Eingaben in ein Tastaturkommando um (kann für Controller geändert werden).
        )

        if self.total != 0:  # If the dataset is not empty, calculate probability. / Wenn der Datensatz nicht leer ist, berechne die Wahrscheinlichkeit.
            prop: float = (
                (self.total - self.class_matrix[example_class]) / self.total  # Probability is inversely proportional to class frequency. / Die Wahrscheinlichkeit ist umgekehrt proportional zur Häufigkeit der Klasse.
            ) ** 2  # Square the probability to enhance its effect. / Quadriert die Wahrscheinlichkeit, um ihren Effekt zu verstärken.
            if prop <= 0.7:  # If probability is less than or equal to 0.7, set to 0.1 for balance. / Wenn die Wahrscheinlichkeit kleiner oder gleich 0,7 ist, setze sie auf 0,1 zur Balance.
                prop = 0.1

            if np.random.rand() <= prop:  # Randomly decide if the example should be added based on probability. / Entscheide zufällig, ob das Beispiel basierend auf der Wahrscheinlichkeit hinzugefügt werden soll.
                self.class_matrix[example_class] += 1  # Increment the count of the class. / Erhöhe die Zählung der Klasse.
                self.total += 1  # Increment the total number of examples. / Erhöhe die Gesamtzahl der Beispiele.
                return True  # Return True if the example is added. / Gib True zurück, wenn das Beispiel hinzugefügt wurde.
            else:
                return False  # Otherwise, return False. / Andernfalls gib False zurück.
        else:
            self.class_matrix[example_class] += 1  # If the dataset is empty, simply add the first example. / Wenn der Datensatz leer ist, füge einfach das erste Beispiel hinzu.
            self.total += 1  # Increment the total count. / Erhöhe die Gesamtzahl.
            return True  # Return True indicating the example is added. / Gib True zurück, dass das Beispiel hinzugefügt wurde.

    @property
    def get_matrix(self) -> np.ndarray:  # Property to get the current class matrix. / Property, um die aktuelle Klassenmatrix zu erhalten.
        return self.class_matrix  # Returns the class matrix. / Gibt die Klassenmatrix zurück.


def save_data(  # Function to save data (images and labels) to disk. / Funktion zum Speichern von Daten (Bilder und Labels) auf der Festplatte.
    dir_path: str,  # Directory path to save the data. / Verzeichnispfad zum Speichern der Daten.
    images: np.ndarray,  # Array of images to save. / Array von Bildern, die gespeichert werden sollen.
    y: np.ndarray,  # Labels associated with the images. / Labels, die den Bildern zugeordnet sind.
    number: int,  # Number identifier for the data. / Nummernbezeichner für die Daten.
    control_mode: str = "keyboard",  # Control mode: either "keyboard" or "controller". / Steuerungsmodus: entweder "keyboard" oder "controller".
):
    assert control_mode in [  # Ensure the control mode is either "keyboard" or "controller". / Stelle sicher, dass der Steuerungsmodus entweder "keyboard" oder "controller" ist.
        "keyboard",
        "controller",
    ], f"Control mode: {control_mode} not supported. Available modes: [keyboard,controller]"  # Raise an error if the control mode is invalid. / Wirft einen Fehler, wenn der Steuerungsmodus ungültig ist.

    filename = (  # Generate the filename based on control mode and labels. / Erzeuge den Dateinamen basierend auf dem Steuerungsmodus und den Labels.
        ("K" if control_mode == "keyboard" else "C")  # "K" for keyboard or "C" for controller. / "K" für Tastatur oder "C" für Controller.
        + str(number)  # Adds the number identifier to the filename. / Fügt die Nummer als Bezeichner zum Dateinamen hinzu.
        + "%"
        + "_".join([",".join([str(e) for e in elem]) for elem in y])  # Adds labels as part of the filename. / Fügt Labels als Teil des Dateinamens hinzu.
        + ".jpeg"  # Final file extension. / Endgültige Dateierweiterung.
    )

    Image.fromarray(  # Creates an image from the numpy array. / Erzeugt ein Bild aus dem numpy-Array.
        cv2.cvtColor(np.concatenate(images, axis=1), cv2.COLOR_BGR2RGB)  # Concatenates images and converts the color format. / Verbindet Bilder und konvertiert das Farbschema.
    ).save(os.path.join(dir_path, filename))  # Saves the image to the specified directory. / Speichert das Bild im angegebenen Verzeichnis.


def get_last_file_num(dir_path: str) -> int:  # Function to retrieve the last file number in the directory. / Funktion zum Abrufen der letzten Dateinummer im Verzeichnis.
    files = [  # List all files in the directory that end with ".jpeg". / Listet alle Dateien im Verzeichnis auf, die mit ".jpeg" enden.
        int(f.split("%")[0][1:])  # Extract the file number from the filename. / Extrahiert die Dateinummer aus dem Dateinamen.
        for f in os.listdir(dir_path)  # Loops through files in the directory. / Schleift durch die Dateien im Verzeichnis.
        if os.path.isfile(os.path.join(dir_path, f)) and f.endswith(".jpeg")  # Filters only ".jpeg" files. / Filtert nur ".jpeg"-Dateien.
    ]

    return -1 if len(files) == 0 else max(files)  # Returns the last file number or -1 if no files are found. / Gibt die letzte Dateinummer zurück oder -1, wenn keine Dateien gefunden werden.


def generate_dataset(  # Function to generate a dataset based on screen recording and user input. / Funktion zum Erstellen eines Datensatzes basierend auf Bildschirmaufnahmen und Benutzereingaben.
    output_dir: str,  # Directory to save the generated dataset. / Verzeichnis zum Speichern des erstellten Datensatzes.
    width: int = 1600,  # The width of the game window. / Die Breite des Spielfensters.
    height: int = 900,  # The height of the game window. / Die Höhe des Spielfensters.
    full_screen: bool = False,  # Whether the game is in full-screen mode. / Ob das Spiel im Vollbildmodus ist.
    max_examples_per_second: int = 4,  # Maximum number of examples to capture per second. / Maximale Anzahl von Beispielen, die pro Sekunde erfasst werden.
    use_probability: bool = True,  # Whether to use probability to balance the dataset. / Ob Wahrscheinlichkeit verwendet wird, um den Datensatz auszugleichen.
    control_mode: str = "keyboard",  # Type of user input: "keyboard" or "controller". / Art der Benutzereingabe: "keyboard" oder "controller".
) -> None:  # Function does not return anything. / Die Funktion gibt nichts zurück.


# Assert that the control_mode is either "keyboard" or "controller"
assert control_mode in [
    "keyboard",  # "keyboard" is a valid control mode
    "controller",  # "controller" is also a valid control mode
], f"Control mode: {control_mode} not supported. Available modes: [keyboard,controller]"  # Raise an error if the control_mode is neither of the above

# Überprüfen, ob das Verzeichnis existiert. Wenn nicht, wird es erstellt.
if not os.path.exists(output_dir):  # Check if the output directory exists
    print(f"{output_dir} does not exits. We will create it.")  # Print a message if the directory does not exist
    os.makedirs(output_dir)  # Create the directory if it doesn't exist

# Create an ImageSequencer object with specific settings for the game window and controls
img_sequencer = ImageSequencer(  # Create an ImageSequencer to manage image sequences
    width=width,  # Set the width of the image sequence
    height=height,  # Set the height of the image sequence
    get_controller_input=True,  # Specify that controller input should be tracked
    control_mode=control_mode,  # Set the control mode ("keyboard" or "controller")
    full_screen=full_screen,  # Set whether the game should run in full-screen mode
)

# Define the data_balancer as a type that can either be a BalancedDataset or None
data_balancer: Union[BalancedDataset, None]  # Define the type of data_balancer
if use_probability:  # Check if probability balancing is enabled
    data_balancer = BalancedDataset()  # Create a new BalancedDataset if probability balancing is used
else:
    data_balancer = None  # Otherwise, set data_balancer to None

# Initialize the number of files by getting the last file number and incrementing by 1
number_of_files: int = get_last_file_num(output_dir) + 1  # Get the number of files in the output directory
last_num: int = 5  # The image sequence starts with images containing zeros, wait until it's filled

# Flag to control whether the application should close
close_app: bool = False  # Flag to control the closing of the application

# Main loop that runs until close_app is set to True
while not close_app:  # While the application is not closed
    try:
        start_time: float = time.time()  # Record the start time of the loop
        while last_num == img_sequencer.num_sequence:  # Wait until the image sequence is different from the last
            time.sleep(0.01)  # Sleep for 10 milliseconds to prevent overloading the CPU

        last_num = img_sequencer.num_sequence  # Update last_num with the current sequence number
        img_seq, controller_input = img_sequencer.get_sequence()  # Get the current image sequence and controller input

        # If probability balancing is not used or the dataset is balanced, save the data
        if not use_probability or data_balancer.balance_dataset(
            input_value=controller_input[-1]  # Balance dataset using the last controller input value
        ):
            save_data(  # Save the data (images and input)
                dir_path=output_dir,  # Directory to save the data
                images=img_seq,  # Images from the sequence
                y=controller_input,  # Controller input values
                number=number_of_files,  # File number
                control_mode=control_mode,  # Control mode
            )

            number_of_files += 1  # Increment the file number after saving

        wait_time: float = (start_time + 1 / max_examples_per_second) - time.time()  # Calculate the wait time to control FPS
        if wait_time > 0:  # If there's still time to wait before capturing the next frame
            time.sleep(wait_time)  # Sleep for the calculated wait time

        # Print status information, including FPS and examples per second
        print(
            f"Recording at {img_sequencer.screen_recorder.fps} FPS\n"  # Current FPS
            f"Examples per second: {round(1/(time.time()-start_time),1)} \n"  # Calculated examples per second
            f"Images in sequence {len(img_seq)}\n"  # Number of images in the current sequence
            f"Training data len {number_of_files} sequences\n"  # Number of training sequences saved
            f"User input: {controller_input[-1]}\n"  # Last user input (controller input)
            f"Examples per class matrix:\n"  # Print matrix of examples per class if balancing is used
            f"{None if not use_probability else data_balancer.get_matrix.T}\n"  # Show the balancing matrix
            f"Push Ctrl + C to exit",  # Instructions to exit the loop
            end="\r",  # Overwrite the last printed line to update the status
        )

    except KeyboardInterrupt:  # Handle manual interruption (Ctrl + C)
        print()  # Print a newline to end the message
        img_sequencer.stop()  # Stop the image sequencer
        close_app: bool = True  # Set the flag to close the application

# The entry point of the script
if __name__ == "__main__":  # Check if this script is being executed directly
    parser = argparse.ArgumentParser(  # Create a command-line argument parser
        description="Generate training data from the game. See the README.md file for more info."  # Description of the script
    )

    parser.add_argument(  # Argument to specify the directory where examples will be saved
        "--save_dir",
        type=str,
        default=os.getcwd(),  # Default to the current working directory
        help="The directory where the examples will be saved.",  # Help message
    )

    parser.add_argument(  # Argument to specify the width of the game window
        "--width", type=int, default=1600, help="The width of the game window."  # Default to 1600 pixels
    )
    parser.add_argument(  # Argument to specify the height of the game window
        "--height", type=int, default=900, help="The height of the game window."  # Default to 900 pixels
    )

    parser.add_argument(  # Argument to enable full-screen mode
        "--full_screen",
        action="store_true",  # This flag enables full-screen mode if set
        help="If the game is played in full screen mode.",  # Help message
    )

    parser.add_argument(  # Argument to specify the maximum examples per second
        "--examples_per_second",
        type=int,
        default=8,  # Default to 8 examples per second
        help="The maximum number of examples per second to capture.",  # Help message
    )

    parser.add_argument(  # Argument to control whether to balance examples per class or not
        "--save_everything",
        action="store_true",  # This flag disables balancing
        help="Do not try to balance the number of examples per class recorded. "
        "Not recommended you will end up with a huge amount of examples, "
        "specially if you set the examples_per_second to a high value.",  # Help message
    )

    parser.add_argument(  # Argument to specify the control mode
        "--control_mode",
        type=str,
        default="keyboard",  # Default to "keyboard"
        choices=["keyboard", "controller"],  # Only allow "keyboard" or "controller" as valid options
        help='Type of the user input: "keyboard" or "controller"',  # Help message
    )

    args = parser.parse_args()  # Parse the command-line arguments

    # Call the function to generate the dataset with the parsed arguments
    generate_dataset(  # Generate training data
        output_dir=args.save_dir,  # Directory to save data
        width=args.width,  # Width of the game window
        height=args.height,  # Height of the game window
        full_screen=args.full_screen,  # Whether the game should be full-screen
        max_examples_per_second=args.examples_per_second,  # Max examples per second
        use_probability=not args.save_everything,  # Use balancing if save_everything is not set
        control_mode=args.control_mode,  # Control mode (keyboard or controller)
    )
