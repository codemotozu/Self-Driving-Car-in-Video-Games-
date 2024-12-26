from model import Tedd1104ModelPL  # Importing the TEDD1104 model class from the model module. / Importieren der TEDD1104 Modellklasse aus dem Modell-Modul.
from keyboard.getkeys import key_check  # Importing the key_check function to detect key presses. / Importieren der Funktion key_check zum Erkennen von Tastendrücken.
import argparse  # Importing the argparse module for command-line argument parsing. / Importieren des argparse-Moduls zur Verarbeitung von Befehlszeilenargumenten.
from screen.screen_recorder import ImageSequencer  # Importing the ImageSequencer for managing screen capture sequences. / Importieren des ImageSequencers zum Verwalten von Bildschirmaufnahme-Sequenzen.
import torch  # Importing PyTorch for model execution. / Importieren von PyTorch zur Ausführung des Modells.
import logging  # Importing logging for creating log messages. / Importieren von logging zum Erstellen von Log-Nachrichten.
import time  # Importing time for handling time-related functions. / Importieren von time zur Handhabung zeitbezogener Funktionen.
from tkinter import *  # Importing all elements from the tkinter library for GUI creation. / Importieren aller Elemente aus der tkinter-Bibliothek zur Erstellung von GUIs.
import numpy as np  # Importing numpy for numerical operations, especially arrays. / Importieren von numpy für numerische Operationen, besonders mit Arrays.
import cv2  # Importing OpenCV for computer vision tasks. / Importieren von OpenCV für Aufgaben der Computer Vision.
from torchvision import transforms  # Importing transforms from torchvision for image pre-processing. / Importieren von transforms aus torchvision zur Vorverarbeitung von Bildern.
from utils import mse  # Importing the mean squared error (mse) function from utils. / Importieren der mittleren quadratischen Fehlerfunktion (mse) aus utils.
from keyboard.inputsHandler import select_key  # Importing the select_key function for handling keyboard inputs. / Importieren der Funktion select_key zur Handhabung von Tastatureingaben.
from keyboard.getkeys import id_to_key  # Importing the id_to_key function to map key IDs to their actual key names. / Importieren der Funktion id_to_key zur Zuordnung von Tasten-IDs zu den tatsächlichen Tastenbezeichnern.
import math  # Importing the math module for mathematical functions. / Importieren des math-Moduls für mathematische Funktionen.

from typing import Optional  # Importing Optional for type hinting, allowing optional arguments. / Importieren von Optional für die Typisierung, um optionale Argumente zu ermöglichen.

try:  # Attempting to import Xbox controller emulation. / Versuch, die Xbox-Controller-Emulation zu importieren.
    from controller.xbox_controller_emulator import XboxControllerEmulator  # Importing the Xbox controller emulator class. / Importieren der Xbox-Controller-Emulator-Klasse.
    
    _controller_available = True  # Setting a flag to indicate that the controller emulator is available. / Festlegen eines Flags, das anzeigt, dass der Controller-Emulator verfügbar ist.
except ImportError:  # Handling the case where the Xbox controller emulator cannot be imported. / Umgang mit dem Fall, dass der Xbox-Controller-Emulator nicht importiert werden kann.
    _controller_available = False  # Setting the flag to False when the emulator is unavailable. / Setzen des Flags auf False, wenn der Emulator nicht verfügbar ist.
    XboxControllerEmulator = None  # Setting the class to None when the emulator is unavailable. / Setzen der Klasse auf None, wenn der Emulator nicht verfügbar ist.
    print(  # Printing a warning message to the user. / Ausgeben einer Warnmeldung an den Benutzer.
        f"[WARNING!] Controller emulation unavailable, see controller/setup.md for more info. "
        f"You can ignore this warning if you will use the keyboard as controller for TEDD1104."
    )

if torch.cuda.is_available():  # Checking if a CUDA-compatible GPU is available. / Überprüfen, ob eine CUDA-kompatible GPU verfügbar ist.
    device = torch.device("cuda:0")  # Setting the device to the first GPU. / Festlegen des Geräts auf die erste GPU.
else:  # If no GPU is found, defaulting to CPU. / Wenn keine GPU gefunden wird, Standard auf CPU setzen.
    device = torch.device("cpu")  # Setting the device to CPU. / Festlegen des Geräts auf CPU.
    logging.warning("GPU not found, using CPU, inference will be very slow.")  # Logging a warning about the slower performance on CPU. / Protokollieren einer Warnung über die langsamere Leistung auf der CPU.

def run_ted1104(  # Defining the function that will run the TEDD1104 model. / Definieren der Funktion, die das TEDD1104-Modell ausführt.
    checkpoint_path: str,  # Path to the model checkpoint. / Pfad zum Modell-Checkpoint.
    enable_evasion: bool,  # Flag to enable evasion behavior. / Flag zum Aktivieren des Ausweichverhaltens.
    show_current_control: bool,  # Flag to show current control method (AI or user). / Flag zum Anzeigen der aktuellen Steuerungsmethode (KI oder Benutzer).
    num_parallel_sequences: int = 2,  # Number of parallel sequences to process. / Anzahl der parallel zu verarbeitenden Sequenzen.
    width: int = 1600,  # Width of the game window. / Breite des Spiel-Fensters.
    height: int = 900,  # Height of the game window. / Höhe des Spiel-Fensters.
    full_screen: bool = False,  # Whether the game is running in full screen. / Ob das Spiel im Vollbildmodus läuft.
    evasion_score=1000,  # Score threshold for triggering evasion. / Punkteschwelle für das Auslösen des Ausweichens.
    control_mode: str = "keyboard",  # Control method: "keyboard" or "controller". / Steuerungsmethode: "keyboard" oder "controller".
    enable_segmentation: str = False,  # Flag to enable segmentation feature. / Flag zum Aktivieren der Segmentierungsfunktion.
    dtype=torch.float32,  # Data type for the model (e.g., float32). / Datentyp für das Modell (z.B. float32).
) -> None:  # Function returns nothing. / Die Funktion gibt nichts zurück.
    """
    Run TEDD1104 model in Real-Time inference
    Run TEDD1104-Modell in Echtzeit-Inferenz
    """
    assert control_mode in [  # Ensuring control_mode is either 'keyboard' or 'controller'. / Sicherstellen, dass control_mode entweder 'keyboard' oder 'controller' ist.
        "keyboard",
        "controller",
    ], f"{control_mode} control mode not supported. Supported dataset types: [keyboard, controller].  "  # Raise an error if control_mode is invalid. / Fehler auslösen, wenn control_mode ungültig ist.

    if control_mode == "controller" and not _controller_available:  # Check if controller mode is selected but the emulator is unavailable. / Überprüfen, ob der Controller-Modus ausgewählt wurde, aber der Emulator nicht verfügbar ist.
        raise ModuleNotFoundError(  # Raising an error if the controller emulator is not available. / Fehler auslösen, wenn der Controller-Emulator nicht verfügbar ist.
            f"Controller emulation not available see controller/setup.md for more info."
        )

    show_what_ai_sees: bool = False  # Variable for controlling if AI's view should be displayed. / Variable zum Steuern, ob die Sicht der KI angezeigt werden soll.
    fp16: bool  # Variable for controlling if 16-bit precision should be used. / Variable zum Steuern, ob 16-Bit-Präzision verwendet werden soll.

    model = Tedd1104ModelPL.load_from_checkpoint(  # Loading the model from a checkpoint file. / Laden des Modells aus einer Checkpoint-Datei.
        checkpoint_path=checkpoint_path  # Using the checkpoint path provided. / Verwendung des bereitgestellten Checkpoint-Pfads.
    )

    model.eval()  # Setting the model to evaluation mode. / Setzen des Modells in den Evaluierungsmodus.
    model.to(dtype=dtype, device=device)  # Moving the model to the specified device and dtype. / Verschieben des Modells auf das angegebene Gerät und den Datentyp.

    image_segformer = None  # Initializing the segmentation variable as None. / Initialisieren der Segmentierungs-Variable als None.
    if enable_segmentation:  # If segmentation is enabled, load the segmentation model. / Wenn Segmentierung aktiviert ist, lade das Segmentierungsmodell.
        from segmentation.segmentation_segformer import ImageSegmentation  # Importing the ImageSegmentation class. / Importieren der ImageSegmentation-Klasse.
        image_segformer = ImageSegmentation(device=device)  # Initializing the segmentation model. / Initialisieren des Segmentierungsmodells.

    if control_mode == "controller":  # If controller mode is selected, initialize the Xbox controller emulator. / Wenn der Controller-Modus ausgewählt ist, initialisiere den Xbox-Controller-Emulator.
        xbox_controller: Optional[XboxControllerEmulator] = XboxControllerEmulator()  # Initialize the controller emulator if needed. / Initialisieren des Controller-Emulators, falls erforderlich.
    else:  # If keyboard mode is selected, do nothing related to controller. / Wenn der Tastatur-Modus ausgewählt ist, nichts in Bezug auf den Controller tun.
        xbox_controller = None  # Setting controller to None when using keyboard. / Setzen des Controllers auf None bei Verwendung der Tastatur.



    transform = transforms.Compose(  # Defining a sequence of image transformations for preprocessing. / Definieren einer Reihe von Bildtransformationen zur Vorverarbeitung.
        [
            transforms.ToTensor(),  # Convert images to tensor. / Konvertieren der Bilder in Tensoren.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image with mean and standard deviation. / Normalisieren des Bildes mit Mittelwert und Standardabweichung.
        ]
    )

    img_sequencer = ImageSequencer(  # Creating an ImageSequencer for capturing screen sequences. / Erstellen eines ImageSequencers für das Aufnehmen von Bildschirmsequenzen.
        width=width,  # Setting the width of the game window. / Festlegen der Breite des Spiel-Fensters.
        height=height,  # Setting the height of the game window. / Festlegen der Höhe des Spiel-Fensters.
        full_screen=full_screen,  # Whether to capture in full screen. / Ob das Bild im Vollbildmodus erfasst werden soll.
        get_controller_input=False,  # Not getting controller input here. / Hier keine Controller-Eingaben abrufen.
        num_sequences=num_parallel_sequences,  # Number of sequences to capture in parallel. / Anzahl der parallel zu erfassenden Sequenzen.
        total_wait_secs=5,  # Waiting time before starting the sequence capture. / Wartezeit, bevor die Sequenzaufnahme startet.
    )

    if show_current_control:  # If the current control mode is to be displayed, create a Tkinter window. / Wenn der aktuelle Steuerungsmodus angezeigt werden soll, ein Tkinter-Fenster erstellen.
        root = Tk()  # Create a Tkinter root window. / Erstellen eines Tkinter-Hauptfensters.
        var = StringVar()  # Creating a StringVar to hold the text for display. / Erstellen einer StringVar zur Anzeige des Textes.
        var.set("T.E.D.D. 1104 Driving")  # Setting the displayed text. / Festlegen des angezeigten Textes.
        text_label = Label(root, textvariable=var, fg="green", font=("Courier", 44))  # Creating a label to display the text. / Erstellen eines Labels zur Anzeige des Textes.
        text_label.pack()  # Adding the label to the window. / Hinzufügen des Labels zum Fenster.
    else:  # If the control mode is not to be displayed, set the variables to None. / Wenn der Steuerungsmodus nicht angezeigt werden soll, die Variablen auf None setzen.
        root = None
        var = None
        text_label = None

    last_time: float = time.time()  # Storing the current time to track elapsed time. / Speichern der aktuellen Zeit, um die vergangene Zeit zu verfolgen.
    score: np.float = np.float(0)  # Initializing the score variable. / Initialisieren der Punktzahl-Variable.
    last_num: int = 5  # Setting the initial number of sequences. / Festlegen der Anfangszahl der Sequenzen.

    close_app: bool = False  # Flag to control if the application should be closed. / Flag zur Steuerung, ob die Anwendung geschlossen werden soll.
    model_prediction = np.zeros(3 if control_mode == "controller" else 1)  # Initializing the model prediction array. / Initialisieren des Modellvorhersage-Arrays.

    lt: float = 0  # Left trigger initial value. / Anfangswert für den linken Trigger.
    rt: float = 0  # Right trigger initial value. / Anfangswert für den rechten Trigger.
    lx: float = 0  # Left joystick x-axis initial value. / Anfangswert der x-Achse des linken Joysticks.


while not close_app:  # While the app is not closed, keep running the loop / Solange die App nicht geschlossen ist, läuft die Schleife weiter
    try:  # Try to execute the following block of code / Versuche, den folgenden Code auszuführen
        while last_num == img_sequencer.num_sequence:  # Wait until the image sequencer generates a new sequence number / Warte, bis der Bild-Sequencer eine neue Sequenznummer generiert
            time.sleep(0.01)  # Sleep for 0.01 seconds to avoid overloading the CPU / Schlafe für 0,01 Sekunden, um die CPU nicht zu überlasten

        last_num = img_sequencer.num_sequence  # Update last_num with the current image sequence number / Aktualisiere last_num mit der aktuellen Bildsequenznummer
        img_seq, _ = img_sequencer.get_sequence()  # Get the current image sequence from the sequencer / Hole die aktuelle Bildsequenz vom Sequencer

        init_copy_time: float = time.time()  # Record the current time to calculate reaction time later / Nimm die aktuelle Zeit auf, um später die Reaktionszeit zu berechnen

        keys = key_check()  # Check the state of the keys / Überprüfe den Zustand der Tasten
        if "J" not in keys:  # If the "J" key is not pressed, proceed with automated control / Wenn die "J"-Taste nicht gedrückt ist, fahre mit der automatischen Steuerung fort

            x: torch.tensor = torch.stack(  # Stack the image sequence into a tensor for processing / Staple die Bildsequenz zu einem Tensor für die Verarbeitung
                (
                    transform(img_seq[0] / 255.0),  # Transform each image in the sequence to tensor format / Transformiere jedes Bild in der Sequenz in Tensorformat
                    transform(img_seq[1] / 255.0),
                    transform(img_seq[2] / 255.0),
                    transform(img_seq[3] / 255.0),
                    transform(img_seq[4] / 255.0),
                ),
                dim=0,  # Stack them along a new dimension / Staple sie entlang einer neuen Dimension
            ).to(device=device, dtype=dtype)  # Move the tensor to the specified device and dtype / Verschiebe den Tensor auf das angegebene Gerät und den angegebenen Datentyp

            with torch.no_grad():  # Disable gradient calculation to save memory during inference / Deaktiviere die Berechnung von Gradienten, um Speicher während der Inferenz zu sparen
                model_prediction: torch.tensor = (  # Get the model prediction for the image sequence / Hole die Modellvorhersage für die Bildsequenz
                    model(x, output_mode=control_mode, return_best=True)[0]  # Pass the input tensor to the model and get the prediction / Gib den Eingabetensor an das Modell und hole die Vorhersage
                    .cpu()  # Move the prediction to the CPU / Verschiebe die Vorhersage zur CPU
                    .numpy()  # Convert the prediction to a NumPy array / Konvertiere die Vorhersage in ein NumPy-Array
                )

            if control_mode == "controller":  # If control mode is "controller", adjust controller state based on prediction / Wenn der Steuerungsmodus "controller" ist, passe den Zustand des Controllers basierend auf der Vorhersage an

                if model_prediction[1] > 0:  # If the right trigger value is positive, set it / Wenn der Wert des rechten Triggers positiv ist, setze ihn
                    rt = min(1.0, float(model_prediction[1])) * 2 - 1  # Scale the value for the right trigger / Skaliere den Wert für den rechten Trigger
                    lt = -1  # Left trigger is set to the minimum value / Der linke Trigger wird auf den Minimalwert gesetzt
                else:  # If the right trigger value is negative, set it differently / Wenn der Wert des rechten Triggers negativ ist, setze ihn anders
                    rt = -1  # Right trigger is set to the minimum value / Der rechte Trigger wird auf den Minimalwert gesetzt
                    lt = min(1.0, math.fabs(float(model_prediction[1]))) * 2 - 1  # Scale the left trigger value / Skaliere den Wert des linken Triggers

                lx = max(-1.0, min(1.0, float(model_prediction[0])))  # Scale the left joystick value to be within range [-1, 1] / Skaliere den Wert des linken Joysticks auf den Bereich [-1, 1]

                xbox_controller.set_controller_state(  # Set the controller state based on the predictions / Setze den Zustand des Controllers basierend auf den Vorhersagen
                    lx=lx,
                    lt=lt,
                    rt=rt,
                )
            else:  # If not in controller mode, perform other control actions / Wenn nicht im Controller-Modus, führe andere Steuerungsaktionen aus
                select_key(model_prediction)  # Select a key based on the model's prediction / Wähle eine Taste basierend auf der Vorhersage des Modells

            key_push_time: float = time.time()  # Record the time when a key is pressed / Nimm die Zeit auf, wenn eine Taste gedrückt wird

            if show_current_control:  # If showing current control state, update the display / Wenn der aktuelle Steuerungszustand angezeigt wird, aktualisiere die Anzeige
                var.set("T.E.D.D. 1104 Driving")  # Set the display text for driving mode / Setze den Anzeigetext für den Fahrmodus
                text_label.config(fg="green")  # Change the text color to green / Ändere die Textfarbe auf grün
                root.update()  # Update the GUI display / Aktualisiere die GUI-Anzeige

            if enable_evasion:  # If evasion is enabled, check the evasion condition / Wenn die Ausweichfunktion aktiviert ist, überprüfe die Ausweichbedingung
                score = mse(img_seq[0], img_seq[4])  # Calculate the mean squared error between the first and last image in the sequence / Berechne den mittleren quadratischen Fehler zwischen dem ersten und letzten Bild in der Sequenz
                if score < evasion_score:  # If the score is below the threshold, perform evasion maneuver / Wenn der Fehler unter dem Schwellenwert liegt, führe das Ausweichmanöver aus
                    if show_current_control:  # Update the display to show evasion / Aktualisiere die Anzeige, um das Ausweichmanöver zu zeigen
                        var.set("Evasion maneuver")
                        text_label.config(fg="blue")  # Change the text color to blue for evasion / Ändere die Textfarbe auf blau für das Ausweichmanöver
                        root.update()
                    if control_mode == "controller":  # If in controller mode, perform evasive actions with the controller / Wenn im Controller-Modus, führe Ausweichaktionen mit dem Controller aus
                        xbox_controller.set_controller_state(lx=0, lt=1.0, rt=-1.0)  # Set evasive movement on the controller / Setze die Ausweichbewegung auf dem Controller
                        time.sleep(1)  # Wait for a second before taking further actions / Warte eine Sekunde, bevor weitere Aktionen ausgeführt werden
                        if np.random.rand() > 0.5:  # Randomly choose a direction for the evasive action / Wähle zufällig eine Richtung für das Ausweichmanöver
                            xbox_controller.set_controller_state(
                                lx=1.0, lt=0.0, rt=-1.0
                            )  # Move right / Bewege nach rechts
                        else:
                            xbox_controller.set_controller_state(
                                lx=-1.0, lt=0.0, rt=-1.0
                            )  # Move left / Bewege nach links
                        time.sleep(0.2)  # Wait for 0.2 seconds before next action / Warte 0,2 Sekunden bis zur nächsten Aktion
                    else:  # If not in controller mode, select a key for evasion / Wenn nicht im Controller-Modus, wähle eine Taste für das Ausweichmanöver
                        select_key(4)
                        time.sleep(1)
                        if np.random.rand() > 0.5:
                            select_key(6)
                        else:
                            select_key(8)
                        time.sleep(0.2)

                    if show_current_control:  # Update the display after evasion / Aktualisiere die Anzeige nach dem Ausweichmanöver
                        var.set("T.E.D.D. 1104 Driving")
                        text_label.config(fg="green")
                        root.update()

        else:  # If the "J" key is pressed, switch to manual control / Wenn die "J"-Taste gedrückt ist, wechsle in den manuellen Steuerungsmodus
            if show_current_control:
                var.set("Manual Control")  # Show manual control on display / Zeige manuelle Steuerung auf dem Display
                text_label.config(fg="red")  # Change the text color to red for manual control / Ändere die Textfarbe auf rot für die manuelle Steuerung
                root.update()

            if control_mode == "controller":  # If in controller mode, set the controller to the manual state / Wenn im Controller-Modus, setze den Controller auf den manuellen Zustand
                xbox_controller.set_controller_state(lx=0.0, lt=-1, rt=-1.0)

            key_push_time: float =

 0.0  # Reset the key press time for manual control / Setze die Zeit der Tasteneingabe auf 0 für die manuelle Steuerung

        if show_what_ai_sees:  # If the AI's view should be shown, display the images / Wenn die Ansicht der KI gezeigt werden soll, zeige die Bilder an

            if enable_segmentation:  # If segmentation is enabled, apply it to the images / Wenn die Segmentierung aktiviert ist, wende sie auf die Bilder an
                img_seq = image_segformer.add_segmentation(images=img_seq)

            cv2.imshow("window1", img_seq[0])  # Display the first image in the sequence / Zeige das erste Bild in der Sequenz an
            cv2.waitKey(1)  # Wait for 1 ms before showing the next image / Warte 1 ms, bevor das nächste Bild angezeigt wird
            cv2.imshow("window2", img_seq[1])  # Display the second image in the sequence / Zeige das zweite Bild in der Sequenz an
            cv2.waitKey(1)
            cv2.imshow("window3", img_seq[2])  # Display the third image in the sequence / Zeige das dritte Bild in der Sequenz an
            cv2.waitKey(1)
            cv2.imshow("window4", img_seq[3])  # Display the fourth image in the sequence / Zeige das vierte Bild in der Sequenz an
            cv2.waitKey(1)
            cv2.imshow("window5", img_seq[4])  # Display the fifth image in the sequence / Zeige das fünfte Bild in der Sequenz an
            cv2.waitKey(1)

        if "L" in keys:  # If the "L" key is pressed, toggle the display of the AI's input images / Wenn die "L"-Taste gedrückt wird, schalte die Anzeige der Eingabebilder der KI um
            time.sleep(0.1)  # Wait for key release / Warte auf das Loslassen der Taste
            if show_what_ai_sees:
                cv2.destroyAllWindows()  # Close all image windows if they are being shown / Schließe alle Bildfenster, wenn sie angezeigt werden
                show_what_ai_sees = False  # Set flag to false to stop showing images / Setze das Flag auf False, um das Anzeigen von Bildern zu stoppen
            else:
                show_what_ai_sees = True  # Otherwise, enable showing the images / Andernfalls aktiviere das Anzeigen der Bilder

        time_it: float = time.time() - last_time  # Calculate the time taken for the loop iteration / Berechne die benötigte Zeit für die Schleifeniteration

        if control_mode == "controller":  # If in controller mode, display control information / Wenn im Controller-Modus, zeige Steuerinformationen an
            info_message = (
                f"LX: {int(model_prediction[0] * 100)}%"  # Left joystick percentage / Prozentsatz des linken Joysticks
                f"\n LT: {int(lt * 100)}%\n"  # Left trigger percentage / Prozentsatz des linken Triggers
                f"RT: {int(rt * 100)}%"  # Right trigger percentage / Prozentsatz des rechten Triggers
            )
        else:  # If not in controller mode, display key prediction / Wenn nicht im Controller-Modus, zeige die Tasten-Vorhersage an

            info_message = f"Predicted Key: {id_to_key(model_prediction)}"  # Show predicted key / Zeige die vorhergesagte Taste an

        print(  # Output the current status to the console / Gib den aktuellen Status auf der Konsole aus
            f"Recording at {img_sequencer.screen_recorder.fps} FPS\n"
            f"Actions per second {None if time_it == 0 else 1 / time_it}\n"
            f"Reaction time: {round(key_push_time - init_copy_time, 3) if key_push_time > 0 else 0} secs\n"
            f"{info_message}\n"
            f"Difference from img 1 to img 5 {None if not enable_evasion else score}\n"
            f"Push Ctrl + C to exit\n"
            f"Push L to see the input images\n"
            f"Push J to use to use manual control\n",
            end="\r",  # Overwrite the current output line / Überschreibe die aktuelle Ausgabelinie
        )

        last_time = time.time()  # Record the last time the loop ran / Nimm die letzte Zeit auf, zu der die Schleife lief

    except KeyboardInterrupt:  # If the user interrupts the program, stop the loop / Wenn der Benutzer das Programm unterbricht, stoppe die Schleife
        print()  # Print a new line to avoid overwriting the output / Drucke eine neue Zeile, um das Überschreiben der Ausgabe zu vermeiden
        img_sequencer.stop()  # Stop the image sequencer / Stoppe den Bild-Sequencer
        if control_mode == "controller":  # If in controller mode, stop the controller as well / Wenn im Controller-Modus, stoppe auch den Controller
            xbox_controller.stop()
        close_app = True  # Set the close_app flag to True to end the program / Setze das close_app-Flag auf True, um das Programm zu beenden

if __name__ == "__main__":  # Checks if the script is being run directly (not imported as a module). / Überprüft, ob das Skript direkt ausgeführt wird (nicht als Modul importiert).
    parser = argparse.ArgumentParser()  # Creates an argument parser to handle command-line arguments. / Erstellt einen Argumentparser, um Befehlszeilenargumente zu verarbeiten.

    parser.add_argument(  # Adds an argument for the model checkpoint path. / Fügt ein Argument für den Modell-Checkpoint-Pfad hinzu.
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint file.",  # Specifies that the checkpoint file path is required. / Gibt an, dass der Pfad zur Checkpoint-Datei erforderlich ist.
    )

    parser.add_argument("--width", type=int, default=1600, help="Game window width")  # Adds an argument for the game window width, with a default of 1600. / Fügt ein Argument für die Fensterbreite des Spiels hinzu, mit einem Standardwert von 1600.
    parser.add_argument("--height", type=int, default=900, help="Game window height")  # Adds an argument for the game window height, with a default of 900. / Fügt ein Argument für die Fensterhöhe des Spiels hinzu, mit einem Standardwert von 900.

    parser.add_argument(  # Adds an argument to enable evasion behavior if the vehicle gets stuck. / Fügt ein Argument hinzu, um das Ausweichverhalten zu aktivieren, wenn das Fahrzeug stecken bleibt.
        "--enable_evasion",
        action="store_true",  # If set, enables evasion. / Wenn gesetzt, wird das Ausweichen aktiviert.
        help="Enable evasion, if the vehicle gets stuck we will reverse and randomly turn left/right.",  # Describes the evasion behavior. / Beschreibt das Ausweichverhalten.
    )

    parser.add_argument(  # Adds an argument to display whether TEDD or the user is driving. / Fügt ein Argument hinzu, um anzuzeigen, ob TEDD oder der Benutzer fährt.
        "--show_current_control",
        action="store_true",  # If set, the control mode (TEDD or user) will be shown. / Wenn gesetzt, wird der Steuerungsmodus (TEDD oder Benutzer) angezeigt.
        help="Show if TEDD or the user is driving in the screen .",  # Describes the control mode display. / Beschreibt die Anzeige des Steuerungsmodus.
    )

    parser.add_argument(  # Adds an argument to specify the number of parallel sequences for recording. / Fügt ein Argument hinzu, um die Anzahl paralleler Sequenzen für die Aufzeichnung festzulegen.
        "--num_parallel_sequences",
        type=int,
        default=3,  # Sets the default to 3 parallel sequences. / Setzt den Standardwert auf 3 parallele Sequenzen.
        help="number of parallel sequences to record, if the number is higher the model will do more "
        "iterations per second (will push keys more often) provided your GPU is fast enough. "
        "This improves the performance of the model but increases the CPU and RAM usage.",  # Explains how increasing the number of sequences affects performance. / Erklärt, wie die Erhöhung der Anzahl von Sequenzen die Leistung beeinflusst.
    )

    parser.add_argument(  # Adds an argument to specify the threshold to trigger evasion. / Fügt ein Argument hinzu, um den Schwellenwert für das Ausweichen festzulegen.
        "--evasion_score",
        type=float,
        default=200,  # Sets the default evasion score to 200. / Setzt den Standardwert für den Ausweichwert auf 200.
        help="Threshold to trigger the evasion.",  # Describes the threshold for evasion. / Beschreibt den Schwellenwert für das Ausweichen.
    )

    parser.add_argument(  # Adds an argument to choose the control device (keyboard or controller). / Fügt ein Argument hinzu, um das Steuergerät (Tastatur oder Controller) auszuwählen.
        "--control_mode",
        type=str,
        choices=["keyboard", "controller"],
        default="keyboard",  # Sets the default control mode to keyboard. / Setzt den Standardsteuerungsmodus auf Tastatur.
        help="Device that TEDD will use from driving 'keyboard' or 'controller' (xbox controller).",  # Describes the available control devices. / Beschreibt die verfügbaren Steuergeräte.
    )

    parser.add_argument(  # Adds an argument for full-screen mode. / Fügt ein Argument für den Vollbildmodus hinzu.
        "--full_screen",
        action="store_true",  # If set, the game will run in full-screen mode. / Wenn gesetzt, wird das Spiel im Vollbildmodus ausgeführt.
        help="If you are playing in full screen (no window border on top) set this flag",  # Describes the flag for full-screen mode. / Beschreibt das Flag für den Vollbildmodus.
    )

    parser.add_argument(  # Adds an argument for enabling experimental segmentation. / Fügt ein Argument hinzu, um die experimentelle Segmentierung zu aktivieren.
        "--enable_segmentation",
        action="store_true",  # If set, enables the segmentation feature. / Wenn gesetzt, wird die Segmentierungsfunktion aktiviert.
        help="Experimental. Enable segmentation using segformer (It will only apply segmentation"
        "to the images displayed to the user if you push the 'L' key). Requires huggingface transformers to be "
        "installed (https://huggingface.co/docs/transformers/index). Very GPU demanding!",  # Explains the segmentation feature and requirements. / Erklärt die Segmentierungsfunktion und die Anforderungen.
    )

    parser.add_argument(  # Adds an argument for setting the data type used in inference. / Fügt ein Argument hinzu, um den Datentyp für die Inferenz festzulegen.
        "--dtype",
        choices=["32", "16", "bf16"],
        default="32",  # Sets the default data type to FP32. / Setzt den Standard-Datentyp auf FP32.
        help="Use FP32, FP16 or BF16 (bfloat16) for inference. "
        "BF16 requires a GPU with BF16 support (like Volta or Ampere) and Pytorch >= 1.10",  # Describes the available data types and requirements. / Beschreibt die verfügbaren Datentypen und Anforderungen.
    )

    args = parser.parse_args()  # Parses the command-line arguments. / Verarbeitet die Befehlszeilenargumente.

    if args.dtype == "32":  # Checks if the data type is FP32. / Überprüft, ob der Datentyp FP32 ist.
        dtype = torch.float32  # Sets the data type to FP32. / Setzt den Datentyp auf FP32.
    elif args.dtype == "16":  # Checks if the data type is FP16. / Überprüft, ob der Datentyp FP16 ist.
        dtype = torch.float16  # Sets the data type to FP16. / Setzt den Datentyp auf FP16.
    elif args.dtype == "bf16":  # Checks if the data type is BF16. / Überprüft, ob der Datentyp BF16 ist.
        dtype = torch.bfloat16  # Sets the data type to BF16. / Setzt den Datentyp auf BF16.
    else:  # If an invalid data type is provided. / Wenn ein ungültiger Datentyp angegeben ist.
        raise ValueError(f"Invalid dtype {args.dtype}. Choose from 32, 16 or bf16")  # Raises an error if the data type is invalid. / Löst einen Fehler aus, wenn der Datentyp ungültig ist.

    run_ted1104(  # Calls the function to run the TEDD model with the parsed arguments. / Ruft die Funktion auf, um das TEDD-Modell mit den verarbeiteten Argumenten auszuführen.
        checkpoint_path=args.checkpoint_path,  # Passes the checkpoint path. / Gibt den Checkpoint-Pfad weiter.
        width=args.width,  # Passes the game window width. / Gibt die Fensterbreite des Spiels weiter.
        height=args.height,  # Passes the game window height. / Gibt die Fensterhöhe des Spiels weiter.
        full_screen=args.full_screen,  # Passes whether full screen mode is enabled. / Gibt an, ob der Vollbildmodus aktiviert ist.
        enable_evasion=args.enable_evasion,  # Passes whether evasion behavior is enabled. / Gibt an, ob das Ausweichverhalten aktiviert ist.
        show_current_control=args.show_current_control,  # Passes whether to show the current control mode. / Gibt an, ob der aktuelle Steuerungsmodus angezeigt werden soll.
        num_parallel_sequences=args.num_parallel_sequences,  # Passes the number of parallel sequences. / Gibt die Anzahl der parallelen Sequenzen weiter.
        evasion_score=args.evasion_score,  # Passes the evasion score threshold. / Gibt den Schwellenwert für das Ausweichen weiter.
        control_mode=args.control_mode,  # Passes the control mode (keyboard or controller). / Gibt den Steuerungsmodus (Tastatur oder Controller) weiter.
        enable_segmentation=args.enable_segmentation,  # Passes whether segmentation is enabled. / Gibt an, ob die Segmentierung aktiviert ist.
        dtype=dtype,  # Passes the data type for inference. / Gibt den Datentyp für die Inferenz weiter.
    )
