from screen.grabber import Grabber  # Import the Grabber class for screen capturing | Importiert die Grabber-Klasse zur Bildschirmaufnahme
from controller.xbox_controller_reader import XboxControllerReader  # Import Xbox controller reader for input handling | Importiert Xbox-Controller-Leser für die Eingabeverarbeitung
import numpy as np  # Import NumPy for numerical operations | Importiert NumPy für numerische Berechnungen
import time  # Import time module for delays and timestamps | Importiert das Zeitmodul für Verzögerungen und Zeitstempel
import cv2  # Import OpenCV for image processing | Importiert OpenCV zur Bildverarbeitung
import threading  # Import threading for parallel execution | Importiert Threading für parallele Ausführung
import logging  # Import logging for warnings and information | Importiert Logging für Warnungen und Informationen
import math  # Import math for mathematical calculations | Importiert Math für mathematische Berechnungen
from typing import Union  # Import Union for type hinting multiple types | Importiert Union zur Typisierung mehrerer Typen
from keyboard.getkeys import key_check, keys_to_id  # Import key-checking utilities for keyboard input | Importiert Funktionen zur Tastenprüfung für Tastatureingaben


def preprocess_image(image):  # Function to preprocess an image | Funktion zur Vorverarbeitung eines Bildes
    """
    Given an image resize it and convert it to a numpy array | Verkleinert ein Bild und konvertiert es in ein NumPy-Array
    """
    processed_image = cv2.resize(image, (480, 270))  # Resize image to 480x270 | Skaliert das Bild auf 480x270
    return np.asarray(processed_image, dtype=np.uint8)  # Convert resized image to uint8 NumPy array | Konvertiert das skalierte Bild in ein uint8-NumPy-Array


class ScreenRecorder:  # Class to handle screen recording | Klasse zur Verwaltung von Bildschirmaufnahmen
    """
    Captures screenshots using ImageGRAB from PIL | Nimmt Screenshots mit ImageGRAB von PIL auf
    """

    fps: int  # Frames per second | Bilder pro Sekunde
    width: int  # Width of the screen capture | Breite der Bildschirmaufnahme
    height: int  # Height of the screen capture | Höhe der Bildschirmaufnahme
    screen_grabber: Grabber  # Screen grabber object | Bildschirm-Greifer-Objekt
    front_buffer: np.ndarray  # Front buffer for image | Frontpuffer für das Bild
    back_buffer: np.ndarray  # Back buffer for image | Backpuffer für das Bild
    get_controller_input: bool  # Flag for capturing controller input | Flag zur Erfassung von Controller-Eingaben
    controller_input: np.ndarray  # Buffer to store controller input | Puffer zur Speicherung von Controller-Eingaben
    controller_reader: XboxControllerReader  # Xbox controller reader | Xbox-Controller-Leser
    img_thread: threading.Thread  # Thread for image capture | Thread zur Bildaufnahme

    def __init__(
        self,
        width: int = 1600,
        height: int = 900,
        full_screen: bool = False,
        get_controller_input: bool = False,
        control_mode: str = "keyboard",
        total_wait_secs: int = 5,
    ):  # Constructor to initialize screen recorder | Konstruktor zur Initialisierung des Screen-Recorders
        """
        INIT

        :param int width: Width of the game window | Breite des Spiel-Fensters
        :param int height:  Height of the game window | Höhe des Spiel-Fensters
        :param bool full_screen: True if the game is in full screen (no window border on top). False if not | True, wenn das Spiel im Vollbildmodus ist (kein Fensterrahmen oben). False, wenn nicht
        :param bool get_controller_input: True if the controller input should be captured | True, wenn die Controller-Eingabe erfasst werden soll
        :param str control_mode: Record the input from the "keyboard" or "controller" | Aufzeichnen der Eingaben von "Tastatur" oder "Controller"
        :param int total_wait_secs: Total time to wait for the controller and image recorder to be ready (in seconds) | Gesamte Wartezeit, um den Controller und den Bildrecorder bereit zu machen (in Sekunden)
        """
        print(f"We will capture a window of W:{width} x H:{height} size")  # Print recording dimensions | Gibt die Aufnahmedimensionen aus

        assert control_mode in [
            "keyboard",
            "controller",
        ], f"Control mode: {control_mode} not supported. Available modes: [keyboard,controller]"  # Validate control mode | Überprüft den Steuerungsmodus

        self.control_mode = control_mode  # Set control mode | Setzt den Steuerungsmodus
        self.width = width  # Set width of capture | Legt die Breite der Aufnahme fest
        self.height = height  # Set height of capture | Legt die Höhe der Aufnahme fest
        self.get_controller_input = get_controller_input  # Enable/disable controller input | Aktiviert/deaktiviert Controller-Eingaben

        if full_screen:  # Check if fullscreen mode is enabled | Überprüft, ob der Vollbildmodus aktiviert ist
            self.screen_grabber = Grabber(bbox=(0, 0, width, height))  # Set grabber for fullscreen | Legt den Grabber für Vollbild fest
        else:
            self.screen_grabber = Grabber(bbox=(1, 26, width + 1, height + 26))  # Adjust for window borders | Passt an Fenstergrenzen an

        self.front_buffer = np.zeros((width, height, 3), dtype=np.int8)  # Initialize front buffer | Initialisiert den Frontpuffer
        self.back_buffer = np.zeros((width, height, 3), dtype=np.int8)  # Initialize back buffer | Initialisiert den Backpuffer

        if get_controller_input:  # If controller input is enabled | Wenn Controller-Eingabe aktiviert ist
            if control_mode == "keyboard":  
                self.controller_input = np.zeros(1, dtype=np.int)  # Initialize input buffer for keyboard | Initialisiert Eingabepuffer für Tastatur
            else:
                self.controller_input = np.zeros(3, dtype=np.float32)  # Initialize input buffer for controller | Initialisiert Eingabepuffer für Controller

        self.stop_recording: threading.Event = threading.Event()  # Create stop recording event | Erstellt das Stoppen-Ereignis für die Aufnahme
        self.img_thread: threading.Thread = threading.Thread(target=self._img_thread, args=[self.stop_recording])  # Create a thread for image capture | Erstellt einen Thread zur Bildaufnahme
        self.img_thread.setDaemon(True)  # Set thread as daemon | Setzt den Thread als Daemon
        self.img_thread.start()  # Start the image capture thread | Startet den Bildaufnahme-Thread

        for delay in range(int(total_wait_secs), 0, -1):  # Wait for initialization | Wartet auf die Initialisierung
            print(f"Initializing image recorder, waiting {delay} seconds to prevent wrong readings...", end="\r")  # Print countdown | Gibt den Countdown aus
            time.sleep(1)  # Wait for 1 second | Wartet 1 Sekunde

    def _img_thread(self, stop_event: threading.Event):  # Thread function to continuously capture screen | Thread-Funktion zur kontinuierlichen Bildschirmaufnahme
        """
        Thread that continuously captures the screen | Thread, der kontinuierlich den Bildschirm aufnimmt

        :param threading.Event stop_event: Event to stop the thread | Event, um den Thread zu stoppen
        """
        if self.get_controller_input and self.control_mode == "controller":  # If controller input is enabled and mode is controller | Wenn Controller-Eingabe aktiviert ist und der Modus "Controller" ist
            self.controller_reader = XboxControllerReader(total_wait_secs=2)  # Initialize controller reader | Initialisiert den Controller-Leser

        while not stop_event.is_set():  # Loop until stop event is triggered | Schleife bis das Stop-Ereignis ausgelöst wird
            last_time = time.time()  # Record current time | Nimmt die aktuelle Zeit auf
            self.front_buffer = self.screen_grabber.grab(None)  # Capture screen into front buffer | Nimmt den Bildschirm in den Frontpuffer auf

            # Swap buffers and handle controller input | Tauscht Puffer und verarbeitet Controller-Eingaben
            self.front_buffer, self.back_buffer, self.controller_input = (
                self.back_buffer,
                self.front_buffer,
                None if not self.get_controller_input else (
                    keys_to_id(key_check())  # Get keyboard input if using keyboard | Holt Tastatureingaben, wenn Tastatur verwendet wird
                    if self.control_mode == "keyboard"  
                    else self.controller_reader.read()  # Get controller input if using controller | Holt Controller-Eingaben, wenn Controller verwendet wird
                ),
            )

            self.fps = int(1.0 / (time.time() - last_time))  # Calculate frames per second | Berechnet die Bilder pro Sekunde

        print("Image capture thread stopped")  # Notify that thread has stopped | Benachrichtigt, dass der Thread gestoppt wurde

    def get_image(self) -> (np.ndarray, Union[np.ndarray, None]):  # Return the last captured image and controller input | Gibt das zuletzt aufgenommene Bild und die Controller-Eingabe zurück
        """
        Returns the last captured image and controller input
        :return: last captured image, controller input | Gibt das zuletzt aufgenommene Bild und die Controller-Eingabe zurück
        """
        return self.back_buffer, self.controller_input
    def stop(self):  # Stops the screen recording and the sequence thread | Stoppt die Bildschirmaufnahme und den Sequenz-Thread
        """
        Stops the screen recording and the sequence thread | Stoppt die Bildschirmaufnahme und den Sequenz-Thread
        """
        self.stop_recording.set()  # Set stop recording event | Setzt das Stoppen-Ereignis für die Aufnahme

class ImageSequencer:  # Class to handle the sequential capture of images | Klasse zur Verwaltung der sequenziellen Aufnahme von Bildern
    """
    Class that sequentially captures images from a screen recorder | Klasse, die Bilder sequenziell von einem Bildschirmrekorder aufnimmt
    """

    screen_recorder: ScreenRecorder  # Screen recorder object | Bildschirmrekorder-Objekt
    num_sequences: int  # Number of sequences to capture | Anzahl der zu erfassenden Sequenzen
    image_sequences: np.ndarray  # Array to store captured images | Array zum Speichern der aufgenommenen Bilder
    input_sequences: np.ndarray  # Array to store controller inputs | Array zum Speichern der Controller-Eingaben
    get_controller_input: bool  # Flag to capture controller input | Flag zur Erfassung von Controller-Eingaben
    capture_rate: float  # Frames per second for capturing | Bilder pro Sekunde für die Aufnahme
    sequence_delay: float  # Delay between each sequence | Verzögerung zwischen den einzelnen Sequenzen
    num_sequence: int  # Counter for total sequence number | Zähler für die Gesamtsequenznummer
    actual_sequence: int  # Counter for current sequence | Zähler für die aktuelle Sequenz

    def __init__(
        self,
        width: int = 1600,
        height: int = 900,
        full_screen: bool = False,
        get_controller_input: bool = False,
        capturerate: float = 10.0,
        num_sequences: int = 1,
        total_wait_secs: int = 10,
        control_mode: str = "keyboard",
    ):  # Constructor to initialize ImageSequencer | Konstruktor zur Initialisierung des ImageSequencers
        """
        INIT

        :param int width: Width of the game window | Breite des Spiel-Fensters
        :param int height: Height of the game window | Höhe des Spiel-Fensters
        :param bool full_screen: True if the game is in full screen (no window border on top). False if not | True, wenn das Spiel im Vollbildmodus ist (kein Fensterrahmen oben). False, wenn nicht
        :param bool get_controller_input: True if the controller input should be captured | True, wenn die Controller-Eingabe erfasst werden soll
        :param float capturerate: The capture rate in frames per second | Die Erfassungsrate in Bildern pro Sekunde
        :param int num_sequences: The number of parallel sequences to capture | Die Anzahl der parallelen Sequenzen, die aufgenommen werden sollen
        :param int total_wait_secs: The total time to wait for the game to fill the sequences (in seconds) | Gesamte Wartezeit, um das Spiel zu füllen und die Sequenzen aufzunehmen (in Sekunden)
        :param str control_mode: Record the input from the "keyboard" or "controller" | Aufzeichnen der Eingaben von "Tastatur" oder "Controller"
        """
        assert control_mode in [
            "keyboard",
            "controller",
        ], f"Control mode: {control_mode} not supported. Available modes: [keyboard,controller]"  # Validate control mode | Überprüft den Steuerungsmodus

        self.screen_recorder = ScreenRecorder(  # Create a screen recorder object | Erstellt ein ScreenRecorder-Objekt
            width=width,
            height=height,
            get_controller_input=get_controller_input,
            control_mode=control_mode,
            total_wait_secs=5,
            full_screen=full_screen,
        )

        self.num_sequences = num_sequences  # Set the number of sequences | Setzt die Anzahl der Sequenzen
        self.image_sequences = np.repeat(  # Initialize array to store image sequences | Initialisiert das Array zur Speicherung der Bildsequenzen
            np.expand_dims(
                np.asarray(
                    [
                        np.zeros((270, 480, 3)),
                        np.zeros((270, 480, 3)),
                        np.zeros((270, 480, 3)),
                        np.zeros((270, 480, 3)),
                        np.zeros((270, 480, 3)),
                    ],
                    dtype=np.uint8,
                ),
                0,
            ),
            num_sequences,
            axis=0,
        )

        self.get_controller_input = get_controller_input  # Store whether to capture controller input | Speichert, ob Controller-Eingaben erfasst werden sollen

        if get_controller_input:  # If controller input is enabled | Wenn Controller-Eingaben aktiviert sind
            if control_mode == "keyboard":
                self.input_sequences = np.repeat(  # Initialize array to store keyboard input sequences | Initialisiert das Array zur Speicherung der Tastatureingabesequenzen
                    np.expand_dims(
                        np.asarray(
                            [
                                np.zeros(1),
                                np.zeros(1),
                                np.zeros(1),
                                np.zeros(1),
                                np.zeros(1),
                            ],
                            dtype=int,
                        ),
                        0,
                    ),
                    num_sequences,
                    axis=0,
                )
            else:
                self.input_sequences = np.repeat(  # Initialize array to store controller input sequences | Initialisiert das Array zur Speicherung der Controller-Eingabesequenzen
                    np.expand_dims(
                        np.asarray(
                            [
                                np.zeros(3),
                                np.zeros(3),
                                np.zeros(3),
                                np.zeros(3),
                                np.zeros(3),
                            ],
                            dtype=np.float32,
                        ),
                        0,
                    ),
                    num_sequences,
                    axis=0,
                )

        self.capture_rate = capturerate  # Set the capture rate | Setzt die Erfassungsrate
        self.sequence_delay: float = 1.0 / capturerate / num_sequences  # Calculate delay between sequences | Berechnet die Verzögerung zwischen den Sequenzen

        self.num_sequence = 0  # Initialize sequence counter | Initialisiert den Sequenzzähler
        self.actual_sequence = 0  # Initialize current sequence counter | Initialisiert den Zähler der aktuellen Sequenz

        self.stop_recording: threading.Event = threading.Event()  # Create stop recording event | Erstellt das Stoppen-Ereignis für die Aufnahme
        self.sequence_thread: threading.Thread = threading.Thread(  # Create a thread for sequence capture | Erstellt einen Thread zur Sequenzaufnahme
            target=self._sequence_thread, args=[self.stop_recording]
        )
        self.sequence_thread.setDaemon(True)  # Set the sequence thread as daemon | Setzt den Sequenz-Thread als Daemon
        self.sequence_thread.start()  # Start the sequence capture thread | Startet den Sequenzaufnahme-Thread

        for delay in range(int(total_wait_secs), 0, -1):  # Wait for initialization | Wartet auf die Initialisierung
            print(f"Initializing image sequencer, waiting {delay} seconds to prevent wrong readings...", end="\r")  # Print countdown | Gibt den Countdown aus
            time.sleep(1)  # Wait for 1 second | Wartet 1 Sekunde

    def _sequence_thread(self, stop_event: threading.Event):  # Function to capture image sequences | Funktion zur Aufnahme von Bildsequenzen
        """
        Thread that continuously captures sequences of images | Thread, der kontinuierlich Bildsequenzen aufnimmt
        :param threading.Event stop_event: Event to stop the thread | Event, um den Thread zu stoppen
        """

        while not stop_event.is_set():  # Loop until stop event is triggered | Schleife bis das Stop-Ereignis ausgelöst wird
            for i in range(self.num_sequences):  # Loop through each sequence | Schleife durch jede Sequenz
                start_time: float = time.time()  # Record the start time for sequence | Nimmt die Startzeit für die Sequenz auf

                image, user_input = np.copy(self.screen_recorder.get_image())  # Get image and input | Holt Bild und Eingabe

                self.image_sequences[i][0] = preprocess_image(image)  # Preprocess image and store in sequence | Vorverarbeitet das Bild und speichert es in der Sequenz
                self.image_sequences[i] = self.image_sequences[i][[1, 2, 3, 4, 0]]  # Shift the image sequence | Verschiebt die Bildsequenz

                if self.get_controller_input:  # If controller input is enabled | Wenn Controller-Eingaben aktiviert sind
                    self.input_sequences[i][0] = user_input  # Store controller input | Speichert die Controller-Eingabe
                    self.input_sequences[i] = self.input_sequences[i][[1, 2, 3, 4, 0]]  # Shift the input sequence | Verschiebt die Eingabesequenz

                self.actual_sequence = i  # Set the current sequence | Setzt die aktuelle Sequenz
                self.num_sequence += 1  # Increment the total sequence number | Erhöht die Gesamtsequenznummer

                wait_time: float = self.sequence_delay - (time.time() - start_time)  # Calculate the remaining wait time | Berechnet die verbleibende Wartezeit
                if wait_time > 0:  # If there is time left, sleep | Wenn noch Zeit übrig ist, warte
                    time.sleep(wait_time)
                else:  # If not, log a warning | Wenn nicht, gebe eine Warnung aus
                    logging.warning(
                        f"{math.fabs(wait_time)} delay in the sequence capture, consider reducing num_sequences"
                    )

        print("Image sequence thread stopped")  # Notify when the thread stops | Benachrichtige, wenn der Thread gestoppt wird

    @property
    def sequence_number(self) -> int:  # Property to get the current sequence number | Eigenschaft, um die aktuelle Sequenznummer zu erhalten
        """
        Returns the current sequence number | Gibt die aktuelle Sequenznummer zurück
        :return: int - current sequence number | Rückgabewert: int - aktuelle Sequenznummer
        """
        return self.num_sequence

    def stop(self):  # Stops the image sequence capture | Stoppt die Bildsequenzaufnahme
        """
        Stops the screen recording and the sequence thread | Stoppt die Bildschirmaufnahme und den Sequenz-Thread
        """
        self.stop_recording.set()  # Stop the recording | Stoppt die Aufnahme
        self.screen_recorder.stop()  # Stop the screen recorder | Stoppt den Bildschirmrekorder

    def get_sequence(self) -> (np.ndarray, Union[np.ndarray, None]):  # Get the last captured sequence | Holt die zuletzt aufgenommene Sequenz
        """
        Return the last sequence and the controller input if requested | Gibt die letzte Sequenz und die Controller-Eingabe zurück, wenn diese angefordert wird
        :return: (np.ndarray, Union[np.ndarray, None]) - last sequence and the controller input if requested | Rückgabewert: (np.ndarray, Union[np.ndarray, None]) - letzte Sequenz und die Controller-Eingabe, wenn angefordert
        """

        return (
            np.copy(self.image_sequences[self.actual_sequence]),  # Return the last image sequence | Gibt die letzte Bildsequenz zurück
            None if not self.get_controller_input else np.copy(self.input_sequences[self.actual_sequence]),  # Return the controller input if enabled | Gibt die Controller-Eingabe zurück, wenn aktiviert
        )
