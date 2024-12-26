import pygame  # Import the pygame library, which is used for handling joystick inputs and creating games. / Importiere die Pygame-Bibliothek, die für das Verarbeiten von Joystick-Eingaben und das Erstellen von Spielen verwendet wird.
import time  # Import the time library, which is used to pause the execution of the program. / Importiere die Zeit-Bibliothek, die verwendet wird, um die Ausführung des Programms zu pausieren.
import sys  # Import the sys library, which allows interaction with the system (e.g., exiting the program). / Importiere die Sys-Bibliothek, die die Interaktion mit dem System ermöglicht (z. B. das Beenden des Programms).
import logging  # Import the logging library, which is used for logging messages (e.g., warnings). / Importiere die Logging-Bibliothek, die zum Aufzeichnen von Nachrichten (z. B. Warnungen) verwendet wird.

class XboxControllerReader:  # Define a class to read the state of an Xbox controller. / Definiere eine Klasse, die den Zustand eines Xbox-Controllers liest.
    """
    Reads the current state of a Xbox Controller  # Description: Reads the current state of an Xbox controller. / Beschreibt: Liest den aktuellen Zustand eines Xbox-Controllers.
    May work with other similar controllers too  # It can potentially work with other controllers that follow a similar protocol. / Es kann möglicherweise auch mit anderen Controllern arbeiten, die ein ähnliches Protokoll verwenden.

    You need to install pygame to use this class: https://www.pygame.org/wiki/GettingStarted  # To use this class, you must first install pygame. / Um diese Klasse zu verwenden, müssen Sie zuerst Pygame installieren.
    """

    joystick: pygame.joystick  # Declare a variable to hold the joystick object from Pygame. / Deklariere eine Variable, um das Joystick-Objekt von Pygame zu speichern.
    name: str  # Declare a variable to store the name of the joystick. / Deklariere eine Variable, um den Namen des Joysticks zu speichern.
    joystick_id: int  # Declare a variable to store the joystick ID. / Deklariere eine Variable, um die Joystick-ID zu speichern.

    def __init__(self, total_wait_secs: int = 10):  # Initialize the XboxControllerReader class with an optional wait time in seconds. / Initialisiere die Klasse XboxControllerReader mit einer optionalen Wartezeit in Sekunden.
        """
        Init  # Description: Initialization method for setting up the joystick. / Beschreibung: Initialisierungsmethode zum Einrichten des Joysticks.

        - Total_wait_secs: Integer, number of seconds to wait. Pygame take some time to initialize, during the first
        seconds you may get wrong readings for the controller, waiting a few seconds before starting reading
        is recommended.  # Explanation: Wait for a few seconds to ensure the joystick is initialized correctly before reading. / Erklärung: Warte einige Sekunden, um sicherzustellen, dass der Joystick korrekt initialisiert ist, bevor die Eingaben gelesen werden.
        """
        pygame.init()  # Initialize all Pygame modules. / Initialisiere alle Pygame-Module.
        pygame.joystick.init()  # Initialize the joystick module in Pygame. / Initialisiere das Joystick-Modul in Pygame.
        try:  # Try block to attempt to initialize the joystick. / Versuchsblock, um den Joystick zu initialisieren.
            self.joystick = pygame.joystick.Joystick(0)  # Try to get the first joystick connected. / Versuche, den ersten angeschlossenen Joystick zu bekommen.
        except pygame.error:  # If there is an error (no joystick detected), handle the exception. / Wenn ein Fehler auftritt (kein Joystick erkannt), handle die Ausnahme.
            logging.warning(  # Log a warning message if no joystick is found. / Protokolliere eine Warnung, wenn kein Joystick gefunden wird.
                "No controller found, ensure that your controlled is connected and is recognized by windows"  # Detailed warning about controller connection. / Detaillierte Warnung zur Verbindung des Controllers.
            )
            sys.exit()  # Exit the program if no joystick is detected. / Beende das Programm, wenn kein Joystick erkannt wird.

        self.joystick.init()  # Initialize the joystick object after it's found. / Initialisiere das Joystick-Objekt, nachdem es gefunden wurde.
        self.name = self.joystick.get_name()  # Get the name of the joystick. / Hole den Namen des Joysticks.
        self.joystick_id = self.joystick.get_id()  # Get the unique ID of the joystick. / Hole die eindeutige ID des Joysticks.

        for delay in range(int(total_wait_secs), 0, -1):  # Loop to wait for the specified number of seconds. / Schleife, um die angegebene Anzahl an Sekunden zu warten.
            print(  # Print the current countdown to the console. / Drucke den aktuellen Countdown auf die Konsole.
                f"Initializing controller reader, waiting {delay} seconds to prevent wrong readings...",  # Display the remaining wait time. / Zeige die verbleibende Wartezeit an.
                end="\r",  # Keep the output on the same line. / Halte die Ausgabe in derselben Zeile.
            )
            time.sleep(1)  # Pause the program for 1 second between iterations. / Pausiere das Programm für 1 Sekunde zwischen den Iterationen.

        print(f"Recording input from: {self.name} ({self.joystick_id})\n")  # Inform the user which controller is being used. / Informiere den Benutzer, welcher Controller verwendet wird.

    def read(self) -> (float, float, float, float):  # Method to read the current state of the controller. / Methode zum Lesen des aktuellen Zustands des Controllers.
        """
        Reads the current state of the controller  # Description: Reads the state of the joystick. / Beschreibung: Liest den Zustand des Joysticks.

        Input:  # Explanation: The input comes from the joystick's current position. / Erklärung: Die Eingabe kommt von der aktuellen Position des Joysticks.

        Output:  # Explanation: The method returns joystick values for the X-axis and two triggers. / Erklärung: Die Methode gibt die Joystick-Werte für die X-Achse und zwei Trigger zurück.
         -lx: Float, current X value of the right stick in range [-1,1]  # Left stick X value. / Linke Stick-X-Wert.
         -lt: Float, current L value in range [-1,1]  # Left trigger value. / Linker Trigger-Wert.
         -rt: Float, current R value in range [-1,1]  # Right trigger value. / Rechter Trigger-Wert.
        """
        _ = pygame.event.get()  # Process any Pygame events (though not used here). / Verarbeite alle Pygame-Ereignisse (obwohl hier nicht verwendet).
        lx, lt, rt = (  # Get the current X-axis value of the right stick, left trigger, and right trigger. / Hole den aktuellen X-Achsen-Wert des rechten Sticks, linken Trigger und rechten Trigger.
            self.joystick.get_axis(0),  # Get the X-axis value of the right stick. / Hole den X-Achsen-Wert des rechten Sticks.
            self.joystick.get_axis(4),  # Get the left trigger value. / Hole den Wert des linken Triggers.
            self.joystick.get_axis(5),  # Get the right trigger value. / Hole den Wert des rechten Triggers.
        )

        return lx, lt, rt  # Return the joystick values for X, left trigger, and right trigger. / Gib die Joystick-Werte für X, linken Trigger und rechten Trigger zurück.
