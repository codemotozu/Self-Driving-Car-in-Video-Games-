# direct inputs

# Author: hodka (https://stackoverflow.com/users/3550306/hodka) 
# Source to this solution and code: 
# http://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game 
# http://www.gamespp.com/directx/directInputKeyboardScanCodes.html

import ctypes  # Imports the ctypes library for working with C data types and calling functions in DLLs. / Importiert die ctypes-Bibliothek für die Arbeit mit C-Datentypen und das Aufrufen von Funktionen in DLLs.
import time  # Imports the time library to use for adding delays in the script. / Importiert die Zeitbibliothek, um Verzögerungen im Skript hinzuzufügen.

SendInput = ctypes.windll.user32.SendInput  # Loads the SendInput function from the user32.dll to simulate keyboard and mouse input. / Lädt die Funktion SendInput aus der user32.dll, um Tastatur- und Maus-Eingaben zu simulieren.

W = 0x11  # Hexadecimal key code for the 'W' key. / Hexadezimaler Tastencode für die 'W'-Taste.
A = 0x1E  # Hexadecimal key code for the 'A' key. / Hexadezimaler Tastencode für die 'A'-Taste.
S = 0x1F  # Hexadecimal key code for the 'S' key. / Hexadezimaler Tastencode für die 'S'-Taste.
D = 0x20  # Hexadecimal key code for the 'D' key. / Hexadezimaler Tastencode für die 'D'-Taste.

# C struct redefinitions

PUL = ctypes.POINTER(ctypes.c_ulong)  # Creates a pointer to a ctypes unsigned long type. / Erstellt einen Zeiger auf einen ctypes unsigned long-Typ.

class KeyBdInput(ctypes.Structure):  # Defines a structure to represent keyboard input. / Definiert eine Struktur zur Darstellung von Tastatureingaben.
    _fields_ = [
        ("wVk", ctypes.c_ushort),  # Virtual key code for the key being pressed or released. / Virtueller Tastencode für die gedrückte oder freigegebene Taste.
        ("wScan", ctypes.c_ushort),  # Hardware scan code of the key. / Hardware-Scan-Code der Taste.
        ("dwFlags", ctypes.c_ulong),  # Flags for the input event (e.g., key press or release). / Flags für das Eingabereignis (z. B. Tasten drücken oder loslassen).
        ("time", ctypes.c_ulong),  # Timestamp of the event. / Zeitstempel des Ereignisses.
        ("dwExtraInfo", PUL),  # Additional data associated with the event. / Zusätzliche Daten, die mit dem Ereignis verbunden sind.
    ]

class HardwareInput(ctypes.Structure):  # Defines a structure to represent hardware input events. / Definiert eine Struktur zur Darstellung von Hardware-Eingabeereignissen.
    _fields_ = [
        ("uMsg", ctypes.c_ulong),  # Message ID for the hardware event. / Nachrichten-ID für das Hardware-Ereignis.
        ("wParamL", ctypes.c_short),  # Low word of the parameter for the event. / Niedriges Wort des Parameters für das Ereignis.
        ("wParamH", ctypes.c_ushort),  # High word of the parameter for the event. / Hohes Wort des Parameters für das Ereignis.
    ]

class MouseInput(ctypes.Structure):  # Defines a structure for mouse input events. / Definiert eine Struktur für Maus-Eingabeereignisse.
    _fields_ = [
        ("dx", ctypes.c_long),  # Horizontal movement of the mouse. / Horizontale Bewegung der Maus.
        ("dy", ctypes.c_long),  # Vertical movement of the mouse. / Vertikale Bewegung der Maus.
        ("mouseData", ctypes.c_ulong),  # Additional mouse data (e.g., button press). / Zusätzliche Mausdaten (z. B. Tastenanschläge).
        ("dwFlags", ctypes.c_ulong),  # Flags for the mouse event (e.g., mouse button click). / Flags für das Mausereignis (z. B. Maustastenklick).
        ("time", ctypes.c_ulong),  # Timestamp for the event. / Zeitstempel für das Ereignis.
        ("dwExtraInfo", PUL),  # Extra data associated with the mouse event. / Zusätzliche Daten, die mit dem Mausereignis verbunden sind.
    ]

class Input_I(ctypes.Union):  # Union that holds either keyboard, mouse, or hardware input data. / Union, die entweder Tastatur-, Maus- oder Hardware-Eingabedaten hält.
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]  # Defines fields for the different types of input. / Definiert Felder für die verschiedenen Eingabetypen.

class Input(ctypes.Structure):  # Main structure for an input event. / Hauptstruktur für ein Eingabereignis.
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]  # Defines the type of input (keyboard, mouse, or hardware) and its data. / Definiert den Eingabetyp (Tastatur, Maus oder Hardware) und seine Daten.

# Actual Functions

def PressKey(hexKeyCode):  # Function to simulate a key press. / Funktion zur Simulation eines Tastendrucks.
    extra = ctypes.c_ulong(0)  # Initializes extra information as zero. / Initialisiert zusätzliche Informationen als null.
    ii_ = Input_I()  # Creates an instance of the Input_I union. / Erstellt eine Instanz der Input_I-Union.
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))  # Sets up the keyboard input with the given key code. / Richtet die Tastatureingabe mit dem angegebenen Tastencode ein.
    x = Input(ctypes.c_ulong(1), ii_)  # Creates an Input structure with the type set to keyboard input. / Erstellt eine Input-Struktur mit dem Typ auf Tastatureingabe gesetzt.
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))  # Sends the input event to the system. / Sendet das Eingabereignis an das System.

def ReleaseKey(hexKeyCode):  # Function to simulate a key release. / Funktion zur Simulation des Loslassens einer Taste.
    extra = ctypes.c_ulong(0)  # Initializes extra information as zero. / Initialisiert zusätzliche Informationen als null.
    ii_ = Input_I()  # Creates an instance of the Input_I union. / Erstellt eine Instanz der Input_I-Union.
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))  # Sets up the keyboard input for key release. / Richtet die Tastatureingabe für das Loslassen der Taste ein.
    x = Input(ctypes.c_ulong(1), ii_)  # Creates an Input structure with the type set to keyboard input. / Erstellt eine Input-Struktur mit dem Typ auf Tastatureingabe gesetzt.
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))  # Sends the key release event to the system. / Sendet das Ereignis des Tastenklicks an das System.

if __name__ == "__main__":  # Checks if this script is being run directly. / Überprüft, ob dieses Skript direkt ausgeführt wird.
    PressKey(0x11)  # Simulates pressing the 'W' key (0x11). / Simuliert das Drücken der 'W'-Taste (0x11).
    time.sleep(1)  # Waits for 1 second before releasing the key. / Wartet 1 Sekunde, bevor die Taste losgelassen wird.
    ReleaseKey(0x11)  # Simulates releasing the 'W' key (0x11). / Simuliert das Loslassen der 'W'-Taste (0x11).
    time.sleep(1)  # Waits for 1 second before ending the script. / Wartet 1 Sekunde, bevor das Skript endet.

