# inputsHandler.py  # This is the name of the file, which is used for handling input actions. / Dies ist der Name der Datei, die für die Verarbeitung von Eingabebefehlen verwendet wird.

# Authors: Iker García Ferrero and Eritz Yerga  # This line specifies the authors of the code. / Diese Zeile gibt die Autoren des Codes an.

from keyboard.game_control import ReleaseKey, PressKey  # Importing the functions ReleaseKey and PressKey from the keyboard game_control module. / Importieren der Funktionen ReleaseKey und PressKey aus dem Modul keyboard.game_control.

def noKey() -> None:  # This function is defined to release all keys. / Diese Funktion wird definiert, um alle Tasten freizugeben.
    """
    Release all keys  # Description of the function. / Beschreibung der Funktion.
    """
    ReleaseKey(0x11)  # Releases the key with the virtual key code 0x11. / Gibt die Taste mit dem virtuellen Tastencode 0x11 frei.
    ReleaseKey(0x1E)  # Releases the key with the virtual key code 0x1E. / Gibt die Taste mit dem virtuellen Tastencode 0x1E frei.
    ReleaseKey(0x1F)  # Releases the key with the virtual key code 0x1F. / Gibt die Taste mit dem virtuellen Tastencode 0x1F frei.
    ReleaseKey(0x20)  # Releases the key with the virtual key code 0x20. / Gibt die Taste mit dem virtuellen Tastencode 0x20 frei.

def W() -> None:  # This function is defined to release all keys and press the 'W' key. / Diese Funktion wird definiert, um alle Tasten freizugeben und die Taste 'W' zu drücken.
    """
    Release all keys and push W  # Description of the function. / Beschreibung der Funktion.
    """
    PressKey(0x11)  # Presses the key with the virtual key code 0x11 (W key). / Drückt die Taste mit dem virtuellen Tastencode 0x11 (W-Taste).
    ReleaseKey(0x1E)  # Releases the key with the virtual key code 0x1E. / Gibt die Taste mit dem virtuellen Tastencode 0x1E frei.
    ReleaseKey(0x1F)  # Releases the key with the virtual key code 0x1F. / Gibt die Taste mit dem virtuellen Tastencode 0x1F frei.
    ReleaseKey(0x20)  # Releases the key with the virtual key code 0x20. / Gibt die Taste mit dem virtuellen Tastencode 0x20 frei.

def A() -> None:  # This function is defined to release all keys and press the 'A' key. / Diese Funktion wird definiert, um alle Tasten freizugeben und die Taste 'A' zu drücken.
    """
    Release all keys and push A  # Description of the function. / Beschreibung der Funktion.
    """
    ReleaseKey(0x11)  # Releases the key with the virtual key code 0x11. / Gibt die Taste mit dem virtuellen Tastencode 0x11 frei.
    PressKey(0x1E)  # Presses the key with the virtual key code 0x1E (A key). / Drückt die Taste mit dem virtuellen Tastencode 0x1E (A-Taste).
    ReleaseKey(0x1F)  # Releases the key with the virtual key code 0x1F. / Gibt die Taste mit dem virtuellen Tastencode 0x1F frei.
    ReleaseKey(0x20)  # Releases the key with the virtual key code 0x20. / Gibt die Taste mit dem virtuellen Tastencode 0x20 frei.

def S() -> None:  # This function is defined to release all keys and press the 'S' key. / Diese Funktion wird definiert, um alle Tasten freizugeben und die Taste 'S' zu drücken.
    """
    Release all keys and push S  # Description of the function. / Beschreibung der Funktion.
    """
    ReleaseKey(0x11)  # Releases the key with the virtual key code 0x11. / Gibt die Taste mit dem virtuellen Tastencode 0x11 frei.
    ReleaseKey(0x1E)  # Releases the key with the virtual key code 0x1E. / Gibt die Taste mit dem virtuellen Tastencode 0x1E frei.
    PressKey(0x1F)  # Presses the key with the virtual key code 0x1F (S key). / Drückt die Taste mit dem virtuellen Tastencode 0x1F (S-Taste).
    ReleaseKey(0x20)  # Releases the key with the virtual key code 0x20. / Gibt die Taste mit dem virtuellen Tastencode 0x20 frei.

def D() -> None:  # This function is defined to release all keys and press the 'D' key. / Diese Funktion wird definiert, um alle Tasten freizugeben und die Taste 'D' zu drücken.
    """
    Release all keys and push D  # Description of the function. / Beschreibung der Funktion.
    """
    ReleaseKey(0x11)  # Releases the key with the virtual key code 0x11. / Gibt die Taste mit dem virtuellen Tastencode 0x11 frei.
    ReleaseKey(0x1E)  # Releases the key with the virtual key code 0x1E. / Gibt die Taste mit dem virtuellen Tastencode 0x1E frei.
    ReleaseKey(0x1F)  # Releases the key with the virtual key code 0x1F. / Gibt die Taste mit dem virtuellen Tastencode 0x1F frei.
    PressKey(0x20)  # Presses the key with the virtual key code 0x20 (D key). / Drückt die Taste mit dem virtuellen Tastencode 0x20 (D-Taste).

def WA() -> None:  # This function is defined to release all keys and press 'W' and 'A'. / Diese Funktion wird definiert, um alle Tasten freizugeben und die Tasten 'W' und 'A' zu drücken.
    """
    Release all keys and push W and A  # Description of the function. / Beschreibung der Funktion.
    """
    PressKey(0x11)  # Presses the key with the virtual key code 0x11 (W key). / Drückt die Taste mit dem virtuellen Tastencode 0x11 (W-Taste).
    PressKey(0x1E)  # Presses the key with the virtual key code 0x1E (A key). / Drückt die Taste mit dem virtuellen Tastencode 0x1E (A-Taste).
    ReleaseKey(0x1F)  # Releases the key with the virtual key code 0x1F. / Gibt die Taste mit dem virtuellen Tastencode 0x1F frei.
    ReleaseKey(0x20)  # Releases the key with the virtual key code 0x20. / Gibt die Taste mit dem virtuellen Tastencode 0x20 frei.

def WD() -> None:  # This function is defined to release all keys and press 'W' and 'D'. / Diese Funktion wird definiert, um alle Tasten freizugeben und die Tasten 'W' und 'D' zu drücken.
    """
    Release all keys and push W and D  # Description of the function. / Beschreibung der Funktion.
    """
    PressKey(0x11)  # Presses the key with the virtual key code 0x11 (W key). / Drückt die Taste mit dem virtuellen Tastencode 0x11 (W-Taste).
    ReleaseKey(0x1E)  # Releases the key with the virtual key code 0x1E. / Gibt die Taste mit dem virtuellen Tastencode 0x1E frei.
    ReleaseKey(0x1F)  # Releases the key with the virtual key code 0x1F. / Gibt die Taste mit dem virtuellen Tastencode 0x1F frei.
    PressKey(0x20)  # Presses the key with the virtual key code 0x20 (D key). / Drückt die Taste mit dem virtuellen Tastencode 0x20 (D-Taste).

def SA() -> None:  # This function is defined to release all keys and press 'S' and 'A'. / Diese Funktion wird definiert, um alle Tasten freizugeben und die Tasten 'S' und 'A' zu drücken.
    """
    Release all keys and push S and A  # Description of the function. / Beschreibung der Funktion.
    """
    ReleaseKey(0x11)  # Releases the key with the virtual key code 0x11. / Gibt die Taste mit dem virtuellen Tastencode 0x11 frei.
    PressKey(0x1E)  # Presses the key with the virtual key code 0x1E (A key). / Drückt die Taste mit dem virtuellen Tastencode 0x1E (A-Taste).
    PressKey(0x1F)  # Presses the key with the virtual key code 0x1F (S key). / Drückt die Taste mit dem virtuellen Tastencode 0x1F (S-Taste).
    ReleaseKey(0x20)  # Releases the key with the virtual key code 0x20. / Gibt die Taste mit dem virtuellen Tastencode 0x20 frei.

def SD() -> None:  # This function is defined to release all keys and press 'S' and 'D'. / Diese Funktion wird definiert, um alle Tasten freizugeben und die Tasten 'S' und 'D' zu drücken.
    """
    Release all keys and push S and D  # Description of the function. / Beschreibung der Funktion.
    """
    ReleaseKey(0x11)  # Releases the key with the virtual key code 0x11. / Gibt die Taste mit dem virtuellen Tastencode 0x11 frei.
    ReleaseKey(0x1E)  # Releases the key with the virtual key code 0x1E. / Gibt die Taste mit dem virtuellen Tastencode 0x1E frei.
    PressKey(0x1F)  # Presses the key with the virtual key code 0x1F (S key). / Drückt die Taste mit dem virtuellen Tastencode 0x1F (S-Taste).
    PressKey(0x20)  # Presses the key with the virtual key code 0x20 (D key). / Drückt die Taste mit dem virtuellen Tastencode 0x20 (D-Taste).

def select_key(key: int) -> None:  # This function selects and presses a key based on an integer input. / Diese Funktion wählt und drückt eine Taste basierend auf einer ganzzahligen Eingabe.
    """
    Given a key in integer format, send to windows the virtual key push  # Description of the function. / Beschreibung der Funktion.
    """
    if key == 0:  # Checks if the key value is 0, meaning no key is pressed. / Überprüft, ob der Wert der Taste 0 ist, was bedeutet, dass keine Taste gedrückt wird.
        noKey()  # Calls the noKey function to release all keys. / Ruft die Funktion noKey auf, um alle Tasten freizugeben.
    elif key == 1:  # If the key is 1, the 'A' key is pressed. / Wenn die Taste 1 ist, wird die 'A'-Taste gedrückt.
        A()  # Calls the A function to press the 'A' key. / Ruft die Funktion A auf, um die 'A'-Taste zu drücken.
    elif key == 2:  # If the key is 2, the 'D' key is pressed. / Wenn die Taste 2 ist, wird die 'D'-Taste gedrückt.
        D()  # Calls the D function to press the 'D' key. / Ruft die Funktion D auf, um die 'D'-Taste zu drücken.
    elif key == 3:  # If the key is 3, the 'W' key is pressed. / Wenn die Taste 3 ist, wird die 'W'-Taste gedrückt.
        W()  # Calls the W function to press the 'W' key. / Ruft die Funktion W auf, um die 'W'-Taste zu drücken.
    elif key == 4:  # If the key is 4, the 'S' key is pressed. / Wenn die Taste 4 ist, wird die 'S'-Taste gedrückt.
        S()  # Calls the S function to press the 'S' key. / Ruft die Funktion S auf, um die 'S'-Taste zu drücken.
    elif key == 5:  # If the key is 5, both 'W' and 'A' are pressed. / Wenn die Taste 5 ist, werden sowohl 'W' als auch 'A' gedrückt.
        WA()  # Calls the WA function to press 'W' and 'A' keys. / Ruft die Funktion WA auf, um die Tasten 'W' und 'A' zu drücken.
    elif key == 6:  # If the key is 6, both 'S' and 'A' are pressed. / Wenn die Taste 6 ist, werden sowohl 'S' als auch 'A' gedrückt.
        SA()  # Calls the SA function to press 'S' and 'A' keys. / Ruft die Funktion SA auf, um die Tasten 'S' und 'A' zu drücken.
    elif key == 7:  # If the key is 7, both 'W' and 'D' are pressed. / Wenn die Taste 7 ist, werden sowohl 'W' als auch 'D' gedrückt.
        WD()  # Calls the WD function to press 'W' and 'D' keys. / Ruft die Funktion WD auf, um die Tasten 'W' und 'D' zu drücken.
    elif key == 8:  # If the key is 8, both 'S' and 'D' are pressed. / Wenn die Taste 8 ist, werden sowohl 'S' als auch 'D' gedrückt.
        SD()  # Calls the SD function to press 'S' and 'D' keys. / Ruft die Funktion SD auf, um die Tasten 'S' und 'D' zu drücken.
