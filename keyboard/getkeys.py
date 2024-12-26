# getkeys.py
# Citation: Box Of Hats (https://github.com/Box-Of-Hats)

import win32api as wapi  # Imports the win32api module to interact with Windows API, used for key detection / Importiert das win32api-Modul, um mit der Windows-API zu interagieren, wird für die Tastenabfrage verwendet.

keyList = []  # ["\b"]  # Initializes an empty list to store key characters / Initialisiert eine leere Liste, um Tastencodes zu speichern.
for char in "WASDJL":  # "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\": Iterates through the string "WASDJL" to add keys to the list / Iteriert durch den String "WASDJL", um Tasten zur Liste hinzuzufügen.
    keyList.append(char)  # Adds each character in the string "WASDJL" to the keyList / Fügt jedes Zeichen im String "WASDJL" zur keyList hinzu.

def key_check() -> str:  # Defines a function to check which keys are currently pressed / Definiert eine Funktion, die überprüft, welche Tasten derzeit gedrückt sind.
    """
    Checks if any keys are pressed.
    """
    keys = []  # Initializes an empty list to store the keys that are pressed / Initialisiert eine leere Liste, um die gedrückten Tasten zu speichern.
    for key in keyList:  # Loops through each key in the keyList / Schleift durch jede Taste in der keyList.
        if wapi.GetAsyncKeyState(ord(key)):  # Checks if the key is pressed by using the Windows API function / Überprüft, ob die Taste mit der Windows-API-Funktion gedrückt ist.
            keys.append(key)  # If the key is pressed, add it to the keys list / Wenn die Taste gedrückt ist, wird sie zur keys-Liste hinzugefügt.
    return "".join(set(keys))  # Returns the pressed keys as a string / Gibt die gedrückten Tasten als String zurück.

def keys_to_id(keys: str) -> int:  # Defines a function that converts key names to integer IDs / Definiert eine Funktion, die Tastenbezeichner in Ganzzahl-IDs umwandelt.
    """
    Converts a keys name to an integer.

    :param str keys: The keys name.
    :returns:
        int - The key id.
    """
    if keys == "A":  # If the key is 'A', return the ID 1 / Wenn die Taste 'A' ist, gebe die ID 1 zurück.
        return 1
    if keys == "D":  # If the key is 'D', return the ID 2 / Wenn die Taste 'D' ist, gebe die ID 2 zurück.
        return 2
    if keys == "W":  # If the key is 'W', return the ID 3 / Wenn die Taste 'W' ist, gebe die ID 3 zurück.
        return 3
    if keys == "S":  # If the key is 'S', return the ID 4 / Wenn die Taste 'S' ist, gebe die ID 4 zurück.
        return 4
    if keys == "AW" or keys == "WA":  # If the keys are 'AW' or 'WA', return ID 5 / Wenn die Tasten 'AW' oder 'WA' sind, gebe die ID 5 zurück.
        return 5
    if keys == "AS" or keys == "SA":  # If the keys are 'AS' or 'SA', return ID 6 / Wenn die Tasten 'AS' oder 'SA' sind, gebe die ID 6 zurück.
        return 6
    if keys == "DW" or keys == "WD":  # If the keys are 'DW' or 'WD', return ID 7 / Wenn die Tasten 'DW' oder 'WD' sind, gebe die ID 7 zurück.
        return 7
    if keys == "DS" or keys == "SD":  # If the keys are 'DS' or 'SD', return ID 8 / Wenn die Tasten 'DS' oder 'SD' sind, gebe die ID 8 zurück.
        return 8

    return 0  # If the key does not match any of the above, return 0 / Wenn die Taste keine der oben genannten entspricht, gebe 0 zurück.


def id_to_key(key: int) -> str:  # Defines a function that converts a key ID to a key name / Definiert eine Funktion, die eine Tasten-ID in einen Tastenbezeichner umwandelt.
    """
    Converts a key id to a string.

    :param int key: The key id.
    :returns:
        str - The key name.
    """
    if key == 1:  # If the key ID is 1, return 'A' / Wenn die Tasten-ID 1 ist, gebe 'A' zurück.
        return "A"
    if key == 2:  # If the key ID is 2, return 'D' / Wenn die Tasten-ID 2 ist, gebe 'D' zurück.
        return "D"
    if key == 3:  # If the key ID is 3, return 'W' / Wenn die Tasten-ID 3 ist, gebe 'W' zurück.
        return "W"
    if key == 4:  # If the key ID is 4, return 'S' / Wenn die Tasten-ID 4 ist, gebe 'S' zurück.
        return "S"
    if key == 5:  # If the key ID is 5, return 'AW' / Wenn die Tasten-ID 5 ist, gebe 'AW' zurück.
        return "AW"
    if key == 6:  # If the key ID is 6, return 'AS' / Wenn die Tasten-ID 6 ist, gebe 'AS' zurück.
        return "AS"
    if key == 7:  # If the key ID is 7, return 'DW' / Wenn die Tasten-ID 7 ist, gebe 'DW' zurück.
        return "DW"
    if key == 8:  # If the key ID is 8, return 'DS' / Wenn die Tasten-ID 8 ist, gebe 'DS' zurück.
        return "DS"
    return "none"  # If the key ID does not match any above, return 'none' / Wenn die Tasten-ID keine der oben genannten entspricht, gebe 'none' zurück.
