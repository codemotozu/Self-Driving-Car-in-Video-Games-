# Grabber.py https://gist.github.com/tzickel/5c2c51ddde7a8f5d87be730046612cd0
# Author: tzickel (https://gist.github.com/tzickel) 
# A port of https://github.com/phoboslab/jsmpeg-vnc/blob/master/source/grabber.c to python
# License information (GPLv3) is here https://github.com/phoboslab/jsmpeg-vnc/blob/master/README.md
# EN: This file implements a screen capture tool in Python.
# DE: Diese Datei implementiert ein Bildschirmaufnahme-Tool in Python.

from ctypes import Structure, c_int, POINTER, WINFUNCTYPE, windll, WinError, sizeof
# EN: Imports from the ctypes library for working with Windows API and data structures.
# DE: Importiert aus der ctypes-Bibliothek zur Arbeit mit Windows-API und Datenstrukturen.

from ctypes.wintypes import (
    BOOL,
    HWND,
    RECT,
    HDC,
    HBITMAP,
    HGDIOBJ,
    DWORD,
    LONG,
    WORD,
    UINT,
    LPVOID,
)
# EN: Additional Windows API types required for interaction with system-level functions.
# DE: Zusätzliche Windows-API-Typen, die für die Interaktion mit Systemfunktionen benötigt werden.

import numpy as np
# EN: Imports NumPy for handling numerical arrays.
# DE: Importiert NumPy zur Handhabung numerischer Arrays.

SRCCOPY = 0x00CC0020
# EN: Constant for copying bits from the source to the destination in BitBlt.
# DE: Konstante zum Kopieren von Bits von der Quelle zum Ziel in BitBlt.

DIB_RGB_COLORS = 0
BI_RGB = 0
# EN: Constants related to bitmap color modes and compression types.
# DE: Konstanten im Zusammenhang mit Bitmap-Farbmodi und Kompressionstypen.

class BITMAPINFOHEADER(Structure):
    _fields_ = [
        ("biSize", DWORD),
        ("biWidth", LONG),
        ("biHeight", LONG),
        ("biPlanes", WORD),
        ("biBitCount", WORD),
        ("biCompression", DWORD),
        ("biSizeImage", DWORD),
        ("biXPelsPerMeter", LONG),
        ("biYPelsPerMeter", LONG),
        ("biClrUsed", DWORD),
        ("biClrImportant", DWORD),
    ]
# EN: Defines the structure for bitmap information headers used in screen capture.
# DE: Definiert die Struktur für Bitmap-Informationsköpfe, die bei der Bildschirmaufnahme verwendet werden.

def err_on_zero_or_null_check(result, func, args):
    if not result:
        raise WinError()
    return args
# EN: Checks for errors in Windows API calls and raises an exception if the result is zero or null.
# DE: Überprüft Windows-API-Aufrufe auf Fehler und löst eine Ausnahme aus, wenn das Ergebnis Null ist.

def quick_win_define(name, output, *args, **kwargs):
    dllname, fname = name.split(".")
    params = kwargs.get("params", None)
    if params:
        params = tuple([(x,) for x in params])
    func = (WINFUNCTYPE(output, *args))((fname, getattr(windll, dllname)), params)
    err = kwargs.get("err", err_on_zero_or_null_check)
    if err:
        func.errcheck = err
    return func
# EN: A utility function to define and wrap Windows API functions for easier access in Python.
# DE: Eine Hilfsfunktion zum Definieren und Einbinden von Windows-API-Funktionen für einfacheren Zugriff in Python.

GetClientRect = quick_win_define(
    "user32.GetClientRect", BOOL, HWND, POINTER(RECT), params=(1, 2)
)
# EN: Wraps the GetClientRect function to retrieve the dimensions of a window's client area.
# DE: Bindet die Funktion GetClientRect ein, um die Dimensionen des Client-Bereichs eines Fensters abzurufen.

GetDC = quick_win_define("user32.GetDC", HDC, HWND)
# EN: Wraps the GetDC function to retrieve a device context for a window.
# DE: Bindet die Funktion GetDC ein, um einen Geräte-Kontext für ein Fenster abzurufen.

CreateCompatibleDC = quick_win_define("gdi32.CreateCompatibleDC", HDC, HDC)
# EN: Wraps the CreateCompatibleDC function to create a memory device context compatible with a given device context.
# DE: Bindet die Funktion CreateCompatibleDC ein, um einen kompatiblen Speicher-Geräte-Kontext zu erstellen.

CreateCompatibleBitmap = quick_win_define(
    "gdi32.CreateCompatibleBitmap", HBITMAP, HDC, c_int, c_int
)
# EN: Creates a bitmap compatible with a device context.
# DE: Erstellt ein Bitmap, das mit einem Geräte-Kontext kompatibel ist.

ReleaseDC = quick_win_define("user32.ReleaseDC", c_int, HWND, HDC)
# EN: Releases a device context back to the system.
# DE: Gibt einen Geräte-Kontext an das System zurück.

DeleteDC = quick_win_define("gdi32.DeleteDC", BOOL, HDC)
# EN: Deletes a device context to free resources.
# DE: Löscht einen Geräte-Kontext, um Ressourcen freizugeben.

DeleteObject = quick_win_define("gdi32.DeleteObject", BOOL, HGDIOBJ)
# EN: Deletes a GDI object like a bitmap or brush.
# DE: Löscht ein GDI-Objekt wie ein Bitmap oder einen Pinsel.

SelectObject = quick_win_define("gdi32.SelectObject", HGDIOBJ, HDC, HGDIOBJ)
# EN: Selects a GDI object into a device context.
# DE: Wählt ein GDI-Objekt in einen Geräte-Kontext aus.

BitBlt = quick_win_define(
    "gdi32.BitBlt", BOOL, HDC, c_int, c_int, c_int, c_int, HDC, c_int, c_int, DWORD
)
# EN: Performs a bit-block transfer of pixel data from source to destination.
# DE: Führt einen Bit-Block-Transfer von Pixeldaten von der Quelle zum Ziel durch.

GetDIBits = quick_win_define(
    "gdi32.GetDIBits",
    c_int,
    HDC,
    HBITMAP,
    UINT,
    UINT,
    LPVOID,
    POINTER(BITMAPINFOHEADER),
    UINT,
)
# EN: Retrieves the bitmap data as an array of pixel values.
# DE: Ruft die Bitmap-Daten als Array von Pixelwerten ab.

GetDesktopWindow = quick_win_define("user32.GetDesktopWindow", HWND)
# EN: Retrieves a handle to the desktop window.
# DE: Ruft einen Handle des Desktop-Fensters ab.

class Grabber(object):
    def __init__(self, window=None, with_alpha=False, bbox=None):
        window = window or GetDesktopWindow()
        self.window = window
        rect = GetClientRect(window)
        self.width = rect.right - rect.left
        self.height = rect.bottom - rect.top
        if bbox:
            bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            if not bbox[2] or not bbox[3]:
                bbox[2] = self.width - bbox[0]
                bbox[3] = self.height - bbox[1]
            self.x, self.y, self.width, self.height = bbox
        else:
            self.x = 0
            self.y = 0
        self.windowDC = GetDC(window)
        self.memoryDC = CreateCompatibleDC(self.windowDC)
        self.bitmap = CreateCompatibleBitmap(self.windowDC, self.width, self.height)
        self.bitmapInfo = BITMAPINFOHEADER()
        self.bitmapInfo.biSize = sizeof(BITMAPINFOHEADER)
        self.bitmapInfo.biPlanes = 1
        self.bitmapInfo.biBitCount = 32 if with_alpha else 24
        self.bitmapInfo.biWidth = self.width
        self.bitmapInfo.biHeight = -self.height
        self.bitmapInfo.biCompression = BI_RGB
        self.bitmapInfo.biSizeImage = 0
        self.channels = 4 if with_alpha else 3
        self.closed = False
# EN: Initializes the Grabber class for capturing the screen with optional parameters.
# DE: Initialisiert die Grabber-Klasse zur Bildschirmaufnahme mit optionalen Parametern.

    def __del__(self):
        try:
            self.close()
        except:
            pass
# EN: Ensures resources are freed when the object is deleted.
# DE: Stellt sicher, dass Ressourcen beim Löschen des Objekts freigegeben werden.

    def close(self):
        if self.closed:
            return
        ReleaseDC(self.window, self.windowDC)
        DeleteDC(self.memoryDC)
        DeleteObject(self.bitmap)
        self.closed = True
# EN: Releases all resources associated with the Grabber instance.
# DE: Gibt alle Ressourcen frei, die mit der Grabber-Instanz verbunden sind.

    def grab(self, output=None):
        if self.closed:
            raise ValueError("Grabber already closed")
        if output is None:
            output = np.empty((self.height, self.width, self.channels), dtype="uint8")
        else:
            if output.shape != (self.height, self.width, self.channels):
                raise ValueError("Invalid output dimensions")
        SelectObject(self.memoryDC, self.bitmap)
        BitBlt(
            self.memoryDC,
            0,
            0,
            self.width,
            self.height,
            self.windowDC,
            self.x,
            self.y,
            SRCCOPY,
        )
        GetDIBits(
            self.memoryDC,
            self.bitmap,
            0,
            self.height,
            output.ctypes.data,
            self.bitmapInfo,
            DIB_RGB_COLORS,
        )
        return output
# EN: Captures the screen and returns the pixel data as a NumPy array.
# DE: Nimmt den Bildschirm auf und gibt die Pixeldaten als NumPy-Array zurück.
