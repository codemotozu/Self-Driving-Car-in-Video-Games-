import pyxinput  # Imports the pyxinput module, used to simulate Xbox controller inputs. | Importiert das pyxinput-Modul, das zur Simulation von Xbox-Controller-Eingaben verwendet wird.
import random    # Imports the random module, used to generate random numbers. | Importiert das random-Modul, das zur Erzeugung zufälliger Zahlen verwendet wird.
import time      # Imports the time module, used for adding delays in the program. | Importiert das time-Modul, das für Verzögerungen im Programm verwendet wird.
#test  # A comment, indicating a test or placeholder line. | Ein Kommentar, der auf einen Test oder Platzhalter hinweist.

class XboxControllerEmulator:  # Defines a class to emulate an Xbox 360 controller. | Definiert eine Klasse zur Emulation eines Xbox 360-Controllers.
    """
    Emulates a xbox 360 controller using pyxinput  # A description of the class functionality. | Eine Beschreibung der Funktionsweise der Klasse.
    """

    virtual_controller: pyxinput.vController  # Declares a variable to hold the virtual controller object. | Deklariert eine Variable, die das virtuelle Controller-Objekt hält.

    def __init__(self):  # Initializes the class when an object is created. | Initialisiert die Klasse, wenn ein Objekt erstellt wird.
        self.virtual_controller = pyxinput.vController()  # Creates a virtual Xbox controller object using pyxinput. | Erstellt ein virtuelles Xbox-Controller-Objekt mit pyxinput.
        print("Virtual xbox 360 controller crated")  # Prints a message confirming creation. | Gibt eine Nachricht aus, die die Erstellung bestätigt.

    def stop(self):  # Defines a method to stop the controller emulation. | Definiert eine Methode, um die Controller-Emulation zu stoppen.
        self.virtual_controller.UnPlug()  # Unplugs the virtual controller, ending the emulation. | Trennt den virtuellen Controller, wodurch die Emulation beendet wird.
        print("Virtual xbox 360 controller removed")  # Prints a message confirming removal. | Gibt eine Nachricht aus, die die Entfernung bestätigt.

    def set_axis_lx(self, lx: float):  # Sets the value for the left analog stick's X-axis. | Setzt den Wert für die X-Achse des linken Analogsticks.
        """
        Sets the x value for the right stick  # Description of what the method does. | Beschreibung dessen, was die Methode macht.
        """
        assert -1.0 <= lx <= 1.0, f"Controller values must be in range [-1,1]. x: {lx}"  # Ensures the input value is within the valid range. | Stellt sicher, dass der Eingabewert im gültigen Bereich liegt.
        self.virtual_controller.set_value("AxisLx", lx)  # Sets the X-axis value of the left stick. | Setzt den X-Achsen-Wert des linken Sticks.

    def set_axis_ly(self, ly: float):  # Sets the value for the left analog stick's Y-axis. | Setzt den Wert für die Y-Achse des linken Analogsticks.
        """
        Sets the x value for the left stick  # Description of what the method does. | Beschreibung dessen, was die Methode macht.
        """
        assert -1.0 <= ly <= 1.0, f"Controller values must be in range [-1,1]. y: {ly}"  # Ensures the input value is within the valid range. | Stellt sicher, dass der Eingabewert im gültigen Bereich liegt.
        self.virtual_controller.set_value("AxisLy", ly)  # Sets the Y-axis value of the left stick. | Setzt den Y-Achsen-Wert des linken Sticks.

    def set_axis(self, lx: float, ly: float):  # Sets both the X and Y values for the left analog stick. | Setzt sowohl die X- als auch Y-Werte für den linken Analogstick.
        """
        Sets the x and y values for the right stick  # Description of what the method does. | Beschreibung dessen, was die Methode macht.
        """
        self.set_axis_lx(lx)  # Calls the method to set the X value of the left stick. | Ruft die Methode auf, um den X-Wert des linken Sticks zu setzen.
        self.set_axis_ly(ly)  # Calls the method to set the Y value of the left stick. | Ruft die Methode auf, um den Y-Wert des linken Sticks zu setzen.

    def set_trigger_lt(self, lt: float):  # Sets the value for the left trigger. | Setzt den Wert für den linken Trigger.
        """
        Sets the t value for the left trigger  # Description of what the method does. | Beschreibung dessen, was die Methode macht.
        """
        assert -1.0 <= lt <= 1.0, f"Controller values must be in range [-1,1]. lt: {lt}"  # Ensures the input value is within the valid range. | Stellt sicher, dass der Eingabewert im gültigen Bereich liegt.
        self.virtual_controller.set_value("TriggerL", lt)  # Sets the value for the left trigger. | Setzt den Wert für den linken Trigger.

    def set_trigger_rt(self, rt: float):  # Sets the value for the right trigger. | Setzt den Wert für den rechten Trigger.
        """
        Sets the t value for the right trigger  # Description of what the method does. | Beschreibung dessen, was die Methode macht.
        """
        assert -1.0 <= rt <= 1.0, f"Controller values must be in range [-1,1]. rt: {rt}"  # Ensures the input value is within the valid range. | Stellt sicher, dass der Eingabewert im gültigen Bereich liegt.
        self.virtual_controller.set_value("TriggerR", rt)  # Sets the value for the right trigger. | Setzt den Wert für den rechten Trigger.

    def set_controller_state(self, lx: float, lt: float, rt: float):  # Sets the values for the left stick and triggers. | Setzt die Werte für den linken Stick und die Trigger.
        """
        Sets the x value for the left stick and the t value for the left and right triggers  # Description of what the method does. | Beschreibung dessen, was die Methode macht.
        """
        self.set_axis_lx(lx)  # Sets the left stick's X-axis value. | Setzt den X-Achsen-Wert des linken Sticks.
        self.set_trigger_lt(lt)  # Sets the left trigger's value. | Setzt den Wert des linken Triggers.
        self.set_trigger_rt(rt)  # Sets the right trigger's value. | Setzt den Wert des rechten Triggers.

    def test(self, num_tests: int = 10, delay: float = 0.5):  # Tests the controller by generating random values for inputs. | Testet den Controller, indem zufällige Werte für die Eingaben generiert werden.
        """
        Tests if the virtual controller works correctly using random values  # Description of the test method. | Beschreibung der Testmethode.
        """
        print(f"Testing left stick...")  # Prints a message indicating the start of testing for the left stick. | Gibt eine Nachricht aus, die den Beginn des Tests für den linken Stick anzeigt.
        for test_no in range(num_tests):  # Loops for the number of tests specified. | Schleife für die angegebene Anzahl an Tests.
            lx, ly = (
                1 - (random.random() * 2),  # Random value between -1 and 1 for the X-axis. | Zufälliger Wert zwischen -1 und 1 für die X-Achse.
                1 - (random.random() * 2),  # Random value between -1 and 1 for the Y-axis. | Zufälliger Wert zwischen -1 und 1 für die Y-Achse.
            )
            print(f"LX: {lx} \t LY: {ly}")  # Prints the random values for the X and Y axes. | Gibt die zufälligen Werte für die X- und Y-Achse aus.
            self.set_axis(
                lx=lx, ly=ly,  # Sets the left stick's X and Y values. | Setzt die X- und Y-Werte des linken Sticks.
            )
            time.sleep(delay)  # Waits for the specified delay before the next test. | Wartet auf die angegebene Verzögerung vor dem nächsten Test.
        self.set_axis(
            lx=0.0, ly=0.0,  # Resets the left stick's position to the center. | Setzt die Position des linken Sticks auf die Mitte zurück.
        )

        print(f"Testing left trigger...")  # Prints a message indicating the start of testing for the left trigger. | Gibt eine Nachricht aus, die den Beginn des Tests für den linken Trigger anzeigt.
        for test_no in range(num_tests):  # Loops for the number of tests specified. | Schleife für die angegebene Anzahl an Tests.
            lt = 1 - (random.random() * 2)  # Random value between -1 and 1 for the left trigger. | Zufälliger Wert zwischen -1 und 1 für den linken Trigger.
            print(f"LT: {lt}")  # Prints the random value for the left trigger. | Gibt den zufälligen Wert für den linken Trigger aus.
            self.set_trigger_lt(lt=lt)  # Sets the left trigger's value. | Setzt den Wert des linken Triggers.
            time.sleep(delay)  # Waits for the specified delay before the next test. | Wartet auf die angegebene Verzögerung vor dem nächsten Test.
        self.set_trigger_lt(lt=0.0)  # Resets the left trigger's value to the center. | Setzt den Wert des linken Triggers auf die Mitte zurück.

        print(f"Testing right trigger...")  # Prints a message indicating the start of testing for the right trigger. | Gibt eine Nachricht aus, die den Beginn des Tests für den rechten Trigger anzeigt.
        for test_no in range(num_tests):  # Loops for the number of tests specified. | Schleife für die angegebene Anzahl an Tests.
            rt = 1 - (random.random() * 2)  # Random value between -1 and 1 for the right trigger. | Zufälliger Wert zwischen -1 und 1 für den rechten Trigger.
            print(f"RT: {rt}")  # Prints the random value for the right trigger. | Gibt den zufälligen Wert für den rechten Trigger aus.
            self.set_trigger_rt(rt=rt)  # Sets the right trigger's value. | Setzt den Wert des rechten Triggers.
            time.sleep(delay)  # Waits for the specified delay before the next test. | Wartet auf die angegebene Verzögerung vor dem nächsten Test.
        self.set_trigger_rt(rt=0.0)  # Resets the right trigger's value to the center. | Setzt den Wert des rechten Triggers auf die Mitte zurück.
