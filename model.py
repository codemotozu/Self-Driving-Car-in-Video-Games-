from typing import List  # Allows type hinting for lists / Ermöglicht Typanmerkungen für Listen
import torch  # PyTorch library for deep learning / PyTorch-Bibliothek für Deep Learning
import torch.nn as nn  # Submodule for neural network components / Submodul für neuronale Netzkomponenten
import torchvision  # PyTorch module for computer vision / PyTorch-Modul für Computer Vision
import torchvision.models as models  # Pre-trained models for vision tasks / Vorgefertigte Modelle für Vision-Aufgaben
import pytorch_lightning as pl  # Framework for PyTorch to simplify training loops / Framework für PyTorch zur Vereinfachung von Trainingsschleifen
import torchmetrics  # Metrics library for PyTorch / Bibliothek für Metriken in PyTorch
from optimizers.optimizer import get_adafactor, get_adamw  # Import specific optimizers / Importiert spezifische Optimierer
from optimizers.scheduler import (  # Import scheduler functions / Importiert Scheduler-Funktionen
    get_reducelronplateau,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from packaging import version  # Used to check versioning / Wird zur Versionsprüfung verwendet

class WeightedMseLoss(nn.Module):  # Defines a custom loss function class / Definiert eine benutzerdefinierte Verlustfunktion-Klasse
    """
    Weighted mse loss columnwise
    Weighted MSE-Verlust berechnet spaltenweise
    """
    def __init__(  # Constructor for the loss class / Konstruktor für die Verlustklasse
        self,
        weights: List[float] = None,  # List of weights for each dimension / Liste der Gewichte für jede Dimension
        reduction: str = "mean",  # Method to reduce the loss / Methode zur Reduktion des Verlusts
    ):
        """
        INIT
        Initialization method / Initialisierungsmethode

        :param List[float] weights: List of weights for each joystick / Liste der Gewichte pro Joystick
        :param str reduction: "mean" or "sum" / "mean" oder "sum"
        """
        assert reduction in ["sum", "mean"], (  # Check that reduction is valid / Überprüft, ob die Reduktionsmethode gültig ist
            f"Reduction method: {reduction} not implemented. "
            f"Available reduction methods: [sum,mean]"
        )
        super(WeightedMseLoss, self).__init__()  # Calls the base class constructor / Ruft den Konstruktor der Basisklasse auf

        self.reduction = reduction  # Stores the reduction method / Speichert die Reduktionsmethode
        if not weights:  # If weights are not provided, default to [1.0, 1.0] / Falls keine Gewichte angegeben, Standardwerte [1.0, 1.0]
            weights = [1.0, 1.0]
        weights = torch.tensor(weights)  # Convert weights to a tensor / Konvertiert die Gewichte in einen Tensor
        weights.requires_grad = False  # Disables gradient computation for weights / Deaktiviert Gradientenberechnung für Gewichte

        self.register_buffer("weights", weights)  # Registers weights as a buffer / Registriert Gewichte als Puffer

    def forward(  # Forward pass to compute the loss / Vorwärtsdurchlauf zur Berechnung des Verlusts
        self,
        predicted: torch.tensor,  # Predicted values / Vorhergesagte Werte
        target: torch.tensor,  # Target values / Zielwerte
    ) -> torch.tensor:
        """
        Forward pass
        Vorwärtsdurchlauf

        :param torch.tensor predicted: Predicted values [batch_size, 2] / Vorhergesagte Werte [batch_size, 2]
        :param torch.tensor target: Target values [batch_size, 2] / Zielwerte [batch_size, 2]
        :return: Loss [1] if reduction is "mean" else [2] / Verlust [1], wenn Reduktion "mean" ist, sonst [2]
        """
        if self.reduction == "mean":  # If reduction is mean, calculate mean loss / Falls Reduktion "mean", berechne Durchschnittsverlust
            loss_per_joystick: torch.tensor = torch.mean(  # Mean squared error for each dimension / Mittlerer quadratischer Fehler für jede Dimension
                (predicted - target) ** 2, dim=0
            )
            return torch.mean(self.weights * loss_per_joystick)  # Weighted mean of losses / Gewichtetes Mittel der Verluste
        else:  # If reduction is sum, calculate sum loss / Falls Reduktion "sum", berechne Summenverlust
            loss_per_joystick: torch.tensor = torch.sum(  # Sum squared error for each dimension / Summierter quadratischer Fehler für jede Dimension
                (predicted - target) ** 2, dim=0
            )
            return self.weights * loss_per_joystick  # Weighted sum of losses / Gewichtete Summe der Verluste



class CrossEntropyLoss(torch.nn.Module):  # Define a custom class for weighted Cross Entropy loss, inheriting from PyTorch's Module class. / Definiert eine benutzerdefinierte Klasse für gewichteten Cross Entropy Verlust, die von Pytorch's Module-Klasse erbt.
    """  
    Weighted CrossEntropyLoss  # A docstring describing the purpose of the class. / Eine Dokumentationszeichenkette, die den Zweck der Klasse beschreibt.
    """

    def __init__(  # Constructor method to initialize the class. / Konstruktormethode, um die Klasse zu initialisieren.
        self,
        weights: List[float] = None,  # Optional parameter for specifying weights for each class. / Optionaler Parameter zum Festlegen von Gewichtungen für jede Klasse.
        reduction: str = "mean",  # Specifies the reduction method for the loss ("mean" or "sum"). / Gibt die Reduktionsmethode für den Verlust an ("mean" oder "sum").
        label_smoothing: float = 0.0,  # Specifies label smoothing factor, which helps regularize the model. / Gibt den Label-Glättungsfaktor an, der das Modell reguliert.
    ):
        """
        INIT  # Constructor docstring. / Konstruktor-Dokumentationszeichenkette.

        :param List[float] weights: List of weights for each class. / Liste von Gewichtungen für jede Klasse.
        :param str reduction: "mean" or "sum" for loss reduction method. / "mean" oder "sum" für die Reduktionsmethode des Verlusts.
        :param float label_smoothing: A float in [0.0, 1.0]. Specifies smoothing for the loss calculation. / Ein Float-Wert im Bereich [0.0, 1.0], der den Glättungsgrad bei der Verlustberechnung angibt.
        """

        assert reduction in ["sum", "mean"], (  # Ensures that reduction is either "mean" or "sum". / Stellt sicher, dass die Reduktionsmethode entweder "mean" oder "sum" ist.
            f"Reduction method: {reduction} not implemented. "
            f"Available reduction methods: [sum,mean]"
        )

        super(CrossEntropyLoss, self).__init__()  # Calls the parent class constructor. / Ruft den Konstruktor der übergeordneten Klasse auf.

        self.reduction = reduction  # Stores the reduction method. / Speichert die Reduktionsmethode.
        if weights:  # If weights are provided, convert them into a tensor and disable gradient calculation for them. / Wenn Gewichtungen bereitgestellt werden, werden sie in einen Tensor umgewandelt, und die Berechnung des Gradienten wird für diese deaktiviert.
            weights = torch.tensor(weights)
            weights.requires_grad = False

        self.register_buffer("weights", weights)  # Registers the weights as a buffer so they don't get updated during backpropagation. / Registriert die Gewichtungen als Puffer, damit sie während des Backpropagationsprozesses nicht aktualisiert werden.

        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss(  # Initializes the standard CrossEntropyLoss with the provided parameters. / Initialisiert den Standard-CrossEntropyLoss mit den bereitgestellten Parametern.
            reduction=reduction,
            weight=weights,
            label_smoothing=label_smoothing,
        )

    def forward(  # Defines the forward pass for the loss calculation. / Definiert den Vorwärtsdurchlauf für die Verlustberechnung.
        self,
        predicted: torch.tensor,  # The predicted values from the model. / Die vorhergesagten Werte des Modells.
        target: torch.tensor,  # The true target values. / Die tatsächlichen Zielwerte.
    ) -> torch.tensor:  # Returns the computed loss. / Gibt den berechneten Verlust zurück.

        """
        Forward pass  # Docstring for the forward pass. / Dokumentationszeichenkette für den Vorwärtsdurchlauf.

        :param torch.tensor predicted: Predicted values [batch_size, 9] / Vorhergesagte Werte [batch_size, 9]
        :param torch.tensor target: Target values [batch_size] / Zielwerte [batch_size]
        :return: Loss [1] if reduction is "mean" else [9] / Rückgabewert: Verlust [1], wenn die Reduktion "mean" ist, sonst [9]
        """
        return self.CrossEntropyLoss(predicted, target)  # Calls the CrossEntropyLoss object to compute the loss. / Ruft das CrossEntropyLoss-Objekt auf, um den Verlust zu berechnen.


class CrossEntropyLossImageReorder(torch.nn.Module):  # Define a custom class for weighted CrossEntropyLoss specific to image reordering. / Definiert eine benutzerdefinierte Klasse für gewichteten CrossEntropyLoss, der für Bildneuordnungen spezifisch ist.
    """ 
    Weighted CrossEntropyLoss for Image Reordering  # A docstring for the image reordering loss class. / Eine Dokumentationszeichenkette für die Bildneuordnungs-Verlustklasse.
    """

    def __init__(  # Constructor method to initialize the class. / Konstruktormethode zur Initialisierung der Klasse.
        self,
        label_smoothing: float = 0.0,  # Smoothing factor for label smoothing. / Glättungsfaktor für die Label-Glättung.
    ):
        """
        INIT  # Constructor docstring. / Konstruktor-Dokumentationszeichenkette.

        :param float label_smoothing: A float in [0.0, 1.0]. Specifies smoothing for the loss calculation. / Ein Float-Wert im Bereich [0.0, 1.0], der den Glättungsgrad bei der Verlustberechnung angibt.
        """

        super(CrossEntropyLossImageReorder, self).__init__()  # Calls the parent class constructor. / Ruft den Konstruktor der übergeordneten Klasse auf.

        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss(  # Initializes the standard CrossEntropyLoss with label smoothing. / Initialisiert den Standard-CrossEntropyLoss mit Label-Glättung.
            label_smoothing=label_smoothing
        )

    def forward(  # Defines the forward pass for loss calculation. / Definiert den Vorwärtsdurchlauf zur Verlustberechnung.
        self,
        predicted: torch.tensor,  # The predicted values from the model. / Die vorhergesagten Werte des Modells.
        target: torch.tensor,  # The true target values. / Die tatsächlichen Zielwerte.
    ) -> torch.tensor:  # Returns the computed loss. / Gibt den berechneten Verlust zurück.

        """
        Forward pass  # Docstring for the forward pass. / Dokumentationszeichenkette für den Vorwärtsdurchlauf.

        :param torch.tensor predicted: Predicted values [batch_size, 5] / Vorhergesagte Werte [batch_size, 5]
        :param torch.tensor target: Target values [batch_size] / Zielwerte [batch_size]
        :return: Loss [1] / Rückgabewert: Verlust [1]
        """

        return self.CrossEntropyLoss(predicted.view(-1, 5), target.view(-1).long())  # Reshapes the tensors and computes the loss. / Formt die Tensors um und berechnet den Verlust.

class ImageReorderingAccuracy(torchmetrics.Metric):  # Defines a custom metric class for measuring image reordering accuracy. / Definiert eine benutzerdefinierte Metrik-Klasse zur Messung der Bildumordnungsgenauigkeit.
    """
    Image Reordering Accuracy Metric  # Describes the purpose of the class. / Beschreibt den Zweck der Klasse.
    """

    def __init__(self, dist_sync_on_step=False):  # Initializes the metric, takes a parameter for distributed sync. / Initialisiert die Metrik, nimmt einen Parameter für die Verteilungssynchronisation.
        """
        INIT  # Briefly mentions the initialization method. / Erwähnt kurz die Initialisierungsmethode.

        :param bool dist_sync_on_step: If True, the metric will be synchronized on step  # Parameter description: if True, synchronizes on each step. / Parameterbeschreibung: Wenn True, wird die Metrik bei jedem Schritt synchronisiert.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)  # Calls the parent class' initializer. / Ruft den Initialisierer der übergeordneten Klasse auf.

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")  # Adds state for correct predictions, initialized to zero. / Fügt den Zustand für richtige Vorhersagen hinzu, initialisiert mit null.
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")  # Adds state for total predictions, initialized to zero. / Fügt den Zustand für die Gesamtzahl der Vorhersagen hinzu, initialisiert mit null.

    def update(self, preds: torch.Tensor, target: torch.Tensor):  # Updates the metric with new predictions and targets. / Aktualisiert die Metrik mit neuen Vorhersagen und Zielen.
        """
        Update the metric with the given predictions and targets  # Describes the purpose of the update method. / Beschreibt den Zweck der Update-Methode.

        :param torch.Tensor preds: Predictions [batch_size, 5]  # Description of the input tensor for predictions. / Beschreibung des Eingabetensors für Vorhersagen.
        :param torch.Tensor target: Target values [batch_size]  # Description of the input tensor for targets. / Beschreibung des Eingabentensors für Zielwerte.
        """
        assert (  # Ensures predictions and targets have the same shape. / Stellt sicher, dass Vorhersagen und Ziele die gleiche Form haben.
            preds.size() == target.size()
        ), f"Pred size: {preds.size()} != Target size: {target.size()}"  # Raises an error if sizes don't match. / Löst einen Fehler aus, wenn die Größen nicht übereinstimmen.

        self.correct += torch.sum(torch.all(preds == target, dim=-1))  # Counts the number of correct predictions. / Zählt die Anzahl der richtigen Vorhersagen.
        self.total += target.size(0)  # Updates the total number of predictions. / Aktualisiert die Gesamtzahl der Vorhersagen.

    def compute(self):  # Computes the final accuracy. / Berechnet die endgültige Genauigkeit.
        return self.correct.float() / self.total  # Returns the ratio of correct to total predictions. / Gibt das Verhältnis von richtigen zu gesamten Vorhersagen zurück.


def get_cnn(cnn_model_name: str, pretrained: bool) -> (torchvision.models, int):  # Function to retrieve a CNN model from torchvision. / Funktion zum Abrufen eines CNN-Modells aus torchvision.
    """
    Get a CNN model from torchvision.models (https://pytorch.org/vision/stable/models.html)  # Describes the purpose of the function. / Beschreibt den Zweck der Funktion.

    :param str cnn_model_name: Name of the CNN model from torchvision.models  # Parameter for the model name. / Parameter für den Modellnamen.
    :param bool pretrained: If True, the model will be loaded with pretrained weights  # Parameter for loading pretrained weights. / Parameter zum Laden von vortrainierten Gewichten.
    :return: CNN model, last layer output size  # Returns the CNN model and the output size of the last layer. / Gibt das CNN-Modell und die Ausgabedimension der letzten Schicht zurück.
    """

    # Get the CNN model  # Retrieves the CNN model dynamically. / Ruft das CNN-Modell dynamisch ab.
    cnn_call_method = getattr(models, cnn_model_name)  # Uses reflection to get the model by name. / Verwendet Reflection, um das Modell nach Name abzurufen.
    cnn_model = cnn_call_method(pretrained=pretrained)  # Initializes the CNN model with or without pretrained weights. / Initialisiert das CNN-Modell mit oder ohne vortrainierte Gewichte.

    # Remove classification layer  # Removes the final classification layer of the model. / Entfernt die endgültige Klassifikationsschicht des Modells.
    _ = cnn_model._modules.popitem()  # Pops off the last module (classification layer). / Entfernt das letzte Modul (Klassifikationsschicht).
    cnn_model = nn.Sequential(*list(cnn_model.children()))  # Converts the model into a sequential model. / Wandelt das Modell in ein sequentielles Modell um.

    # Test output_size of last layer of the CNN (Not the most efficient way, but it works)  # Tests the output size of the CNN's last layer. / Testet die Ausgabedimension der letzten Schicht des CNNs.
    features = cnn_model(torch.zeros((1, 3, 270, 480), dtype=torch.float32))  # Passes a dummy input through the CNN. / Gibt eine Dummy-Eingabe durch das CNN.
    output_size: int = features.reshape(features.size(0), -1).size(1)  # Calculates the flattened output size of the last layer. / Berechnet die flache Ausgabedimension der letzten Schicht.

    return cnn_model, output_size  # Returns the model and the output size. / Gibt das Modell und die Ausgabedimension zurück.


class EncoderCNN(nn.Module):  # Defines a CNN encoder class that extracts features from images. / Definiert eine CNN-Encoder-Klasse, die Merkmale aus Bildern extrahiert.
    """
    Encoder CNN, extracts features from the input images  # Describes the purpose of the EncoderCNN class. / Beschreibt den Zweck der EncoderCNN-Klasse.

    For efficiency the input is a single sequence of [sequence_size*batch_size] images,  # Explains the format of the input to the encoder. / Erklärt das Format der Eingabe in den Encoder.
    the output of the CNN will be packed as sequences of sequence_size vectors.  # Describes how the output will be packed into sequences. / Beschreibt, wie die Ausgabe in Sequenzen gepackt wird.
    """

    def __init__(  # Initializes the CNN encoder with required parameters. / Initialisiert den CNN-Encoder mit den erforderlichen Parametern.
        self,
        embedded_size: int,
        dropout_cnn_out: float,
        cnn_model_name: str,
        pretrained_cnn: bool,
        sequence_size: int = 5,
    ):
        """
        INIT  # Describes the initialization method. / Beschreibt die Initialisierungsmethode.

        :param int embedded_size: Size of the output embedding  # Parameter for embedding size. / Parameter für die Embedding-Größe.
        :param float dropout_cnn_out: Dropout rate for the output of the CNN  # Dropout rate for CNN output. / Dropout-Rate für die CNN-Ausgabe.
        :param str cnn_model_name: Name of the CNN model from torchvision.models  # Parameter for CNN model name. / Parameter für den CNN-Modellnamen.
        :param bool pretrained_cnn: If True, the model will be loaded with pretrained weights  # Parameter for loading pretrained CNN. / Parameter zum Laden des vortrainierten CNN.
        :param int sequence_size: Size of the sequence of images  # Sequence size parameter. / Parameter für die Sequenzgröße von Bildern.
        """
        super(EncoderCNN, self).__init__()  # Calls the parent class' initializer. / Ruft den Initialisierer der übergeordneten Klasse auf.

        self.embedded_size = embedded_size  # Sets the embedded size for the model. / Setzt die eingebettete Größe für das Modell.
        self.cnn_model_name = cnn_model_name  # Stores the CNN model name. / Speichert den CNN-Modellnamen.
        self.dropout_cnn_out = dropout_cnn_out  # Sets the dropout rate for CNN output. / Setzt die Dropout-Rate für die CNN-Ausgabe.
        self.pretrained_cnn = pretrained_cnn  # Indicates if CNN uses pretrained weights. / Gibt an, ob CNN vortrainierte Gewichte verwendet.

        self.cnn, self.cnn_output_size = get_cnn(  # Retrieves the CNN model and its output size. / Ruft das CNN-Modell und seine Ausgabedimension ab.
            cnn_model_name=cnn_model_name, pretrained=pretrained_cnn
        )

        self.dp = nn.Dropout(p=dropout_cnn_out)  # Applies dropout after CNN output. / Wendet Dropout nach der CNN-Ausgabe an.
        self.dense = nn.Linear(self.cnn_output_size, self.cnn_output_size)  # Dense layer for further processing. / Dense-Schicht für die weitere Verarbeitung.
        self.layer_norm = nn.LayerNorm(self.cnn_output_size, eps=1e-05)  # Applies layer normalization. / Wendet Layer-Normalisierung an.

        self.decoder = nn.Linear(self.cnn_output_size, self.embedded_size)  # Decoder layer to output embeddings. / Decoder-Schicht zur Ausgabe von Embeddings.
        self.bias = nn.Parameter(torch.zeros(self.embedded_size))  # Initializes bias for the decoder. / Initialisiert den Bias für den Decoder.
        self.decoder.bias = self.bias  # Assigns the bias to the decoder. / Weist den Bias dem Decoder zu.
        self.gelu = nn.GELU()  # GELU activation function. / GELU-Aktivierungsfunktion.
        self.sequence_size = sequence_size  # Stores the sequence size. / Speichert die Sequenzgröße.

    def forward(self, images: torch.tensor) -> torch.tensor:  # Defines the forward pass. / Definiert den Forward-Pass.
        """
        Forward pass  # Describes the purpose of the forward method. / Beschreibt den Zweck der Forward-Methode.
        :param torch.tensor images: Input images [batch_size * sequence_size, 3, 270, 480]  # Description of input images. / Beschreibung der Eingabebilder.
        :return: Output embedding [batch_size, sequence_size, embedded_size]  # Output format description. / Beschreibung des Ausgabeformats.
        """
        features = self.cnn(images)  # Passes the input images through the CNN. / Gibt die Eingabebilder durch das CNN.
        features = features.reshape(features.size(0), -1)  # Reshapes the CNN output into a flat tensor. / Formatiert die CNN-Ausgabe in einen flachen Tensor um.

        """
        Reshapes the features from the CNN into a time distributed format  # Reshapes features for sequential processing. / Formatiert Merkmale für die sequentielle Verarbeitung um.
        """
        features = features.view(  # Reshapes the tensor to sequence format. / Formatiert den Tensor in das Sequenzformat um.
            int(features.size(0) / self.sequence_size),
            self.sequence_size,
            features.size(1),
        )

        features = self.dp(features)  # Applies dropout to the features. / Wendet Dropout auf die Merkmale an.
        features = self.dense(features)  # Applies dense layer processing. / Wendet die Dense-Schicht-Verarbeitung an.
        features = self.gelu(features)  # Applies GELU activation. / Wendet die GELU-Aktivierung an.
        features = self.layer_norm(features)  # Applies layer normalization. / Wendet die Layer-Normalisierung an.
        features = self.decoder(features)  # Decodes the features into embeddings. / Decodiert die Merkmale in Embeddings.
        return features  # Returns the output embeddings. / Gibt die Ausgabe-Embeddings zurück.

    def _tie_weights(self):  # Ensures that decoder weights stay tied with bias. / Stellt sicher, dass die Decoder-Gewichte mit dem Bias verbunden bleiben.
        self.bias = self.decoder.bias  # Ties the bias weights. / Verbindet die Bias-Gewichte.


class EncoderRNN(nn.Module):  # Defines an RNN encoder class. / Definiert eine RNN-Encoder-Klasse.
    """
    Extracts features from the input sequence using an RNN  # Describes the purpose of the EncoderRNN class. / Beschreibt den Zweck der EncoderRNN-Klasse.
    """

    def __init__(  # Initializes the RNN encoder. / Initialisiert den RNN-Encoder.
        self,
        embedded_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional_lstm: bool,
        dropout_lstm: float,
    ):

        """
        INIT  # Describes the initialization method. / Beschreibt die Initialisierungsmethode.
        :param int embedded_size: Size of the input feature vectors  # Parameter for embedded size. / Parameter für die eingebettete Größe.
        :param int hidden_size: LSTM hidden size  # Parameter for hidden size in LSTM. / Parameter für die versteckte Größe im LSTM.
        :param int num_layers: number of layers in the LSTM  # Parameter for number of LSTM layers. / Parameter für die Anzahl der LSTM-Schichten.
        :param bool bidirectional_lstm: forward or bidirectional LSTM  # Parameter for bidirectional LSTM. / Parameter für bidirektionales LSTM.
        :param float dropout_lstm: dropout probability for the LSTM  # Dropout rate for LSTM. / Dropout-Rate für das LSTM.
        """
        super(EncoderRNN, self).__init__()  # Calls the parent class' initializer. / Ruft den Initialisierer der übergeordneten Klasse auf.

        self.embedded_size = embedded_size  # Sets embedded size for input features. / Setzt die eingebettete Größe für Eingabemerkmale.
        self.hidden_size = hidden_size  # Sets the hidden size for LSTM. / Setzt die versteckte Größe für LSTM.
        self.num_layers = num_layers  # Sets the number of LSTM layers. / Setzt die Anzahl der LSTM-Schichten.
        self.bidirectional_lstm = bidirectional_lstm  # Stores whether LSTM is bidirectional. / Speichert, ob das LSTM bidirektional ist.
        self.dropout_lstm = dropout_lstm  # Sets the dropout rate for LSTM layers. / Setzt die Dropout-Rate für LSTM-Schichten.

        self.lstm: nn.LSTM = nn.LSTM(  # Initializes the LSTM layer. / Initialisiert die LSTM-Schicht.
            embedded_size,
            hidden_size,
            num_layers,
            dropout=dropout_lstm,
            bidirectional=bidirectional_lstm,
            batch_first=True,
        )

        self.bidirectional_lstm = bidirectional_lstm  # Stores bidirectional LSTM flag. / Speichert die Bidirektionalitäts-Flagge für LSTM.

    
    def forward(self, features: torch.tensor) -> torch.tensor:  # Defines the forward pass of the RNN encoder. / Definiert den Forward-Pass des RNN-Encoders.
        """
        Forward pass  # Describes the forward method. / Beschreibt die Forward-Methode.
        :param torch.tensor features: Input features [batch_size, sequence_size, embedded_size]  # Description of input features. / Beschreibung der Eingabemerkmale.
        :return: Output features [batch_size, hidden_size*2 if bidirectional else hidden_size]  # Output description for LSTM features. / Ausgabe-Beschreibung für LSTM-Merkmale.
        """
        output, (h_n, c_n) = self.lstm(features)  # Passes features through LSTM. / Gibt Merkmale durch das LSTM.
        if self.bidirectional_lstm:  # Checks if the LSTM is bidirectional. / Überprüft, ob das LSTM bidirektional ist.
            x = torch.cat((h_n[-2], h_n[-1]), 1)  # Concatenates the hidden states from both directions. / Verbindet die versteckten Zustände aus beiden Richtungen.
        else:
            x = h_n[-1]  # Takes the hidden state from the last layer. / Nimmt den versteckten Zustand der letzten Schicht.
        return x  # Returns the final hidden state as output. / Gibt den endgültigen versteckten Zustand als Ausgabe zurück.




class PositionalEmbedding(nn.Module):  # Define the PositionalEmbedding class, inheriting from nn.Module  # Definiere die PositionalEmbedding-Klasse, die von nn.Module erbt
    """
    Add positional encodings to the transformer input features  # Add position information to transformer input (important for sequential data)  # Füge Positionskodierungen zu den Transformer-Eingabefeatures hinzu
    """

    def __init__(  # Constructor to initialize the PositionalEmbedding class  # Konstruktor zur Initialisierung der PositionalEmbedding-Klasse
        self,
        sequence_length: int,  # Length of the input sequence  # Länge der Eingabesequenz
        d_model: int,  # Dimension of the model (input feature vector size)  # Dimension des Modells (Eingabe-Feature-Vektorgröße)
        dropout: float = 0.1,  # Dropout probability for the positional embeddings  # Dropout-Wahrscheinlichkeit für die Positionskodierungen
    ):
        """
        INIT
        :param int sequence_length: Length of the input sequence  # Sequence length input  # Länge der Eingabesequenz
        :param int d_model: Size of the input feature vectors  # Model's input feature vector size  # Größe der Eingabe-Feature-Vektoren des Modells
        :param float dropout: dropout probability for the embeddings  # Dropout probability  # Dropout-Wahrscheinlichkeit für die Embeddings
        """
        super(PositionalEmbedding, self).__init__()  # Initialize the parent class (nn.Module)  # Initialisiere die Elternklasse (nn.Module)

        self.d_model = d_model  # Save the dimension of the input features  # Speichere die Dimension der Eingabefeatures
        self.sequence_length = sequence_length  # Save the length of the sequence  # Speichere die Länge der Sequenz
        self.dropout = dropout  # Save dropout value  # Speichere den Dropout-Wert

        self.dropout = nn.Dropout(p=dropout)  # Initialize dropout layer for regularization  # Initialisiere die Dropout-Schicht zur Regularisierung
        pe = torch.zeros(self.sequence_length, d_model).float()  # Initialize a tensor of zeros for positional encoding  # Initialisiere einen Tensor aus Nullen für die Positionskodierung
        pe.requires_grad = True  # Allow gradients to be calculated for positional encoding  # Erlaube die Berechnung von Gradienten für die Positionskodierung
        pe = pe.unsqueeze(0)  # Add a batch dimension  # Füge eine Batch-Dimension hinzu
        self.pe = torch.nn.Parameter(pe)  # Convert tensor to learnable parameter  # Wandle den Tensor in einen lernbaren Parameter um
        torch.nn.init.normal_(self.pe, std=0.02)  # Initialize positional encodings with normal distribution  # Initialisiere Positionskodierungen mit einer Normalverteilung
        self.LayerNorm = nn.LayerNorm(self.d_model, eps=1e-05)  # Layer normalization for the input features  # Schichtnormalisierung für die Eingabefeatures
        self.dp = torch.nn.Dropout(p=dropout)  # Dropout layer to regularize the output  # Dropout-Schicht zur Regularisierung des Outputs

    def forward(self, x: torch.tensor) -> torch.tensor:  # Define the forward pass of the class  # Definiere den Forward-Pass der Klasse
        """
        Forward pass

        :param torch.tensor x: Input features [batch_size, sequence_size, embedded_size]  # Input features  # Eingabefeatures
        :return: Output features [batch_size, sequence_size, embedded_size]  # Return processed features  # Rückgabe der verarbeiteten Features
        """
        pe = self.pe[:, : x.size(1)]  # Select the appropriate number of positional encodings  # Wähle die entsprechende Anzahl an Positionskodierungen aus
        x = pe + x  # Add positional encodings to input features  # Addiere die Positionskodierungen zu den Eingabefeatures
        x = self.LayerNorm(x)  # Apply layer normalization to the input  # Wende Schichtnormalisierung auf die Eingabe an
        x = self.dp(x)  # Apply dropout to regularize the output  # Wende Dropout an, um den Output zu regularisieren
        return x  # Return the final features  # Rückgabe der finalen Features

class EncoderTransformer(nn.Module):  # Define the EncoderTransformer class, inheriting from nn.Module  # Definiere die EncoderTransformer-Klasse, die von nn.Module erbt
    """
    Extracts features from the input sequence using a Transformer  # Extract features using a transformer encoder  # Extrahiere Features aus der Eingabesequenz mithilfe eines Transformers
    """

    def __init__(  # Initialize the EncoderTransformer class  # Initialisiere die EncoderTransformer-Klasse
        self,
        d_model: int = 512,  # Dimension of the input feature vectors  # Dimension der Eingabe-Feature-Vektoren
        nhead: int = 8,  # Number of heads in multi-head attention  # Anzahl der Köpfe im Multi-Head-Attention-Mechanismus
        num_layers: int = 1,  # Number of transformer layers  # Anzahl der Transformer-Schichten
        dropout: float = 0.1,  # Dropout probability for transformer layers  # Dropout-Wahrscheinlichkeit für Transformer-Schichten
        sequence_length: int = 5,  # Length of the input sequence  # Länge der Eingabesequenz
    ):
        """
        INIT

        :param int d_model: Size of the input feature vectors  # Input feature vector size  # Größe der Eingabe-Feature-Vektoren
        :param int nhead: Number of heads in the multi-head attention  # Attention head count  # Anzahl der Köpfe im Multi-Head-Attention-Mechanismus
        :param int num_layers: number of transformer layers in the encoder  # Transformer encoder layer count  # Anzahl der Transformer-Schichten im Encoder
        :param float dropout: dropout probability of transformer layers in the encoder  # Dropout probability  # Dropout-Wahrscheinlichkeit der Transformer-Schichten im Encoder
        :param int sequence_length: Length of the input sequence  # Sequence length input  # Länge der Eingabesequenz
        """
        super(EncoderTransformer, self).__init__()  # Initialize the parent class (nn.Module)  # Initialisiere die Elternklasse (nn.Module)
        self.d_model = d_model  # Save input feature vector size  # Speichere die Größe der Eingabe-Feature-Vektoren
        self.nhead = nhead  # Save number of attention heads  # Speichere die Anzahl der Attention-Köpfe
        self.num_layers = num_layers  # Save number of transformer layers  # Speichere die Anzahl der Transformer-Schichten
        self.dropout = dropout  # Save dropout value  # Speichere den Dropout-Wert
        self.sequence_length = sequence_length  # Save sequence length  # Speichere die Länge der Sequenz

        cls_token = torch.zeros(1, 1, self.d_model).float()  # Initialize [CLS] token (used for classification tasks)  # Initialisiere das [CLS]-Token (für Klassifizierungsaufgaben)
        cls_token.require_grad = True  # Allow gradients for the CLS token  # Erlaube Gradienten für das CLS-Token
        self.clsToken = torch.nn.Parameter(cls_token)  # Convert the CLS token to a learnable parameter  # Wandle das CLS-Token in einen lernbaren Parameter um
        torch.nn.init.normal_(cls_token, std=0.02)  # Initialize the CLS token with normal distribution  # Initialisiere das CLS-Token mit einer Normalverteilung

        self.pe = PositionalEmbedding(  # Initialize positional embedding with sequence length and feature size  # Initialisiere die PositionalEmbedding mit Sequenzlänge und Feature-Größe
            sequence_length=self.sequence_length + 1, d_model=self.d_model
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(  # Initialize transformer encoder layer  # Initialisiere eine Transformer-Encoder-Schicht
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            activation="gelu",  # GELU activation function  # GELU-Aktivierungsfunktion
            batch_first=True,  # Batch dimension comes first  # Batch-Dimension kommt zuerst
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(  # Initialize the full transformer encoder  # Initialisiere den gesamten Transformer-Encoder
            encoder_layer, num_layers=self.num_layers
        )

        for parameter in self.transformer_encoder.parameters():  # Set transformer encoder parameters to require gradients  # Setze die Parameter des Transformer-Encoders auf „Benötigt Gradienten“
            parameter.requires_grad = True

    def forward(  # Define the forward pass of the EncoderTransformer  # Definiere den Forward-Pass des EncoderTransformers
        self, features: torch.tensor, attention_mask: torch.tensor = None  # Input features and attention mask  # Eingabefeatures und Attention-Maske
    ) -> torch.tensor:  # Return processed features  # Rückgabe der verarbeiteten Features
        """
        Forward pass

        :param torch.tensor features: Input features [batch_size, sequence_length, embedded_size]  # Input features  # Eingabefeatures
        :param torch.tensor attention_mask: Mask for the input features  # Attention mask to exclude certain positions  # Attention-Maske zum Ausschließen bestimmter Positionen
        :return: Output features [batch_size, d_model]  # Processed features  # Verarbeitete Features
        """

        features = torch.cat(  # Concatenate the [CLS] token with the input features  # Füge das [CLS]-Token mit den Eingabefeatures zusammen
            (self.clsToken.repeat(features.size(0), 1, 1), features), dim=1
        )
        features = self.pe(features)  # Add positional encodings to the input features  # Füge Positionskodierungen zu den Eingabefeatures hinzu
        features = self.transformer_encoder(features, attention_mask)  # Pass through transformer encoder  # Durchlaufe den Transformer-Encoder
        return features  # Return processed features  # Rückgabe der verarbeiteten Features


class OutputLayer(nn.Module):  # Define the OutputLayer class for final classification  # Definiere die OutputLayer-Klasse für die endgültige Klassifizierung
    """
    Output layer of the model  # Ausgabeschicht des Modells
    Based on RobertaClassificationHead:  # Basierend auf RobertaClassificationHead:
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/roberta/modeling_roberta.py  # https://github.com/huggingface/transformers/blob/master/src/transformers/models/roberta/modeling_roberta.py
    """

    def __init__(  # Initialize the OutputLayer class  # Initialisiere die OutputLayer-Klasse
        self,
        d_model: int,  # Size of the encoder output vector  # Größe des Ausgabewerts des Encoders
        num_classes: int,  # Number of classes for classification  # Anzahl der Klassen für die Klassifizierung
        dropout_encoder_features: float = 0.2,  # Dropout probability for the encoder output  # Dropout-Wahrscheinlichkeit für die Ausgabe des Encoders
        from_transformer: bool = True,  # Whether to use the CLS token from transformer output  # Ob das CLS-Token aus der Transformer-Ausgabe verwendet werden soll
    ):
        """
        INIT  # Initialisierung

        :param int d_model: Size of the encoder output vector  # Encoder-Ausgabegröße
        :param int num_classes: Size of output vector  # Anzahl der Klassifikationsklassen
        :param float dropout_encoder_features: Dropout probability of the encoder output  # Dropout für Encoder-Ausgabe
        :param bool from_transformer: If true, get the CLS token from the transformer output  # Wenn wahr, hole das CLS-Token aus der Transformer-Ausgabe
        """
        super(OutputLayer, self).__init__()  # Initialize parent class  # Initialisiere die Elternklasse
        self.d_model = d_model  # Save encoder output size  # Speichere die Encoder-Ausgabegröße
        self.num_classes = num_classes  # Save number of classes for classification  # Speichere die Anzahl der Klassifikationsklassen
        self.dropout_encoder_features = dropout_encoder_features  # Save dropout value  # Speichere den Dropout-Wert
        self.dense = nn.Linear(self.d_model, self.d_model)  # Define a linear layer for feature transformation  # Definiere eine lineare Schicht zur Merkmalsumwandlung
        self.dp = nn.Dropout(p=dropout_encoder_features)  # Define dropout layer for encoder output regularization  # Definiere eine Dropout-Schicht für die Regularisierung der Encoder-Ausgabe
        self.out_proj = nn.Linear(self.d_model, self.num_classes)  # Linear layer for final output  # Lineare Schicht für die endgültige Ausgabe
        self.tanh = nn.Tanh()  # Tanh activation function  # Tanh-Aktivierungsfunktion
        self.from_transformer = from_transformer  # Whether to use the CLS token  # Ob das CLS-Token verwendet werden soll

    def forward(self, x):  # Define the forward pass of the OutputLayer  # Definiere den Vorwärtspass der OutputLayer
        """
        Forward pass  # Vorwärtspass

        :param torch.tensor x: Input features [batch_size, d_model] if RNN else [batch_size, sequence_length+1, d_model]  # Eingabefeatures [batch_size, d_model] wenn RNN, sonst [batch_size, sequenzlänge+1, d_model]
        :return: Output features [num_classes]  # Endgültige Klassifikationsausgabe
        """
        if self.from_transformer:  # If using the transformer output, take the CLS token  # Wenn die Transformer-Ausgabe verwendet wird, nimm das CLS-Token
            x = x[:, 0, :]  # Get the [CLS] token from the transformer output  # Hole das [CLS]-Token aus der Transformer-Ausgabe
        x = self.dp(x)  # Apply dropout  # Wende Dropout an
        x = self.dense(x)  # Apply linear transformation  # Wende die lineare Transformation an
        x = torch.tanh(x)  # Apply Tanh activation  # Wende Tanh-Aktivierung an
        x = self.dp(x)  # Apply dropout again  # Wende erneut Dropout an
        x = self.out_proj(x)  # Final linear transformation to output the classification result  # Endgültige lineare Transformation, um das Klassifikationsergebnis auszugeben
        return x  # Return the output features (classification result)  # Gib die Ausgabefeatures (Klassifikationsergebnis) zurück


# This is the definition of a class called OutputImageOrderingLayer
class OutputImageOrderingLayer(nn.Module):  # Defines a PyTorch neural network module class OutputImageOrderingLayer / Definiert eine PyTorch Neural Network Modulklasse OutputImageOrderingLayer
    """
    Output layer of the image reordering model
    Based on  RobertaLMHead:
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/roberta/modeling_roberta.py
    """
    # A class docstring explaining the purpose of this class / Eine Klassen-Dokumentation, die den Zweck dieser Klasse erklärt

    def __init__(self, d_model: int, num_classes: int):  # Constructor to initialize the object with d_model and num_classes parameters / Konstruktor zur Initialisierung des Objekts mit den Parametern d_model und num_classes
        """
        INIT
        :param int d_model: Size of the encoder output vector  / Größe des Encoder-Ausgabewerts
        :param int num_classes: Size of output vector  / Größe des Ausgabewerts
        """

        super(OutputImageOrderingLayer, self).__init__()  # Calls the parent class's constructor / Ruft den Konstruktor der Elternklasse auf

        self.d_model = d_model  # Sets the model's output vector size / Setzt die Größe des Modell-Ausgabewerts
        self.num_classes = num_classes  # Sets the number of output classes / Setzt die Anzahl der Ausgabeklassen
        self.dense = nn.Linear(self.d_model, self.d_model)  # A linear layer for transformation / Eine lineare Schicht zur Transformation
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-05)  # Normalizes the output of the dense layer / Normalisiert die Ausgabe der dichten Schicht
        self.decoder = nn.Linear(self.d_model, self.num_classes)  # A decoder to produce the final output / Ein Decoder, um die endgültige Ausgabe zu erzeugen
        self.bias = nn.Parameter(torch.zeros(num_classes))  # Bias parameter initialization / Bias-Parameterinitialisierung
        self.decoder.bias = self.bias  # Set the decoder's bias to the defined bias / Setzt den Bias des Decoders auf den definierten Bias
        self.gelu = nn.GELU()  # GELU activation function for the model / GELU-Aktivierungsfunktion für das Modell

    def forward(self, x):  # Defines the forward pass of the model / Definiert den Vorwärtsdurchlauf des Modells
        """
        Forward pass

        :param torch.tensor x: Input features [batch_size, sequence_length+1, d_model]  / Eingabe-Features [batch_size, sequence_length+1, d_model]
        :return: Output features [num_classes]  / Rückgabewerte der Ausgabe [num_classes]
        """
        x = self.dense(x)[:, 1:, :]  # Remove CLS token from input and apply dense layer / Entfernt das CLS-Token aus der Eingabe und wendet die dichte Schicht an
        x = self.gelu(x)  # Apply GELU activation / Wendet die GELU-Aktivierung an
        x = self.layer_norm(x)  # Apply Layer Normalization / Wendet die Schichtnormalisierung an
        x = self.decoder(x)  # Apply the decoder to produce final output / Wendet den Decoder an, um die endgültige Ausgabe zu erzeugen
        return x  # Returns the model's output / Gibt die Modell-Ausgabe zurück


# This is the definition of a class called Controller2Keyboard
class Controller2Keyboard(nn.Module):  # Defines a PyTorch neural network module class Controller2Keyboard / Definiert eine PyTorch Neural Network Modulklasse Controller2Keyboard
    """
    Map controller output to keyboard keys probabilities
    """  # Maps the output of the controller to keyboard keys probabilities / Ordnet die Ausgabe des Controllers den Wahrscheinlichkeiten der Tastaturtasten zu

    def __init__(self):  # Constructor for initializing the Controller2Keyboard object / Konstruktor zur Initialisierung des Controller2Keyboard-Objekts
        """
        INIT
        """
        super(Controller2Keyboard, self).__init__()  # Calls the parent class's constructor / Ruft den Konstruktor der Elternklasse auf
        keys2vector_matrix = torch.tensor(  # Defines a matrix to map keys to vectors / Definiert eine Matrix, um Tasten auf Vektoren abzubilden
            [
                [0.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
                [1.0, 1.0],
                [1.0, -1.0],
            ],
            requires_grad=False,  # The matrix does not require gradient computation / Die Matrix erfordert keine Gradientenberechnung
        )

        self.register_buffer("keys2vector_matrix", keys2vector_matrix)  # Registers the matrix as a buffer in the model / Registriert die Matrix als Puffer im Modell

    def forward(self, x: torch.tensor):  # Forward pass to map the controller input to keyboard keys probabilities / Vorwärtsdurchlauf zur Zuordnung der Controller-Eingabe zu Tastaturtasten-Wahrscheinlichkeiten
        """
        Forward pass

        :param torch.tensor x: Controller input [2]  / Controller-Eingabe [2]
        :return: Keyboard keys probabilities [9]  / Rückgabe: Wahrscheinlichkeiten für Tastaturtasten [9]
        """
        return 1.0 / torch.cdist(x, self.keys2vector_matrix)  # Computes the cosine distance and maps to probabilities / Berechnet die Kosinus-Distanz und ordnet sie den Wahrscheinlichkeiten zu


# This is the definition of a class called Keyboard2Controller
class Keyboard2Controller(nn.Module):  # Defines a PyTorch neural network module class Keyboard2Controller / Definiert eine PyTorch Neural Network Modulklasse Keyboard2Controller
    """
    Map keyboard keys probabilities to controller output
    """  # Maps the keyboard key probabilities to controller input / Ordnet die Tastaturtasten-Wahrscheinlichkeiten der Controller-Eingabe zu

    def __init__(self):  # Constructor for initializing the Keyboard2Controller object / Konstruktor zur Initialisierung des Keyboard2Controller-Objekts
        """
        INIT
        """
        super(Keyboard2Controller, self).__init__()  # Calls the parent class's constructor / Ruft den Konstruktor der Elternklasse auf
        keys2vector_matrix = torch.tensor(  # Defines a matrix to map keys to vectors / Definiert eine Matrix, um Tasten auf Vektoren abzubilden
            [
                [0.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
                [1.0, 1.0],
                [1.0, -1.0],
            ],
            requires_grad=False,  # The matrix does not require gradient computation / Die Matrix erfordert keine Gradientenberechnung
        )

        self.register_buffer("keys2vector_matrix", keys2vector_matrix)  # Registers the matrix as a buffer in the model / Registriert die Matrix als Puffer im Modell

    def forward(self, x: torch.tensor):  # Forward pass to map the keyboard key probabilities to controller input / Vorwärtsdurchlauf zur Zuordnung der Tastaturtasten-Wahrscheinlichkeiten der Controller-Eingabe
        """
        Forward pass

        :param torch.tensor x: Keyboard keys probabilities [9]  / Tastaturtasten-Wahrscheinlichkeiten [9]
        :return: Controller input [2]  / Rückgabe: Controller-Eingabe [2]
        """
        controller_inputs = self.keys2vector_matrix.repeat(len(x), 1, 1)  # Repeat the keys2vector matrix for each input batch / Wiederholt die keys2vector-Matrix für jedes Eingabebatch
        return (
            torch.sum(controller_inputs * x.view(len(x), 9, 1), dim=1)  # Computes weighted sum of vectors based on input probabilities / Berechnet die gewichtete Summe der Vektoren basierend auf den Eingabewahrscheinlichkeiten
            / torch.sum(x, dim=-1)[:, None]  # Normalizes the result by dividing by the sum of the probabilities / Normalisiert das Ergebnis, indem es durch die Summe der Wahrscheinlichkeiten geteilt wird
        )

class TEDD1104LSTM(nn.Module):
    """
    T.E.D.D 1104 model with LSTM encoder. The model consists of:
         - A CNN that extract features from the images
         - A RNN (LSTM) that extracts a representation of the image sequence
         - A linear output layer that predicts the controller input.
    """
    # This class defines the TEDD1104LSTM model, which uses CNN for feature extraction and LSTM for sequence processing.
    # Diese Klasse definiert das TEDD1104LSTM-Modell, das CNN zur Merkmalsextraktion und LSTM zur Sequenzverarbeitung verwendet.

    def __init__(
        self,
        cnn_model_name: str,
        pretrained_cnn: bool,
        embedded_size: int,
        hidden_size: int,
        num_layers_lstm: int,
        bidirectional_lstm: bool,
        dropout_cnn_out: float,
        dropout_lstm: float,
        dropout_encoder_features: float,
        control_mode: str = "keyboard",
        sequence_size: int = 5,
    ):
        """
        INIT

        :param int embedded_size: Size of the output embedding
        :param float dropout_cnn_out: Dropout rate for the output of the CNN
        :param str cnn_model_name: Name of the CNN model from torchvision.models
        :param bool pretrained_cnn: If True, the model will be loaded with pretrained weights
        :param int embedded_size: Size of the input feature vectors
        :param int hidden_size: LSTM hidden size
        :param int num_layers_lstm: number of layers in the LSTM
        :param bool bidirectional_lstm: forward or bidirectional LSTM
        :param float dropout_lstm: dropout probability for the LSTM
        :param float dropout_encoder_features: Dropout probability of the encoder output
        :param int sequence_size: Length of the input sequence
        :param control_mode: Model output format: keyboard (Classification task: 9 classes) or controller (Regression task: 2 variables)
        """
        # Constructor method to initialize the model with hyperparameters.
        # Konstruktormethode, um das Modell mit Hyperparametern zu initialisieren.

        super(TEDD1104LSTM, self).__init__()
        # Initialize the parent class (nn.Module).
        # Initialisieren der Elternklasse (nn.Module).

        # Remember hyperparameters.
        # Speichern der Hyperparameter.
        self.cnn_model_name: str = cnn_model_name
        self.pretrained_cnn: bool = pretrained_cnn
        self.sequence_size: int = sequence_size
        self.embedded_size: int = embedded_size
        self.hidden_size: int = hidden_size
        self.num_layers_lstm: int = num_layers_lstm
        self.bidirectional_lstm: bool = bidirectional_lstm
        self.dropout_cnn_out: float = dropout_cnn_out
        self.dropout_lstm: float = dropout_lstm
        self.dropout_encoder_features = dropout_encoder_features
        self.control_mode = control_mode
        # Store hyperparameters as instance variables.
        # Speichern der Hyperparameter als Instanzvariablen.

        self.EncoderCNN: EncoderCNN = EncoderCNN(
            embedded_size=embedded_size,
            dropout_cnn_out=dropout_cnn_out,
            cnn_model_name=cnn_model_name,
            pretrained_cnn=pretrained_cnn,
        )
        # Initialize CNN encoder, which will extract features from images.
        # Initialisieren des CNN-Coders, der Merkmale aus Bildern extrahiert.

        self.EncoderRNN: EncoderRNN = EncoderRNN(
            embedded_size=embedded_size,
            hidden_size=hidden_size,
            num_layers=num_layers_lstm,
            bidirectional_lstm=bidirectional_lstm,
            dropout_lstm=dropout_lstm,
        )
        # Initialize LSTM (RNN) encoder, which will process the sequence of image features.
        # Initialisieren des LSTM (RNN)-Encoders, der die Sequenz der Bildmerkmale verarbeitet.

        self.OutputLayer: OutputLayer = OutputLayer(
            d_model=embedded_size if not self.bidirectional_lstm else embedded_size * 2,
            num_classes=9 if self.control_mode == "keyboard" else 2,
            dropout_encoder_features=self.dropout_encoder_features,
            from_transformer=False,
        )
        # Initialize the output layer which will predict controller input (9 classes or 2 variables).
        # Initialisieren der Ausgabeschicht, die die Controller-Eingabe vorhersagen wird (9 Klassen oder 2 Variablen).

    def forward(
        self, x: torch.tensor, attention_mask: torch.tensor = None
    ) -> torch.tensor:
        """
        Forward pass

        :param torch.tensor x: Input tensor of shape [batch_size * sequence_size, 3, 270, 480]
        :param torch.tensor attention_mask: For compatibility with the Transformer model, this is not used
        :return: Output tensor of shape [9] if control_mode == "keyboard" or [2] if control_mode == "controller"
        """
        # Forward pass through the model.
        # Vorwärtsdurchlauf durch das Modell.

        x = self.EncoderCNN(x)
        # Pass input through CNN encoder to extract features.
        # Eingabe durch den CNN-Encoder zur Merkmalsextraktion leiten.

        x = self.EncoderRNN(x)
        # Pass the CNN output through the LSTM encoder for sequence processing.
        # Das CNN-Ausgangssignal durch den LSTM-Encoder zur Sequenzverarbeitung leiten.

        return self.OutputLayer(x)
        # Finally, pass the LSTM output through the output layer for prediction.
        # Schließlich das LSTM-Ausgangssignal durch die Ausgabeschicht zur Vorhersage leiten.



class TEDD1104Transformer(nn.Module):  # Define the class TEDD1104Transformer which inherits from nn.Module. / Definiert die Klasse TEDD1104Transformer, die von nn.Module erbt.
    """
    T.E.D.D 1104 model with transformer encoder. The model consists of:  # Docstring explaining the purpose of the model. / Docstring, die den Zweck des Modells erklärt.
         - A CNN that extract features from the images  # CNN for feature extraction. / CNN zur Merkmalsextraktion.
         - A transformer that extracts a representation of the image sequence  # Transformer for sequence representation. / Transformer für die Sequenzdarstellung.
         - A linear output layer that predicts the controller input.  # Linear layer for predicting controller input. / Linearer Layer zur Vorhersage des Controller-Eingangs.
    """

    def __init__(  # Initialization method for the class. / Initialisierungsmethode für die Klasse.
        self,
        cnn_model_name: str,  # CNN model name (e.g., 'resnet18'). / CNN Modellname (z.B. 'resnet18').
        pretrained_cnn: bool,  # If True, use pretrained CNN weights. / Wenn True, werden vortrainierte CNN-Gewichte verwendet.
        embedded_size: int,  # Size of the output embedding. / Größe des Ausgabeeinbettungsvektors.
        nhead: int,  # Number of attention heads in the transformer. / Anzahl der Aufmerksamkeitsköpfe im Transformer.
        num_layers_transformer: int,  # Number of layers in the transformer encoder. / Anzahl der Schichten im Transformer-Encoder.
        dropout_cnn_out: float,  # Dropout rate after CNN output. / Dropout-Rate nach dem CNN-Ausgang.
        positional_embeddings_dropout: float,  # Dropout rate for positional embeddings. / Dropout-Rate für Positionseinbettungen.
        dropout_transformer: float,  # Dropout rate for transformer layers. / Dropout-Rate für Transformer-Schichten.
        dropout_encoder_features: float,  # Dropout rate for encoder features. / Dropout-Rate für Encoder-Features.
        control_mode: str = "keyboard",  # Control mode: 'keyboard' (classification) or 'controller' (regression). / Steuerungsmodus: 'keyboard' (Klassifikation) oder 'controller' (Regression).
        sequence_size: int = 5,  # Length of the input sequence. / Länge der Eingabesequenz.
    ):
        """
        INIT  # Initializing hyperparameters. / Initialisierung der Hyperparameter.

        :param int embedded_size: Size of the output embedding / Größe des Ausgabeeinbettungsvektors
        :param float dropout_cnn_out: Dropout rate for the output of the CNN / Dropout-Rate für den Ausgang des CNN
        :param str cnn_model_name: Name of the CNN model from torchvision.models / Name des CNN-Modells aus torchvision.models
        :param bool pretrained_cnn: If True, the model will be loaded with pretrained weights / Wenn True, wird das Modell mit vortrainierten Gewichten geladen.
        :param int nhead: Number of heads in the multi-head attention / Anzahl der Köpfe in der Multi-Head-Attention
        :param int num_layers_transformer: Number of transformer layers in the encoder / Anzahl der Transformer-Schichten im Encoder
        :param float positional_embeddings_dropout: Dropout rate for the positional embeddings / Dropout-Rate für die Positionseinbettungen
        :param float dropout_transformer: Dropout probability of transformer layers / Dropout-Wahrscheinlichkeit der Transformer-Schichten
        :param int sequence_size: Length of the input sequence / Länge der Eingabesequenz
        :param float dropout_encoder_features: Dropout probability of the encoder output / Dropout-Wahrscheinlichkeit des Encoder-Ausgangs
        :param control_mode: Model output format: keyboard (Classification task: 9 classes) or controller (Regression task: 2 variables) / Modell-Ausgabeformat: keyboard (Klassifikationsaufgabe: 9 Klassen) oder controller (Regressionsaufgabe: 2 Variablen)
        """
        super(TEDD1104Transformer, self).__init__()  # Call the parent constructor (initializes nn.Module). / Ruft den Konstruktor der Elternklasse auf (initialisiert nn.Module).

        # Remember hyperparameters. / Merken Sie sich die Hyperparameter.
        self.cnn_model_name: str = cnn_model_name  # Save CNN model name. / Speichern des CNN-Modellnamens.
        self.pretrained_cnn: bool = pretrained_cnn  # Save if pretrained CNN is used. / Speichern, ob ein vortrainiertes CNN verwendet wird.
        self.sequence_size: int = sequence_size  # Save sequence size. / Speichern der Sequenzgröße.
        self.embedded_size: int = embedded_size  # Save the embedding size. / Speichern der Einbettungsgröße.
        self.nhead: int = nhead  # Save the number of heads for attention. / Speichern der Anzahl der Köpfe für die Aufmerksamkeit.
        self.num_layers_transformer: int = num_layers_transformer  # Save the number of transformer layers. / Speichern der Anzahl der Transformer-Schichten.
        self.dropout_cnn_out: float = dropout_cnn_out  # Save CNN dropout rate. / Speichern der CNN-Dropout-Rate.
        self.positional_embeddings_dropout: float = positional_embeddings_dropout  # Save dropout for positional embeddings. / Speichern des Dropouts für Positionseinbettungen.
        self.dropout_transformer: float = dropout_transformer  # Save transformer dropout rate. / Speichern der Transformer-Dropout-Rate.
        self.control_mode = control_mode  # Save control mode (keyboard/controller). / Speichern des Steuerungsmodus (keyboard/controller).
        self.dropout_encoder_features = dropout_encoder_features  # Save dropout for encoder features. / Speichern des Dropouts für Encoder-Features.

        # Define CNN Encoder. / Definieren des CNN-Encoders.
        self.EncoderCNN: EncoderCNN = EncoderCNN(
            embedded_size=embedded_size,  # Pass embedding size. / Übergabe der Einbettungsgröße.
            dropout_cnn_out=dropout_cnn_out,  # Pass CNN dropout rate. / Übergabe der CNN-Dropout-Rate.
            cnn_model_name=cnn_model_name,  # Pass CNN model name. / Übergabe des CNN-Modellnamens.
            pretrained_cnn=pretrained_cnn,  # Pass if pretrained CNN is used. / Übergabe, ob ein vortrainiertes CNN verwendet wird.
            sequence_size=self.sequence_size,  # Pass sequence size. / Übergabe der Sequenzgröße.
        )

        # Define Positional Encoding Layer. / Definieren des Positionsebenen-Codierungs-Layers.
        self.PositionalEncoding = PositionalEmbedding(
            d_model=embedded_size,  # Embedding size for positional encoding. / Einbettungsgröße für Positionseinbettung.
            dropout=self.positional_embeddings_dropout,  # Dropout for positional embedding. / Dropout für Positionseinbettung.
            sequence_length=self.sequence_size,  # Pass sequence length. / Übergabe der Sequenzlänge.
        )

        # Define Transformer Encoder. / Definieren des Transformer-Encoders.
        self.EncoderTransformer: EncoderTransformer = EncoderTransformer(
            d_model=embedded_size,  # Pass embedding size. / Übergabe der Einbettungsgröße.
            nhead=nhead,  # Pass number of attention heads. / Übergabe der Anzahl der Aufmerksamkeitsköpfe.
            num_layers=num_layers_transformer,  # Pass number of transformer layers. / Übergabe der Anzahl der Transformer-Schichten.
            dropout=self.dropout_transformer,  # Pass dropout for transformer layers. / Übergabe des Dropouts für Transformer-Schichten.
        )

        # Define Output Layer. / Definieren des Ausgangs-Layers.
        self.OutputLayer: OutputLayer = OutputLayer(
            d_model=embedded_size,  # Pass embedding size. / Übergabe der Einbettungsgröße.
            num_classes=9 if self.control_mode == "keyboard" else 2,  # Number of output classes based on control mode. / Anzahl der Ausgabeklassen basierend auf dem Steuerungsmodus.
            dropout_encoder_features=dropout_encoder_features,  # Dropout for encoder features. / Dropout für Encoder-Features.
            from_transformer=True,  # Specify output comes from transformer. / Angabe, dass die Ausgabe vom Transformer kommt.
        )

    def forward(  # Define the forward pass method. / Definieren der Vorwärtsdurchgang-Methode.
        self, x: torch.tensor, attention_mask: torch.tensor = None  # Input tensor and optional attention mask. / Eingabe-Tensor und optionale Aufmerksamkeitsmaske.
    ) -> torch.tensor:  # Return tensor output. / Gibt Tensor-Ausgabe zurück.
        """
        Forward pass  # Explanation of forward pass method. / Erklärung der Vorwärtsdurchgang-Methode.

        :param torch.tensor x: Input tensor of shape [batch_size * sequence_size, 3, 270, 480] / Eingabe-Tensor mit der Form [batch_size * sequence_size, 3, 270, 480]
        :param torch.tensor attention_mask: Mask for the input features.  / Maske für die Eingabemerkmale.
                                            [batch_size*heads, sequence_length, sequence_length] 
                                            1 for masked positions and 0 for unmasked positions. / 1 für maskierte Positionen und 0 für nicht maskierte Positionen.
        :return: Output tensor of shape [9] if control_mode == "keyboard" or [2] if control_mode == "controller" / Rückgabe des Tensor-Ausgangs mit der Form [9] bei "keyboard" oder [2] bei "controller".
        """

        x = self.EncoderCNN(x)  # Apply CNN encoder. / Anwendung des CNN-Encoders.
        x = self.PositionalEncoding(x)  # Apply positional encoding. / Anwendung der Positionseinbettung.
        x = self.EncoderTransformer(x, attention_mask=attention_mask)  # Apply transformer encoder with attention mask. / Anwendung des Transformer-Encoders mit Aufmerksamkeitsmaske.
        return self.OutputLayer(x)  # Return output from the final layer. / Rückgabe der Ausgabe aus dem letzten Layer.

class TEDD1104TransformerForImageReordering(nn.Module):  # Define the class TEDD1104TransformerForImageReordering, inheriting from nn.Module / Definiert die Klasse TEDD1104TransformerForImageReordering, die von nn.Module erbt
    """  
    T.E.D.D 1104 for image reordering model consists of:  # A docstring explaining the model components / Eine Beschreibung des Modells
         - A CNN that extract features from the images  # CNN extracts features from images / CNN extrahiert Merkmale aus Bildern
         - A transformer that extracts a representation of the image sequence  # Transformer processes the image sequence representation / Der Transformer verarbeitet die Bildsequenz
         - A linear output layer that predicts the correct order of the input sequence  # The linear layer predicts the correct sequence order / Die lineare Schicht sagt die korrekte Reihenfolge voraus
    """

    def __init__(  # Initialize method for the class / Initialisierungsmethode für die Klasse
        self,
        cnn_model_name: str,  # CNN model name as string (e.g., ResNet) / CNN-Modellname als Zeichenkette (z.B. ResNet)
        pretrained_cnn: bool,  # Whether to load a pretrained CNN model / Ob ein vortrainiertes CNN-Modell geladen wird
        embedded_size: int,  # Size of the embedding vectors / Größe der Embedding-Vektoren
        nhead: int,  # Number of attention heads for the transformer / Anzahl der Attention Heads im Transformer
        num_layers_transformer: int,  # Number of transformer layers / Anzahl der Transformer-Schichten
        dropout_cnn_out: float,  # Dropout rate for the CNN output / Dropout-Rate für die CNN-Ausgabe
        positional_embeddings_dropout: float,  # Dropout rate for positional embeddings / Dropout-Rate für die Positional Embeddings
        dropout_transformer: float,  # Dropout rate for transformer layers / Dropout-Rate für die Transformer-Schichten
        dropout_encoder_features: float,  # Dropout rate for encoder output features / Dropout-Rate für die Encoder-Ausgabefeatures
        sequence_size: int = 5,  # Length of input sequence (default is 5) / Länge der Eingabesequenz (Standard: 5)
    ):
        """
        INIT  # Constructor docstring / Konstruktor-Dokumentation

        :param int embedded_size: Size of the output embedding  # Size of the embedding vectors / Größe der Embeddings
        :param float dropout_cnn_out: Dropout rate for the output of the CNN  # Dropout probability for CNN output / Dropout-Wahrscheinlichkeit für CNN-Ausgabe
        :param str cnn_model_name: Name of the CNN model from torchvision.models  # CNN model name (e.g., 'resnet50') / Name des CNN-Modells (z.B. 'resnet50')
        :param bool pretrained_cnn: If True, the model will be loaded with pretrained weights  # If pretrained CNN weights are used / Wenn True, verwendet es vortrainierte CNN-Gewichte
        :param int embedded_size: Size of the input feature vectors  # Same as above, specifying embedding size / Siehe oben, gibt die Größe des Embeddings an
        :param int nhead: Number of heads in the multi-head attention  # Number of attention heads for the transformer / Anzahl der Attention Heads im Transformer
        :param int num_layers_transformer: number of transformer layers in the encoder  # Number of layers in the transformer / Anzahl der Schichten im Transformer
        :param float positional_embeddings_dropout: Dropout rate for the positional embeddings  # Dropout probability for positional embeddings / Dropout-Wahrscheinlichkeit für Positional Embeddings
        :param float dropout_transformer: dropout probability of transformer layers in the encoder  # Dropout rate in transformer layers / Dropout-Rate in den Transformer-Schichten
        :param int sequence_size: Length of the input sequence  # Length of the sequence (default: 5) / Länge der Sequenz (Standard: 5)
        :param float dropout_encoder_features: Dropout probability of the encoder output  # Dropout rate for encoder features / Dropout-Rate für Encoder-Features
        :param sequence_size: Length of the input sequence  # Redundant, as it's already specified above / Redundant, da oben bereits angegeben
        """

        super(TEDD1104TransformerForImageReordering, self).__init__()  # Call the parent class constructor (nn.Module) / Aufruf des Konstruktors der Elternklasse (nn.Module)

        # Remember hyperparameters.  # Store the hyperparameters for future use / Speichert die Hyperparameter für spätere Verwendung
        self.cnn_model_name: str = cnn_model_name  # CNN model name / CNN-Modellname
        self.pretrained_cnn: bool = pretrained_cnn  # Pretrained CNN flag / Vortrainiertes CNN-Flag
        self.sequence_size: int = sequence_size  # Input sequence size / Eingabesequenzgröße
        self.embedded_size: int = embedded_size  # Output embedding size / Größe der Ausgabebetrachtungen
        self.nhead: int = nhead  # Attention heads for transformer / Attention Heads im Transformer
        self.num_layers_transformer: int = num_layers_transformer  # Transformer layers / Transformer-Schichten
        self.dropout_cnn_out: float = dropout_cnn_out  # CNN output dropout rate / CNN-Ausgabedropout
        self.positional_embeddings_dropout: float = positional_embeddings_dropout  # Positional embedding dropout / Positional Embedding Dropout
        self.dropout_transformer: float = dropout_transformer  # Transformer dropout / Transformer Dropout
        self.dropout_encoder_features = dropout_encoder_features  # Encoder feature dropout / Encoder-Feature-Dropout

        # CNN encoder for feature extraction  # Initialize CNN encoder to extract image features / Initialisiere den CNN-Encoder zur Extraktion von Bildmerkmalen
        self.EncoderCNN: EncoderCNN = EncoderCNN(
            embedded_size=embedded_size,
            dropout_cnn_out=dropout_cnn_out,
            cnn_model_name=cnn_model_name,
            pretrained_cnn=pretrained_cnn,
            sequence_size=self.sequence_size,
        )

        # Positional encoding for input sequence  # Initialize positional encoding for transformer / Initialisiere Positional Encoding für den Transformer
        self.PositionalEncoding = PositionalEmbedding(
            d_model=embedded_size,
            dropout=self.positional_embeddings_dropout,
            sequence_length=self.sequence_size,
        )

        # Transformer encoder for sequence representation  # Initialize transformer encoder / Initialisiere den Transformer-Encoder
        self.EncoderTransformer: EncoderTransformer = EncoderTransformer(
            d_model=embedded_size,
            nhead=nhead,
            num_layers=num_layers_transformer,
            dropout=self.dropout_transformer,
        )

        # Output layer for sequence ordering  # Initialize output layer for image sequence ordering / Initialisiere die Ausgabeschicht zur Bildsequenzordnung
        self.OutputLayer: OutputImageOrderingLayer = OutputImageOrderingLayer(
            d_model=embedded_size,
            num_classes=self.sequence_size,
        )

    def forward(  # Forward pass method / Vorwärtspass-Methode
        self, x: torch.tensor, attention_mask: torch.tensor = None  # Input tensor and optional attention mask / Eingabetensor und optionaler Attention-Mask
    ) -> torch.tensor:  # Returns output tensor after processing / Gibt den Ausgabetensor zurück
        """
        Forward pass  # Docstring for forward pass / Dokumentation des Vorwärtspasses

        :param torch.tensor x: Input tensor of shape [batch_size * sequence_size, 3, 270, 480]  # Shape of input images / Form des Eingabebildes
        :param torch.tensor attention_mask: Mask for the input features  # Attention mask indicating which positions to attend to / Attention-Maske, die angibt, welche Positionen beachtet werden sollen
                                            [batch_size*heads, sequence_length, sequence_length]  # Attention mask shape / Maske-Form erklärt
                                            1 for masked positions and 0 for unmasked positions  # Masking explained / Maskierung erklärt
        :return: Output tensor of shape [9] if control_mode == "keyboard" or [2] if control_mode == "controller"  # Output shape explanation / Rückgabe-Form des Outputs
        """
        x = self.EncoderCNN(x)  # Pass input through the CNN to extract features / Eingabe durch das CNN zur Merkmalsextraktion leiten
        x = self.PositionalEncoding(x)  # Add positional encoding to the features / Positional Encoding zu den Merkmalen hinzufügen
        x = self.EncoderTransformer(x, attention_mask=attention_mask)  # Process through the transformer encoder / Durch den Transformer-Encoder verarbeiten
        return self.OutputLayer(x)  # Output layer to predict the correct order of images / Ausgabe durch die Ausgabeschicht zur Vorhersage der korrekten Reihenfolge der Bilder



class Tedd1104ModelPL(pl.LightningModule):  # Define a class for the Tedd1104Model that inherits from Pytorch Lightning's LightningModule, which simplifies model training. / Definiert eine Klasse für das Tedd1104Model, die von Pytorch Lightnings LightningModule erbt, was das Modelltraining vereinfacht.
    """
    Pytorch Lightning module for the Tedd1104Model  # A brief description of the model purpose. / Eine kurze Beschreibung des Zwecks des Modells.
    """

    def __init__(  # The constructor for initializing the class with the following parameters. / Der Konstruktor zur Initialisierung der Klasse mit den folgenden Parametern.
        self,
        cnn_model_name: str,  # Name of the CNN model. / Name des CNN-Modells.
        pretrained_cnn: bool,  # Whether to use a pretrained CNN model. / Ob ein vortrainiertes CNN-Modell verwendet werden soll.
        embedded_size: int,  # Size of the output embedding. / Größe des Ausgabe-Embeddings.
        nhead: int,  # Number of heads in the multi-head attention mechanism. / Anzahl der Köpfe im Multi-Head Attention-Mechanismus.
        num_layers_encoder: int,  # Number of layers in the encoder. / Anzahl der Schichten im Encoder.
        lstm_hidden_size: int,  # Size of the LSTM hidden state. / Größe des LSTM-Verborgenen Zustands.
        dropout_cnn_out: float,  # Dropout rate for CNN output. / Dropout-Rate für die CNN-Ausgabe.
        positional_embeddings_dropout: float,  # Dropout rate for the positional embeddings. / Dropout-Rate für die Positional Embeddings.
        dropout_encoder: float,  # Dropout rate for the encoder. / Dropout-Rate für den Encoder.
        dropout_encoder_features: float = 0.8,  # Dropout rate for encoder features, default is 0.8. / Dropout-Rate für Encoder-Features, Standardwert ist 0,8.
        control_mode: str = "keyboard",  # Defines control mode, default is "keyboard". / Definiert den Steuerungsmodus, Standard ist "keyboard".
        sequence_size: int = 5,  # Length of the input sequence. / Länge der Eingabesequenz.
        encoder_type: str = "transformer",  # Type of encoder, either "transformer" or "lstm". / Typ des Encoders, entweder "transformer" oder "lstm".
        bidirectional_lstm=True,  # If True, uses a bidirectional LSTM. / Wenn True, wird ein bidirektionales LSTM verwendet.
        weights: List[float] = None,  # Weights for the loss function. / Gewichtungen für die Verlustfunktion.
        label_smoothing: float = 0.0,  # Label smoothing parameter for classification tasks. / Label-Smoothing-Parameter für Klassifizierungsaufgaben.
        accelerator: str = None,  # Accelerator for training (e.g., "cpu", "gpu"). / Beschleuniger für das Training (z.B. "cpu", "gpu").
        optimizer_name: str = "adamw",  # Name of the optimizer to use. / Name des zu verwendenden Optimierers.
        scheduler_name: str = "linear",  # Scheduler name for learning rate adjustment. / Name des Schedulers zur Anpassung der Lernrate.
        learning_rate: float = 1e-5,  # Learning rate for the optimizer. / Lernrate für den Optimierer.
        weight_decay: float = 1e-3,  # Weight decay for regularization. / Gewichtungsverfall zur Regularisierung.
        num_warmup_steps: int = 0,  # Number of warmup steps for scheduler. / Anzahl der Aufwärmschritte für den Scheduler.
        num_training_steps: int = 0,  # Total number of training steps. / Gesamtzahl der Trainingsschritte.
    ):
        """
        INIT  # Initialization docstring, explaining parameters. / Initialisierungs-Dokumentation, die die Parameter erklärt.

        :param int embedded_size: Size of the output embedding / Größe des Ausgabe-Embeddings
        :param float dropout_cnn_out: Dropout rate for the output of the CNN / Dropout-Rate für die Ausgabe des CNN
        :param str cnn_model_name: Name of the CNN model from torchvision.models / Name des CNN-Modells von torchvision.models
        :param bool pretrained_cnn: If True, the model will be loaded with pretrained weights / Wenn True, wird das Modell mit vortrainierten Gewichten geladen
        :param int embedded_size: Size of the input feature vectors / Größe der Eingabefeatur-Vektoren
        :param int nhead: Number of heads in the multi-head attention / Anzahl der Köpfe im Multi-Head Attention
        :param int num_layers_encoder: number of transformer layers in the encoder / Anzahl der Transformer-Schichten im Encoder
        :param float positional_embeddings_dropout: Dropout rate for the positional embeddings / Dropout-Rate für die Positional Embeddings
        :param int sequence_size: Length of the input sequence / Länge der Eingabesequenz
        :param float dropout_encoder: Dropout rate for the encoder / Dropout-Rate für den Encoder
        :param float dropout_encoder_features: Dropout probability of the encoder output / Dropout-Wahrscheinlichkeit für die Encoder-Ausgabe
        :param int lstm_hidden_size: LSTM hidden size / LSTM-Verborgene Größe
        :param bool bidirectional_lstm: forward or bidirectional LSTM / Vorwärts- oder bidirektionales LSTM
        :param List[float] weights: List of weights for the loss function [9] if control_mode == "keyboard" or [2] if control_mode == "controller" / Liste der Gewichtungen für die Verlustfunktion [9], wenn control_mode == "keyboard", oder [2], wenn control_mode == "controller"
        :param str control_mode: Model output format: keyboard (Classification task: 9 classes) or controller (Regression task: 2 variables) / Modell-Ausgabeformat: keyboard (Klassifikationsaufgabe: 9 Klassen) oder controller (Regressionsaufgabe: 2 Variablen)
        :param str encoder_type: Encoder type: transformer or lstm / Encoder-Typ: transformer oder lstm
        :param float label_smoothing: Label smoothing for the classification task / Label Smoothing für die Klassifikationsaufgabe
        :param str optimizer_name: Optimizer to use: adamw or adafactor / Optimierer, der verwendet werden soll: adamw oder adafactor
        :param str scheduler_name: Scheduler to use: linear, polynomial, cosine, plateau / Scheduler zur Verwendung: linear, polynomial, cosine, plateau
        :param float learning_rate: Learning rate / Lernrate
        :param float weight_decay: Weight decay / Gewichtungsverfall
        :param int num_warmup_steps: Number of warmup steps for the scheduler / Anzahl der Aufwärmschritte für den Scheduler
        :param int num_training_steps: Number of training steps / Anzahl der Trainingsschritte
        """

        super(Tedd1104ModelPL, self).__init__()  # Initialize the parent class LightningModule. / Initialisiere die übergeordnete Klasse LightningModule.

        self.encoder_type = encoder_type.lower()  # Store the encoder type as lowercase. / Speichere den Encoder-Typ in Kleinbuchstaben.
        assert self.encoder_type in [
            "lstm",
            "transformer",
        ], f"Encoder type {self.encoder_type} not supported, supported feature encoders [lstm,transformer]."  # Ensure the encoder type is valid. / Stelle sicher, dass der Encoder-Typ gültig ist.

        self.control_mode = control_mode.lower()  # Store the control mode as lowercase. / Speichere den Steuerungsmodus in Kleinbuchstaben.

        assert self.control_mode in [
            "keyboard",
            "controller",
        ], f"{self.control_mode} control mode not supported. Supported dataset types: [keyboard, controller]."  # Ensure the control mode is valid. / Stelle sicher, dass der Steuerungsmodus gültig ist.

        self.cnn_model_name: str = cnn_model_name  # Store CNN model name. / Speichere den CNN-Modellnamen.
        self.pretrained_cnn: bool = pretrained_cnn  # Store whether the CNN is pretrained. / Speichere, ob das CNN vortrainiert ist.
        self.sequence_size: int = sequence_size  # Store the input sequence size. / Speichere die Eingabesequenzgröße.
        self.embedded_size: int = embedded_size  # Store the embedding size. / Speichere die Einbettungsgröße.
        self.nhead: int = nhead  # Store the number of attention heads. / Speichere die Anzahl der Aufmerksamkeitsköpfe.
        self.num_layers_encoder: int = num_layers_encoder  # Store the number of encoder layers. / Speichere die Anzahl der Encoder-Schichten.
        self.dropout_cnn_out: float = dropout_cnn_out  # Store CNN output dropout rate. / Speichere die Dropout-Rate der CNN-Ausgabe.
        self.positional_embeddings_dropout: float = positional_embeddings_dropout  # Store positional embeddings dropout rate. / Speichere die Dropout-Rate der Positional Embeddings.
        self.dropout_encoder: float = dropout_encoder  # Store encoder dropout rate. / Speichere die Dropout-Rate des Encoders.
        self.dropout_encoder_features = dropout_encoder_features  # Store dropout for encoder features. / Speichere das Dropout für die Encoder-Features.
        self.bidirectional_lstm = bidirectional_lstm  # Store if LSTM is bidirectional. / Speichere, ob das LSTM bidirektional ist.
        self.lstm_hidden_size = lstm_hidden_size  # Store LSTM hidden size. / Speichere die LSTM-verborgene Größe.
        self.weights = weights  # Store weights for loss function. / Speichere Gewichtungen für die Verlustfunktion.
        self.label_smoothing = label_smoothing  # Store label smoothing value. / Speichere den Wert für Label Smoothing.
        self.accelerator = accelerator  # Store accelerator type. / Speichere den Typ des Beschleunigers.
        self.learning_rate = learning_rate  # Store the learning rate. / Speichere die Lernrate.
        self.weight_decay = weight_decay  # Store weight decay value. / Speichere den Wert für Gewichtungsverfall.
        self.optimizer_name = optimizer_name  # Store the optimizer name. / Speichere den Namen des Optimierers.
        self.scheduler_name = scheduler_name  # Store the scheduler name. / Speichere den Namen des Schedulers.
        self.num_warmup_steps = num_warmup_steps  # Store the number of warmup steps. / Speichere die Anzahl der Aufwärmschritte.
        self.num_training_steps = num_training_steps  # Store the number of training steps. / Speichere die Anzahl der Trainingsschritte.


if self.encoder_type == "transformer":  # If using a transformer encoder, instantiate the transformer model. / Wenn ein Transformer-Encoder verwendet wird, instanziiere das Transformer-Modell.
    self.model = TEDD1104Transformer(  # Instantiate the TEDD1104Transformer model. / Instanziiere das TEDD1104Transformer-Modell.
        cnn_model_name=self.cnn_model_name,  # Set the CNN model name for the transformer. / Setze den CNN-Modellnamen für den Transformer.
        pretrained_cnn=self.pretrained_cnn,  # Use the pretrained CNN model if specified. / Verwende das vortrainierte CNN-Modell, wenn angegeben.
        embedded_size=self.embedded_size,  # Set the size of the embedded features. / Setze die Größe der eingebetteten Merkmale.
        nhead=self.nhead,  # Set the number of attention heads for the transformer. / Setze die Anzahl der Attention Heads für den Transformer.
        num_layers_transformer=self.num_layers_encoder,  # Set the number of transformer layers. / Setze die Anzahl der Transformer-Schichten.
        dropout_cnn_out=self.dropout_cnn_out,  # Set dropout rate for CNN output. / Setze die Dropout-Rate für den CNN-Ausgang.
        positional_embeddings_dropout=self.positional_embeddings_dropout,  # Set dropout for positional embeddings. / Setze die Dropout-Rate für Positions-Embedding.
        dropout_transformer=self.dropout_encoder,  # Set dropout rate for the transformer. / Setze die Dropout-Rate für den Transformer.
        control_mode=self.control_mode,  # Set the control mode for the model. / Setze den Steuerungsmodus für das Modell.
        sequence_size=self.sequence_size,  # Set the sequence size for the model. / Setze die Sequenzgröße für das Modell.
        dropout_encoder_features=self.dropout_encoder_features,  # Set dropout rate for encoder features. / Setze die Dropout-Rate für die Encoder-Merkmale.
    )
else:  # If using an LSTM encoder, instantiate the LSTM model. / Wenn ein LSTM-Encoder verwendet wird, instanziiere das LSTM-Modell.
    self.model = TEDD1104LSTM(  # Instantiate the TEDD1104LSTM model. / Instanziiere das TEDD1104LSTM-Modell.
        cnn_model_name=self.cnn_model_name,  # Set the CNN model name for the LSTM. / Setze den CNN-Modellnamen für das LSTM.
        pretrained_cnn=self.pretrained_cnn,  # Use the pretrained CNN model if specified. / Verwende das vortrainierte CNN-Modell, wenn angegeben.
        embedded_size=self.embedded_size,  # Set the size of the embedded features. / Setze die Größe der eingebetteten Merkmale.
        hidden_size=self.lstm_hidden_size,  # Set the LSTM hidden size. / Setze die LSTM-Verborgengröße.
        num_layers_lstm=self.num_layers_encoder,  # Set the number of LSTM layers. / Setze die Anzahl der LSTM-Schichten.
        bidirectional_lstm=self.bidirectional_lstm,  # Set if the LSTM should be bidirectional. / Setze, ob das LSTM bidirektional sein soll.
        dropout_cnn_out=self.dropout_cnn_out,  # Set dropout rate for CNN output. / Setze die Dropout-Rate für den CNN-Ausgang.
        dropout_lstm=self.dropout_encoder,  # Set dropout rate for LSTM. / Setze die Dropout-Rate für das LSTM.
        control_mode=self.control_mode,  # Set the control mode for the model. / Setze den Steuerungsmodus für das Modell.
        sequence_size=self.sequence_size,  # Set the sequence size for the model. / Setze die Sequenzgröße für das Modell.
        dropout_encoder_features=self.dropout_encoder_features,  # Set dropout rate for encoder features. / Setze die Dropout-Rate für die Encoder-Merkmale.
    )



        
self.total_batches = 0  # Initializes the total batch count to 0. / Setzt die Gesamtanzahl der Batches auf 0.
self.running_loss = 0  # Initializes the running loss to 0. / Setzt den aktuellen Verlust auf 0.

if version.parse(torchmetrics.__version__) < version.parse("0.10.0"):  # Checks if the version of torchmetrics is less than 0.10.0. / Überprüft, ob die Version von torchmetrics kleiner als 0.10.0 ist.
    accuracy_fn = torchmetrics.Accuracy  # If the version is older, use the Accuracy class. / Wenn die Version älter ist, wird die Accuracy-Klasse verwendet.
else:
    accuracy_fn = torchmetrics.classification.MulticlassAccuracy  # If the version is newer, use the MulticlassAccuracy class. / Wenn die Version neuer ist, wird die MulticlassAccuracy-Klasse verwendet.

self.train_accuracy = accuracy_fn(num_classes=9, top_k=1, average="macro")  # Initializes the accuracy function for training with macro-average for 9 classes. / Initialisiert die Genauigkeitsfunktion für das Training mit dem Makro-Durchschnitt für 9 Klassen.

self.test_accuracy_k1_macro = accuracy_fn(  # Initializes the accuracy function for test with top_k=1 and macro average. / Initialisiert die Genauigkeitsfunktion für den Test mit top_k=1 und Makro-Durchschnitt.
    num_classes=9, top_k=1, average="macro"
)

self.test_accuracy_k3_micro = accuracy_fn(  # Initializes the accuracy function for test with top_k=1 and micro average. / Initialisiert die Genauigkeitsfunktion für den Test mit top_k=1 und Mikro-Durchschnitt.
    num_classes=9, top_k=1, average="micro"
)

self.validation_accuracy_k1_micro = accuracy_fn(  # Initializes the accuracy function for validation with top_k=1 and micro average. / Initialisiert die Genauigkeitsfunktion für die Validierung mit top_k=1 und Mikro-Durchschnitt.
    num_classes=9, top_k=1, average="micro"
)

self.validation_accuracy_k3_micro = accuracy_fn(  # Initializes the accuracy function for validation with top_k=3 and micro average. / Initialisiert die Genauigkeitsfunktion für die Validierung mit top_k=3 und Mikro-Durchschnitt.
    num_classes=9, top_k=3, average="micro"
)

self.validation_accuracy_k1_macro = accuracy_fn(  # Initializes the accuracy function for validation with top_k=1 and macro average. / Initialisiert die Genauigkeitsfunktion für die Validierung mit top_k=1 und Makro-Durchschnitt.
    num_classes=9, top_k=1, average="macro"
)

self.validation_accuracy_k3_macro = accuracy_fn(  # Initializes the accuracy function for validation with top_k=3 and macro average. / Initialisiert die Genauigkeitsfunktion für die Validierung mit top_k=3 und Makro-Durchschnitt.
    num_classes=9, top_k=3, average="macro"
)

self.test_accuracy_k1_micro = accuracy_fn(  # Initializes the accuracy function for test with top_k=1 and micro average (repeated). / Initialisiert die Genauigkeitsfunktion für den Test mit top_k=1 und Mikro-Durchschnitt (wiederholt).
    num_classes=9, top_k=1, average="micro"
)

self.test_accuracy_k3_micro = accuracy_fn(  # Initializes the accuracy function for test with top_k=3 and micro average (repeated). / Initialisiert die Genauigkeitsfunktion für den Test mit top_k=3 und Mikro-Durchschnitt (wiederholt).
    num_classes=9, top_k=3, average="micro"
)

self.test_accuracy_k1_macro = accuracy_fn(  # Initializes the accuracy function for test with top_k=1 and macro average (repeated). / Initialisiert die Genauigkeitsfunktion für den Test mit top_k=1 und Makro-Durchschnitt (wiederholt).
    num_classes=9, top_k=1, average="macro"
)

self.test_accuracy_k3_macro = accuracy_fn(  # Initializes the accuracy function for test with top_k=3 and macro average (repeated). / Initialisiert die Genauigkeitsfunktion für den Test mit top_k=3 und Makro-Durchschnitt (wiederholt).
    num_classes=9, top_k=3, average="macro"
)

if self.control_mode == "keyboard":  # Checks if the control mode is set to "keyboard". / Überprüft, ob der Steuerungsmodus auf "keyboard" gesetzt ist.
    self.criterion = CrossEntropyLoss(  # Initializes the loss function as CrossEntropyLoss. / Initialisiert die Verlustfunktion als CrossEntropyLoss.
        weights=self.weights, label_smoothing=self.label_smoothing  # Sets the weights and label smoothing for the loss function. / Setzt die Gewichtung und Label-Smoothing für die Verlustfunktion.
    )
    self.Keyboard2Controller = Keyboard2Controller()  # Initializes the keyboard-to-controller conversion object. / Initialisiert das Objekt zur Umwandlung von Tastatur in Controller.
else:  # If control mode is not "keyboard", it assumes "controller". / Wenn der Steuerungsmodus nicht "keyboard" ist, wird "controller" angenommen.
    self.validation_distance = torchmetrics.MeanSquaredError()  # Initializes a metric for Mean Squared Error for validation. / Initialisiert eine Metrik für den Mittelwert der quadratischen Fehler für die Validierung.
    self.criterion = WeightedMseLoss(weights=self.weights)  # Initializes the loss function as Weighted Mean Squared Error. / Initialisiert die Verlustfunktion als gewichteten Mittelwert der quadratischen Fehler.
    self.Controller2Keyboard = Controller2Keyboard()  # Initializes the controller-to-keyboard conversion object. / Initialisiert das Objekt zur Umwandlung von Controller in Tastatur.

self.save_hyperparameters()  # Saves the model's hyperparameters. / Speichert die Hyperparameter des Modells.


def forward(self, x, output_mode: str = "keyboard", return_best: bool = True):  # Defines the forward pass of the model with input data and output mode. / Definiert den Vorwärtsdurchgang des Modells mit Eingabedaten und Ausgabemodus.
    """
    Forward pass of the model.

    :param x: input data [batch_size * sequence_size, 3, 270, 480]
    :param output_mode: output mode, either "keyboard" or "controller". If the model uses another mode, we will convert the output to the desired mode.
    :param return_best: if True, we will return the class probabilities, else we will return the class with the highest probability (only for "keyboard" output_mode)
    """
    x = self.model(x)  # Passes input data through the model. / Leitet die Eingabedaten durch das Modell.
    if self.control_mode == "keyboard":  # Checks if the control mode is "keyboard". / Überprüft, ob der Steuerungsmodus "keyboard" ist.
        x = torch.nn.functional.softmax(x, dim=1)  # Applies softmax function to the output for classification. / Wendet die Softmax-Funktion auf die Ausgabe für die Klassifikation an.
        if output_mode == "keyboard":  # Checks if the output mode is "keyboard". / Überprüft, ob der Ausgabemodus "keyboard" ist.
            if return_best:  # If return_best is True, returns the class with the highest probability. / Wenn return_best True ist, wird die Klasse mit der höchsten Wahrscheinlichkeit zurückgegeben.
                return torch.argmax(x, dim=1)  # Returns the class with the highest probability. / Gibt die Klasse mit der höchsten Wahrscheinlichkeit zurück.
            else:
                return x  # Returns class probabilities if return_best is False. / Gibt die Klassenzugehörigkeitswahrscheinlichkeiten zurück, wenn return_best False ist.

        elif output_mode == "controller":  # If the output mode is "controller", converts the output to controller actions. / Wenn der Ausgabemodus "controller" ist, wird die Ausgabe in Controller-Aktionen umgewandelt.
            return self.Keyboard2Controller(x)  # Converts keyboard input to controller output. / Wandelt Tastatureingaben in Controller-Ausgaben um.
        else:
            raise ValueError(  # Raises an error if the output mode is invalid. / Löst einen Fehler aus, wenn der Ausgabemodus ungültig ist.
                f"Output mode: {output_mode} not supported. Supported modes: [keyboard,controller]"
            )

    elif self.control_mode == "controller":  # Checks if the control mode is "controller". / Überprüft, ob der Steuerungsmodus "controller" ist.
        if output_mode == "controller":  # If the output mode is "controller", returns the output as is. / Wenn der Ausgabemodus "controller" ist, wird die Ausgabe unverändert zurückgegeben.
            return x  # Returns the controller output. / Gibt die Controller-Ausgabe zurück.
        elif output_mode == "keyboard":  # If the output mode is "keyboard", converts the controller output to keyboard actions. / Wenn der Ausgabemodus "keyboard" ist, wird die Controller-Ausgabe in Tastatur-Aktionen umgewandelt.
            if return_best:  # If return_best is True, returns the class with the highest probability. / Wenn return_best True ist, wird die Klasse mit der höchsten Wahrscheinlichkeit zurückgegeben.
                return self.argmax(self.Controller2Keyboard(x), dim=-1)  # Converts controller to keyboard actions. / Wandelt Controller in Tastatur-Aktionen um.
            else:
                return self.Controller2Keyboard(x)  # Returns the controller to keyboard output without conversion. / Gibt die Ausgabe von Controller zu Tastatur ohne Umwandlung zurück.
        else:
            raise ValueError(  # Raises an error if the output mode is invalid. / Löst einen Fehler aus, wenn der Ausgabemodus ungültig ist.
                f"Output mode: {output_mode} not supported. Supported modes: [keyboard,controller]"
            )

    else:
        raise ValueError(  # Raises an error if the control mode is invalid. / Löst einen Fehler aus, wenn der Steuerungsmodus ungültig ist.
            f"Control mode: {self.control_mode} not supported. Supported modes: [keyboard,controller]"
        )

def training_step(self, batch, batch_idx):
    """
    Training step.  # Training step, where the model learns from the data. / Trainingsschritt, bei dem das Modell aus den Daten lernt.

    :param batch: batch of data  # A batch of data passed to the model. / Ein Batch von Daten, der dem Modell übergeben wird.
    :param batch_idx: batch index  # Index of the current batch. / Index des aktuellen Batches.
    """
    x, attention_mask, y = batch["images"], batch["attention_mask"], batch["y"]  # Extracts images, attention mask, and labels from the batch. / Extrahiert Bilder, Aufmerksamkeit-Maske und Labels aus dem Batch.
    # x = torch.flatten(x, start_dim=0, end_dim=1)  # Optional line to flatten images, currently commented out. / Optionale Zeile zum Flatten von Bildern, derzeit auskommentiert.
    preds = self.model(x, attention_mask)  # Passes the images and attention mask through the model to get predictions. / Gibt die Bilder und Aufmerksamkeit-Maske durch das Modell, um Vorhersagen zu erhalten.
    loss = self.criterion(preds, y)  # Calculates the loss (error) between predictions and true labels. / Berechnet den Verlust (Fehler) zwischen den Vorhersagen und den tatsächlichen Labels.
    self.total_batches += 1  # Increments the batch count. / Erhöht den Batch-Zähler.
    if self.accelerator != "tpu":  # Checks if the accelerator is not a TPU. / Überprüft, ob der Accelerator kein TPU ist.
        self.running_loss += loss.item()  # Adds the current loss to the running loss. / Addiert den aktuellen Verlust zum laufenden Verlust.
        self.log("Train/loss", loss, sync_dist=True)  # Logs the loss value for training. / Protokolliert den Verlustwert für das Training.
        self.log(
            "Train/running_loss",
            self.running_loss / self.total_batches,
            sync_dist=True,
        )  # Logs the running average of the loss. / Protokolliert den laufenden Durchschnitt des Verlustes.
    else:
        if self.total_batches % 200 == 0:  # If the batch number is a multiple of 200, log the loss. / Wenn die Batch-Nummer ein Vielfaches von 200 ist, wird der Verlust protokolliert.
            self.log("Train/loss", loss, sync_dist=True)  # Logs the loss every 200 batches. / Protokolliert den Verlust alle 200 Batches.

    return {"loss": loss}  # Returns the loss value for further processing. / Gibt den Verlustwert für die weitere Verarbeitung zurück.

def validation_step(self, batch, batch_idx):
    """
    Validation step.  # Validierungsschritt, bei dem das Modell auf einem Validierungsdatensatz getestet wird.

    :param batch: batch of data  # A batch of data for validation. / Ein Batch von Daten für die Validierung.
    :param batch_idx: batch index  # Index of the current validation batch. / Index des aktuellen Validierungs-Batches.
    """
    x, y = batch["images"], batch["y"]  # Extracts images and labels from the validation batch. / Extrahiert Bilder und Labels aus dem Validierungs-Batch.
    # x = torch.flatten(x, start_dim=0, end_dim=1)  # Optional flattening, commented out. / Optionales Flattening, auskommentiert.
    preds = self.forward(x, output_mode="keyboard", return_best=False)  # Gets predictions from the model. / Holt Vorhersagen vom Modell.

    return {"preds": preds, "y": y}  # Returns predictions and true labels. / Gibt Vorhersagen und wahre Labels zurück.

def validation_step_end(self, outputs):
    """
    Validation step end.  # Ende des Validierungsschrittes.

    :param outputs: outputs of the validation step  # Outputs from the validation step. / Ergebnisse des Validierungsschrittes.
    """
    self.validation_accuracy_k1_micro(outputs["preds"], outputs["y"])  # Calculates micro-accuracy for top-1 prediction. / Berechnet die Mikro-Genauigkeit für die Top-1-Vorhersage.
    self.validation_accuracy_k3_micro(outputs["preds"], outputs["y"])  # Calculates micro-accuracy for top-3 prediction. / Berechnet die Mikro-Genauigkeit für die Top-3-Vorhersage.
    self.validation_accuracy_k1_macro(outputs["preds"], outputs["y"])  # Calculates macro-accuracy for top-1 prediction. / Berechnet die Makro-Genauigkeit für die Top-1-Vorhersage.
    self.validation_accuracy_k3_macro(outputs["preds"], outputs["y"])  # Calculates macro-accuracy for top-3 prediction. / Berechnet die Makro-Genauigkeit für die Top-3-Vorhersage.

    self.log(
        "Validation/acc_k@1_micro",
        self.validation_accuracy_k1_micro,
    )  # Logs micro-accuracy for top-1 prediction. / Protokolliert die Mikro-Genauigkeit für die Top-1-Vorhersage.
    self.log(
        "Validation/acc_k@3_micro",
        self.validation_accuracy_k3_micro,
    )  # Logs micro-accuracy for top-3 prediction. / Protokolliert die Mikro-Genauigkeit für die Top-3-Vorhersage.

    self.log(
        "Validation/acc_k@1_macro",
        self.validation_accuracy_k1_macro,
    )  # Logs macro-accuracy for top-1 prediction. / Protokolliert die Makro-Genauigkeit für die Top-1-Vorhersage.
    self.log(
        "Validation/acc_k@3_macro",
        self.validation_accuracy_k3_macro,
    )  # Logs macro-accuracy for top-3 prediction. / Protokolliert die Makro-Genauigkeit für die Top-3-Vorhersage.

    def test_step(self, batch, batch_idx, dataset_idx: int = 0):  # Defines the test_step function, where a batch of data and its index are passed as inputs. | Definiert die Funktion test_step, bei der ein Batch von Daten und dessen Index übergeben werden.
        """
        Test step.  # A short description of the purpose of the method. | Eine kurze Beschreibung des Zwecks der Methode.
        
        :param batch: batch of data  # Describes the batch parameter as the data batch for the test. | Beschreibt den Parameter "batch" als das Daten-Batch für den Test.
        :param batch_idx: batch index  # Describes the batch index. | Beschreibt den Index des Batches.
        """
        x, y = batch["images"], batch["y"]  # Extracts the input images (x) and labels (y) from the batch. | Extrahiert die Eingabebilder (x) und Labels (y) aus dem Batch.
        # x = torch.flatten(x, start_dim=0, end_dim=1)  # A commented-out line that would flatten the input images if used. | Eine auskommentierte Zeile, die die Eingabebilder bei Verwendung flach machen würde.
        preds = self.forward(x, output_mode="keyboard", return_best=False)  # Makes predictions by passing input data through the model's forward method. | Macht Vorhersagen, indem die Eingabedaten durch die Vorwärtsmethode des Modells geleitet werden.

        return {"preds": preds, "y": y}  # Returns the predictions and true labels as a dictionary. | Gibt die Vorhersagen und die wahren Labels als Dictionary zurück.
    
    def test_step_end(self, outputs):  # Defines the end of the test step function, where outputs from the test_step are received. | Definiert das Ende der Testschrittfunktion, bei der die Ausgaben des Testschritts empfangen werden.
        """
        Test step end.  # A short description of the purpose of the method. | Eine kurze Beschreibung des Zwecks der Methode.
        
        :param outputs: outputs of the test step  # Describes the outputs parameter as the result from the test_step function. | Beschreibt den Parameter "outputs" als das Ergebnis der Funktion test_step.
        """
        self.test_accuracy_k1_micro(outputs["preds"], outputs["y"])  # Calculates micro-accuracy for top-1 prediction using the test outputs. | Berechnet die Mikrogenauigkeit für die Top-1 Vorhersage anhand der Testergebnisse.
        self.test_accuracy_k3_micro(outputs["preds"], outputs["y"])  # Calculates micro-accuracy for top-3 prediction. | Berechnet die Mikrogenauigkeit für die Top-3 Vorhersage.
        self.test_accuracy_k1_macro(outputs["preds"], outputs["y"])  # Calculates macro-accuracy for top-1 prediction. | Berechnet die Makrogenauigkeit für die Top-1 Vorhersage.
        self.test_accuracy_k3_macro(outputs["preds"], outputs["y"])  # Calculates macro-accuracy for top-3 prediction. | Berechnet die Makrogenauigkeit für die Top-3 Vorhersage.

        self.log(  # Logs the micro-accuracy for top-1 prediction. | Protokolliert die Mikrogenauigkeit für die Top-1 Vorhersage.
            "Test/acc_k@1_micro",  
            self.test_accuracy_k1_micro,
        )
        self.log(  # Logs the micro-accuracy for top-3 prediction. | Protokolliert die Mikrogenauigkeit für die Top-3 Vorhersage.
            "Test/acc_k@3_micro",
            self.test_accuracy_k3_micro,
        )

        self.log(  # Logs the macro-accuracy for top-1 prediction. | Protokolliert die Makrogenauigkeit für die Top-1 Vorhersage.
            "Test/acc_k@1_macro",
            self.test_accuracy_k1_macro,
        )
        self.log(  # Logs the macro-accuracy for top-3 prediction. | Protokolliert die Makrogenauigkeit für die Top-3 Vorhersage.
            "Test/acc_k@3_macro",
            self.test_accuracy_k3_macro,
        )

    def configure_optimizers(self):  # Defines the method for configuring optimizers and learning rate schedulers. | Definiert die Methode zum Konfigurieren von Optimierern und Lernraten-Schedulern.
        """
        Configure optimizers.  # A short description of the method's purpose. | Eine kurze Beschreibung des Zwecks der Methode.
        """
        if self.optimizer_name.lower() == "adamw":  # Checks if the optimizer is AdamW. | Überprüft, ob der Optimierer AdamW ist.
            optimizer = get_adamw(  # Calls a function to get an AdamW optimizer with the given parameters. | Ruft eine Funktion auf, um einen AdamW-Optimierer mit den angegebenen Parametern zu erhalten.
                parameters=self.parameters(),
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name.lower() == "adafactor":  # Checks if the optimizer is Adafactor. | Überprüft, ob der Optimierer Adafactor ist.
            optimizer = get_adafactor(  # Calls a function to get an Adafactor optimizer. | Ruft eine Funktion auf, um einen Adafactor-Optimierer zu erhalten.
                parameters=self.parameters(),
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:  # Raises an error if the optimizer is not supported. | Löst einen Fehler aus, wenn der Optimierer nicht unterstützt wird.
            raise ValueError(
                f"Unsupported optimizer {self.optimizer_name.lower()}. Choose from adamw and adafactor."  # Error message for unsupported optimizer. | Fehlermeldung für nicht unterstützten Optimierer.
            )

        if self.scheduler_name.lower() == "plateau":  # Checks if the scheduler is Plateau. | Überprüft, ob der Scheduler Plateau ist.
            scheduler = get_reducelronplateau(optimizer=optimizer)  # Uses ReduceLROnPlateau scheduler. | Verwendet den ReduceLROnPlateau-Scheduler.
        elif self.scheduler_name.lower() == "linear":  # Checks if the scheduler is Linear. | Überprüft, ob der Scheduler Linear ist.
            scheduler = get_linear_schedule_with_warmup(  # Uses a linear scheduler with warmup. | Verwendet einen linearen Scheduler mit Aufwärmphase.
                optimizer=optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_training_steps,
            )
        elif self.scheduler_name.lower() == "polynomial":  # Checks if the scheduler is Polynomial. | Überprüft, ob der Scheduler Polynomial ist.
            scheduler = get_polynomial_decay_schedule_with_warmup(  # Uses a polynomial decay scheduler with warmup. | Verwendet einen polynomialen Abkling-Scheduler mit Aufwärmphase.
                optimizer=optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_training_steps,
            )

        elif self.scheduler_name.lower() == "cosine":  # Checks if the scheduler is Cosine. | Überprüft, ob der Scheduler Cosine ist.
            scheduler = get_cosine_schedule_with_warmup(  # Uses a cosine scheduler with warmup. | Verwendet einen Cosine-Scheduler mit Aufwärmphase.
                optimizer=optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_training_steps,
            )
        else:  # Raises an error if the scheduler is not supported. | Löst einen Fehler aus, wenn der Scheduler nicht unterstützt wird.
            raise ValueError(
                f"Unsupported scheduler {self.scheduler_name.lower()}. Choose from linear, polynomial, cosine, plateau."  # Error message for unsupported scheduler. | Fehlermeldung für nicht unterstützten Scheduler.
            )

        return [optimizer], [  # Returns the optimizer and scheduler configuration. | Gibt die Optimierer- und Scheduler-Konfiguration zurück.
            {
                "scheduler": scheduler,
                "monitor": "Validation/acc_k@1_macro",  # Monitors the macro accuracy for validation. | Überwacht die Makrogenauigkeit für die Validierung.
                "interval": "epoch",  # Defines the interval at which the scheduler is applied. | Definiert das Intervall, in dem der Scheduler angewendet wird.
            }
            if self.scheduler_name.lower() == "plateau"  # If Plateau scheduler is used, monitor every epoch. | Wenn der Plateau-Scheduler verwendet wird, wird jede Epoche überwacht.
            else {"scheduler": scheduler, "interval": "step", "frequency": 1}  # For other schedulers, apply every step. | Für andere Scheduler wird jeder Schritt angewendet.
        ]

class Tedd1104ModelPLForImageReordering(pl.LightningModule):  # Defines a class for a PyTorch Lightning model to reorder images / Definiert eine Klasse für ein PyTorch Lightning-Modell zur Bildneuanordnung.
    """
    Pytorch Lightning module for the Tedd1104ModelForImageReordering model  # Documentation string describing the module / Dokumentationsstring, der das Modul beschreibt
    """

    def __init__(  # Constructor to initialize the model with parameters / Konstruktor zur Initialisierung des Modells mit Parametern
        self,  # Reference to the instance of the class / Referenz auf die Instanz der Klasse
        cnn_model_name: str,  # The name of the CNN model to use / Der Name des CNN-Modells, das verwendet werden soll
        pretrained_cnn: bool,  # Whether to use a pretrained CNN model or not / Ob ein vortrainiertes CNN-Modell verwendet werden soll oder nicht
        embedded_size: int,  # The size of the embedding output / Die Größe des Einbettungsausgangs
        nhead: int,  # Number of attention heads in the transformer encoder / Anzahl der Aufmerksamkeitsköpfe im Transformer-Encoder
        num_layers_encoder: int,  # Number of layers in the encoder / Anzahl der Schichten im Encoder
        dropout_cnn_out: float,  # Dropout rate for CNN output / Dropout-Rate für die CNN-Ausgabe
        positional_embeddings_dropout: float,  # Dropout rate for positional embeddings / Dropout-Rate für Positions-Einbettungen
        dropout_encoder: float,  # Dropout rate for the encoder layers / Dropout-Rate für die Encoder-Schichten
        dropout_encoder_features: float = 0.8,  # Dropout rate for encoder features, default is 0.8 / Dropout-Rate für Encoder-Features, Standardwert ist 0.8
        sequence_size: int = 5,  # Length of the sequence input / Länge der Eingabesequenz
        encoder_type: str = "transformer",  # Type of encoder to use: transformer or LSTM / Art des zu verwendenden Encoders: Transformer oder LSTM
        accelerator: str = None,  # Accelerator to use, such as "gpu" / Accelerator, der verwendet werden soll, wie "gpu"
        optimizer_name: str = "adamw",  # Optimizer type: AdamW or AdaFactor / Optimierer-Typ: AdamW oder AdaFactor
        scheduler_name: str = "linear",  # Scheduler type: linear or plateau / Scheduler-Typ: linear oder plateau
        learning_rate: float = 1e-5,  # Learning rate for training / Lernrate für das Training
        weight_decay: float = 1e-3,  # Weight decay to prevent overfitting / Gewichtungsabnahme zur Vermeidung von Überanpassung
        num_warmup_steps: int = 0,  # Number of warm-up steps for the scheduler / Anzahl der Warm-up-Schritte für den Scheduler
        num_training_steps: int = 0,  # Total number of training steps / Gesamtzahl der Trainingsschritte
    ):

        """
        INIT

        :param int embedded_size: Size of the output embedding / Größe der Ausgabeeinbettung
        :param float dropout_cnn_out: Dropout rate for the output of the CNN / Dropout-Rate für die Ausgabe des CNN
        :param str cnn_model_name: Name of the CNN model from torchvision.models / Name des CNN-Modells von torchvision.models
        :param bool pretrained_cnn: If True, the model will be loaded with pretrained weights / Wenn True, wird das Modell mit vortrainierten Gewichten geladen
        :param int embedded_size: Size of the input feature vectors / Größe der Eingabefeature-Vektoren
        :param int nhead: Number of heads in the multi-head attention / Anzahl der Köpfe in der Multi-Head-Attention
        :param int num_layers_encoder: Number of transformer layers in the encoder / Anzahl der Transformer-Schichten im Encoder
        :param float positional_embeddings_dropout: Dropout rate for the positional embeddings / Dropout-Rate für Positions-Einbettungen
        :param int sequence_size: Length of the input sequence / Länge der Eingabesequenz
        :param float dropout_encoder: Dropout rate for the encoder / Dropout-Rate für den Encoder
        :param float dropout_encoder_features: Dropout probability of the encoder output / Dropout-Wahrscheinlichkeit der Encoder-Ausgabe
        :param str encoder_type: Encoder type: transformer or lstm / Encoder-Typ: transformer oder lstm
        :param str optimizer_name: Optimizer to use: adamw or adafactor / Optimierer, der verwendet werden soll: adamw oder adafactor
        :param str scheduler_name: Scheduler to use: linear or plateau / Scheduler, der verwendet werden soll: linear oder plateau
        :param float learning_rate: Learning rate / Lernrate
        :param float weight_decay: Weight decay / Gewichtungsabnahme
        :param int num_warmup_steps: Number of warmup steps for the scheduler / Anzahl der Warm-up-Schritte für den Scheduler
        :param int num_training_steps: Number of training steps / Anzahl der Trainingsschritte
        """

        super(Tedd1104ModelPLForImageReordering, self).__init__()  # Initialize the parent class LightningModule / Initialisiert die Elternklasse LightningModule
        assert encoder_type == "transformer", "Only transformer encoder is supported"  # Ensure that only transformer encoder is used / Sicherstellen, dass nur der Transformer-Encoder verwendet wird
        self.cnn_model_name: str = cnn_model_name  # Store CNN model name / Speichert den Namen des CNN-Modells
        self.pretrained_cnn: bool = pretrained_cnn  # Store whether CNN is pretrained / Speichert, ob das CNN vortrainiert ist
        self.sequence_size: int = sequence_size  # Store sequence size / Speichert die Sequenzgröße
        self.embedded_size: int = embedded_size  # Store embedded size / Speichert die Größe der Einbettung
        self.nhead: int = nhead  # Store the number of attention heads / Speichert die Anzahl der Aufmerksamkeitsköpfe
        self.num_layers_encoder: int = num_layers_encoder  # Store the number of encoder layers / Speichert die Anzahl der Encoder-Schichten
        self.dropout_cnn_out: float = dropout_cnn_out  # Store dropout rate for CNN output / Speichert die Dropout-Rate für die CNN-Ausgabe
        self.positional_embeddings_dropout: float = positional_embeddings_dropout  # Store dropout rate for positional embeddings / Speichert die Dropout-Rate für Positions-Einbettungen
        self.dropout_encoder: float = dropout_encoder  # Store dropout rate for the encoder / Speichert die Dropout-Rate für den Encoder
        self.dropout_encoder_features = dropout_encoder_features  # Store dropout probability for encoder features / Speichert die Dropout-Wahrscheinlichkeit für Encoder-Features
        self.encoder_type = encoder_type  # Store encoder type / Speichert den Encoder-Typ
        self.accelerator = accelerator  # Store accelerator type / Speichert den Accelerator-Typ
        self.learning_rate = learning_rate  # Store learning rate / Speichert die Lernrate
        self.weight_decay = weight_decay  # Store weight decay rate / Speichert die Gewichtungsabnahme
        self.optimizer_name = optimizer_name  # Store optimizer type / Speichert den Optimierer-Typ
        self.scheduler_name = scheduler_name  # Store scheduler type / Speichert den Scheduler-Typ
        self.num_warmup_steps = num_warmup_steps  # Store number of warmup steps / Speichert die Anzahl der Warm-up-Schritte
        self.num_training_steps = num_training_steps  # Store number of training steps / Speichert die Anzahl der Trainingsschritte

      self.model = TEDD1104TransformerForImageReordering(  # Create the model instance for image reordering / Erstellt die Modellinstanz für die Bildneuanordnung
        cnn_model_name=self.cnn_model_name,  # Specify the name of the CNN model / Gibt den Namen des CNN-Modells an
        pretrained_cnn=self.pretrained_cnn,  # Use a pretrained CNN model if specified / Verwendet ein vortrainiertes CNN-Modell, wenn angegeben
        embedded_size=self.embedded_size,  # Define the size of the embedding vector / Definiert die Größe des Einbettungsvektors
        nhead=self.nhead,  # Set the number of attention heads in the transformer model / Legt die Anzahl der Aufmerksamkeitsköpfe im Transformer-Modell fest
        num_layers_transformer=self.num_layers_encoder,  # Set the number of layers in the transformer encoder / Legt die Anzahl der Schichten im Transformer-Encoder fest
        dropout_cnn_out=self.dropout_cnn_out,  # Define the dropout rate for the CNN output / Definiert die Dropout-Rate für die CNN-Ausgabe
        positional_embeddings_dropout=self.positional_embeddings_dropout,  # Set the dropout rate for positional embeddings / Legt die Dropout-Rate für Positions-Einbettungen fest
        dropout_transformer=self.dropout_encoder,  # Set the dropout rate for the transformer encoder / Legt die Dropout-Rate für den Transformer-Encoder fest
        sequence_size=self.sequence_size,  # Define the size of the input sequence / Definiert die Größe der Eingabesequenz
        dropout_encoder_features=self.dropout_encoder_features,  # Set the dropout rate for the features of the transformer encoder / Legt die Dropout-Rate für die Merkmale des Transformer-Encoders fest
    )


        self.total_batches = 0  # Initialize total batch count / Initialisiert die Gesamtzahl der Batches
        self.running_loss = 0  # Initialize running loss / Initialisiert den laufenden Verlust

        self.train_accuracy = ImageReorderingAccuracy()  # Initialize train accuracy metric / Initialisiert die Trainingsgenauigkeitsmetrik
        self.validation_accuracy = ImageReorderingAccuracy()  # Initialize validation accuracy metric / Initialisiert die Validierungsgenauigkeitsmetrik
        self.test_accuracy = ImageReorderingAccuracy()  # Initialize test accuracy metric / Initialisiert die Testgenauigkeitsmetrik

        self.criterion = CrossEntropyLossImageReorder()  # Initialize loss function for image reordering / Initialisiert die Verlustfunktion für die Bildneuanordnung

        self.save_hyperparameters()  # Save all hyperparameters for later use / Speichert alle Hyperparameter für die spätere Verwendung


def forward(self, x, return_best: bool = True):  
    # Defines the forward pass of the model, taking input 'x' and a flag 'return_best' to decide the output.
    # Definiert den Vorwärtsdurchgang des Modells, das die Eingabe 'x' und eine Flagge 'return_best' übernimmt, um den Ausgang zu bestimmen.

    """
    Forward pass of the model.  
    # A docstring explaining that this function performs the forward pass for the model.
    # Eine Dokumentation, die erklärt, dass diese Funktion den Vorwärtsdurchgang für das Modell durchführt.

    :param x: input data [batch_size * sequence_size, 3, 270, 480]  
    # Input data, where 'x' is expected to be a 4D tensor with shape [batch_size * sequence_size, 3, 270, 480].
    # Eingabedaten, wobei 'x' ein 4D-Tensor mit der Form [batch_size * sequence_size, 3, 270, 480] erwartet.

    :param return_best: if True, we will return the class probabilities, else we will return the class with the highest probability  
    # The 'return_best' flag determines whether to return the class probabilities or just the class with the highest probability.
    # Das Flag 'return_best' bestimmt, ob die Klassenzugehörigkeiten oder nur die Klasse mit der höchsten Wahrscheinlichkeit zurückgegeben wird.

    """
    x = self.model(x)  
    # Passes the input 'x' through the model to get the prediction or output.
    # Leitet die Eingabe 'x' durch das Modell, um die Vorhersage oder den Ausgang zu erhalten.

    if return_best:  
        # Checks if the 'return_best' flag is True.
        # Überprüft, ob das Flag 'return_best' wahr ist.

        return torch.argmax(x, dim=-1)  
        # If True, returns the class with the highest probability by finding the maximum value along the last dimension.
        # Wenn wahr, gibt es die Klasse mit der höchsten Wahrscheinlichkeit zurück, indem der maximale Wert entlang der letzten Dimension gesucht wird.

    else:  
        # If 'return_best' is False, returns the raw output (class probabilities).
        # Wenn 'return_best' falsch ist, gibt es die rohen Ausgaben (Klassenzugehörigkeiten) zurück.

        return x  
        # Returns the raw class probabilities.
        # Gibt die rohen Klassenzugehörigkeiten zurück.


def training_step(self, batch, batch_idx):  
    # Defines a single training step for processing a batch of data.
    # Definiert einen einzelnen Trainingsschritt zur Verarbeitung eines Datenbatches.

    """
    Training step.  
    # A docstring explaining that this function performs a training step.
    # Eine Dokumentation, die erklärt, dass diese Funktion einen Trainingsschritt durchführt.

    :param batch: batch of data  
    # The input 'batch' contains a set of data for training.
    # Das Eingabebatch enthält eine Reihe von Daten für das Training.

    :param batch_idx: batch index  
    # The 'batch_idx' is the index of the current batch in the dataset.
    # Der 'batch_idx' ist der Index des aktuellen Batches im Datensatz.

    """
    x, y = batch["images"], batch["y"]  
    # Extracts the image data (x) and the ground truth labels (y) from the batch.
    # Extrahiert die Bilddaten (x) und die echten Beschriftungen (y) aus dem Batch.

    # x = torch.flatten(x, start_dim=0, end_dim=1)  
    # This line is commented out, but if enabled, it would flatten the image tensor across specified dimensions.
    # Diese Zeile ist auskommentiert, aber wenn aktiviert, würde sie den Bild-Tensor über die angegebenen Dimensionen abflachen.

    preds = self.model(x)  
    # Passes the image data (x) through the model to get predictions.
    # Leitet die Bilddaten (x) durch das Modell, um Vorhersagen zu erhalten.

    loss = self.criterion(preds, y)  
    # Calculates the loss by comparing the model's predictions (preds) to the ground truth labels (y).
    # Berechnet den Verlust, indem die Vorhersagen des Modells (preds) mit den echten Beschriftungen (y) verglichen werden.

    self.total_batches += 1  
    # Increments the batch counter.
    # Erhöht den Batch-Zähler.

    if self.accelerator != "tpu":  
        # Checks if the model is not running on a TPU.
        # Überprüft, ob das Modell nicht auf einem TPU ausgeführt wird.

        self.running_loss += loss.item()  
        # Adds the current loss value to the running total loss.
        # Fügt den aktuellen Verlustwert zum kumulierten Verlust hinzu.

        self.log("Train/loss", loss, sync_dist=True)  
        # Logs the current loss for training.
        # Protokolliert den aktuellen Verlust für das Training.

        self.log(  
            "Train/running_loss",  
            self.running_loss / self.total_batches,  
            sync_dist=True,  
        )  
        # Logs the running average loss for training.
        # Protokolliert den laufenden Durchschnittsverlust für das Training.

    else:  
        # If running on a TPU, logs the loss every 200 batches.
        # Wenn auf einem TPU ausgeführt, wird der Verlust alle 200 Batches protokolliert.

        if self.total_batches % 200 == 0:  
            # Every 200 batches, log the loss.
            # Alle 200 Batches wird der Verlust protokolliert.

            self.log("Train/loss", loss, sync_dist=True)  
            # Logs the loss for every 200 batches.
            # Protokolliert den Verlust für alle 200 Batches.

    return {"loss": loss}  
    # Returns the computed loss value as a dictionary.
    # Gibt den berechneten Verlustwert als Dictionary zurück.


    def validation_step(self, batch, batch_idx):  # Define a method for a single validation step, taking in a batch of data and its index. / Definiert eine Methode für einen einzelnen Validierungsschritt, der ein Batch von Daten und seinen Index übernimmt.
        """
        Validation step.  # A docstring that describes the purpose of this function. / Eine Dokumentationszeichenkette, die den Zweck dieser Funktion beschreibt.

        :param batch: batch of data  # Description of the parameter 'batch', which represents a batch of data. / Beschreibung des Parameters 'batch', der ein Batch von Daten darstellt.
        :param batch_idx: batch index  # Description of the parameter 'batch_idx', which is the index of the batch. / Beschreibung des Parameters 'batch_idx', der der Index des Batches ist.
        """
        x, y = batch["images"], batch["y"]  # Extract images (x) and labels (y) from the batch. / Extrahiere Bilder (x) und Labels (y) aus dem Batch.
        # x = torch.flatten(x, start_dim=0, end_dim=1)  # This line is commented out, but would flatten the images if used. / Diese Zeile ist auskommentiert, würde jedoch die Bilder flach machen, wenn sie verwendet wird.
        preds = self.forward(x, return_best=True)  # Perform a forward pass on the images to get predictions. / Führe einen Vorwärtspass auf den Bildern aus, um Vorhersagen zu erhalten.

        return {"preds": preds, "y": y}  # Return the predictions and the ground truth labels as a dictionary. / Gib die Vorhersagen und die tatsächlichen Labels als Dictionary zurück.

    def validation_step_end(self, outputs):  # Define the end of the validation step, taking the outputs of the previous step. / Definiert das Ende des Validierungsschritts, der die Ausgaben des vorherigen Schritts übernimmt.
        """
        Validation step end.  # A docstring that describes the purpose of this function. / Eine Dokumentationszeichenkette, die den Zweck dieser Funktion beschreibt.

        :param outputs: outputs of the validation step  # Description of the parameter 'outputs', which contains the results of the validation step. / Beschreibung des Parameters 'outputs', der die Ergebnisse des Validierungsschritts enthält.
        """
        self.validation_accuracy(outputs["preds"], outputs["y"])  # Calculate and log the validation accuracy based on predictions and ground truth. / Berechne und protokolliere die Validierungsgenauigkeit basierend auf den Vorhersagen und den tatsächlichen Werten.
        self.log(  # Log the validation accuracy to monitor during training. / Protokolliere die Validierungsgenauigkeit, um sie während des Trainings zu überwachen.
            "Validation/acc",  # Name of the metric being logged. / Name der protokollierten Kennzahl.
            self.validation_accuracy,  # Value of the validation accuracy. / Wert der Validierungsgenauigkeit.
        )

    def test_step(self, batch, batch_idx, dataset_idx: int = 0):  # Define a method for a single test step, with additional optional 'dataset_idx' parameter. / Definiert eine Methode für einen einzelnen Testschritt mit einem zusätzlichen optionalen 'dataset_idx'-Parameter.
        """
        Test step.  # A docstring that describes the purpose of this function. / Eine Dokumentationszeichenkette, die den Zweck dieser Funktion beschreibt.

        :param batch: batch of data  # Description of the parameter 'batch', which represents a batch of data. / Beschreibung des Parameters 'batch', der ein Batch von Daten darstellt.
        :param batch_idx: batch index  # Description of the parameter 'batch_idx', which is the index of the batch. / Beschreibung des Parameters 'batch_idx', der der Index des Batches ist.
        """
        x, y = batch["images"], batch["y"]  # Extract images (x) and labels (y) from the batch. / Extrahiere Bilder (x) und Labels (y) aus dem Batch.
        # x = torch.flatten(x, start_dim=0, end_dim=1)  # This line is commented out, but would flatten the images if used. / Diese Zeile ist auskommentiert, würde jedoch die Bilder flach machen, wenn sie verwendet wird.
        preds = self.forward(x, return_best=True)  # Perform a forward pass on the images to get predictions. / Führe einen Vorwärtspass auf den Bildern aus, um Vorhersagen zu erhalten.

        return {"preds": preds, "y": y}  # Return the predictions and the ground truth labels as a dictionary. / Gib die Vorhersagen und die tatsächlichen Labels als Dictionary zurück.

def test_step_end(self, outputs):  # Defines the function `test_step_end` that takes the argument `outputs`. / Definiert die Funktion `test_step_end`, die das Argument `outputs` übernimmt.
    """
    Test step end.  # This is a docstring that describes the function's purpose. / Dies ist ein Docstring, der den Zweck der Funktion beschreibt.

    :param outputs: outputs of the test step  # Describes the `outputs` parameter, which contains the results of the test step. / Beschreibt den Parameter `outputs`, der die Ergebnisse des Testschritts enthält.
    """
    self.test_accuracy(outputs["preds"], outputs["y"])  # Calls the `test_accuracy` function with predictions and true values from `outputs`. / Ruft die Funktion `test_accuracy` mit Vorhersagen und wahren Werten aus `outputs` auf.

    self.log(  # Calls the `log` function to record some information. / Ruft die Funktion `log` auf, um Informationen aufzuzeichnen.
        "Test/acc",  # Specifies the log label for the accuracy. / Gibt das Log-Label für die Genauigkeit an.
        self.test_accuracy,  # Logs the accuracy function for future reference. / Zeichnet die Genauigkeitsfunktion für die zukünftige Bezugnahme auf.
    )

def configure_optimizers(self):  # Defines the function `configure_optimizers`. / Definiert die Funktion `configure_optimizers`.
    """
    Configure optimizers.  # This is a docstring that describes the function's purpose. / Dies ist ein Docstring, der den Zweck der Funktion beschreibt.
    """
    if self.optimizer_name.lower() == "adamw":  # Checks if the optimizer is AdamW, case-insensitive. / Überprüft, ob der Optimierer AdamW ist, unabhängig von der Groß-/Kleinschreibung.
        optimizer = get_adamw(  # Calls the `get_adamw` function to create the AdamW optimizer. / Ruft die Funktion `get_adamw` auf, um den AdamW-Optimierer zu erstellen.
            parameters=self.parameters(),  # Passes model parameters to the optimizer. / Übergibt die Modellparameter an den Optimierer.
            learning_rate=self.learning_rate,  # Sets the learning rate for the optimizer. / Legt die Lernrate für den Optimierer fest.
            weight_decay=self.weight_decay,  # Sets the weight decay for regularization. / Legt den Gewichtungsabfall für die Regularisierung fest.
        )
    elif self.optimizer_name.lower() == "adafactor":  # Checks if the optimizer is Adafactor, case-insensitive. / Überprüft, ob der Optimierer Adafactor ist, unabhängig von der Groß-/Kleinschreibung.
        optimizer = get_adafactor(  # Calls the `get_adafactor` function to create the Adafactor optimizer. / Ruft die Funktion `get_adafactor` auf, um den Adafactor-Optimierer zu erstellen.
            parameters=self.parameters(),  # Passes model parameters to the optimizer. / Übergibt die Modellparameter an den Optimierer.
            learning_rate=self.learning_rate,  # Sets the learning rate for the optimizer. / Legt die Lernrate für den Optimierer fest.
            weight_decay=self.weight_decay,  # Sets the weight decay for regularization. / Legt den Gewichtungsabfall für die Regularisierung fest.
        )
    else:  # If neither AdamW nor Adafactor, raises an error. / Wenn weder AdamW noch Adafactor, wird ein Fehler ausgelöst.
        raise ValueError(  # Raises a `ValueError` if the optimizer is unsupported. / Löst einen `ValueError` aus, wenn der Optimierer nicht unterstützt wird.
            f"Unsupported optimizer {self.optimizer_name.lower()}. Choose from adamw and adafactor."  # Provides an error message with allowed optimizers. / Gibt eine Fehlermeldung mit den zulässigen Optimierern aus.
        )

    if self.scheduler_name.lower() == "plateau":  # Checks if the scheduler is 'plateau'. / Überprüft, ob der Scheduler 'plateau' ist.
        scheduler = get_reducelronplateau(optimizer=optimizer)  # Creates the plateau scheduler for the optimizer. / Erstellt den Plateau-Scheduler für den Optimierer.
    elif self.scheduler_name.lower() == "linear":  # Checks if the scheduler is 'linear'. / Überprüft, ob der Scheduler 'linear' ist.
        scheduler = get_linear_schedule_with_warmup(  # Creates the linear scheduler with warm-up for the optimizer. / Erstellt den linearen Scheduler mit Aufwärmphase für den Optimierer.
            optimizer=optimizer,  # Passes the optimizer to the scheduler. / Übergibt den Optimierer an den Scheduler.
            num_warmup_steps=self.num_warmup_steps,  # Sets the number of warm-up steps. / Legt die Anzahl der Aufwärmschritte fest.
            num_training_steps=self.num_training_steps,  # Sets the total number of training steps. / Legt die Gesamtzahl der Trainingsschritte fest.
        )
    else:  # If neither plateau nor linear, raises an error. / Wenn weder Plateau noch Linear, wird ein Fehler ausgelöst.
        raise ValueError(  # Raises a `ValueError` if the scheduler is unsupported. / Löst einen `ValueError` aus, wenn der Scheduler nicht unterstützt wird.
            f"Unsupported scheduler {self.scheduler_name.lower()}. Choose from plateau and linear."  # Provides an error message with allowed schedulers. / Gibt eine Fehlermeldung mit den zulässigen Schedulern aus.
        )

    return [optimizer], [  # Returns the optimizer and scheduler as lists for further use. / Gibt den Optimierer und Scheduler als Listen zurück, um sie weiter zu verwenden.
        {
            "scheduler": scheduler,  # Specifies the scheduler to use. / Gibt den zu verwendenden Scheduler an.
            "monitor": "Validation/acc_k@1_macro",  # Specifies the metric to monitor for stopping. / Gibt die Metrik an, die für das Stoppen überwacht werden soll.
            "interval": "epoch",  # Sets the interval for scheduler updates to 'epoch'. / Legt das Intervall für die Scheduler-Updates auf 'epoch' fest.
        }
        if self.scheduler_name.lower() == "plateau"  # If the scheduler is plateau, uses epoch-based updates. / Wenn der Scheduler Plateau ist, werden epoch-basierte Updates verwendet.
        else {"scheduler": scheduler, "interval": "step", "frequency": 1}  # Otherwise, uses step-based updates. / Andernfalls werden schrittbasierte Updates verwendet.
    ]
