from torch.optim import AdamW  # Import the AdamW optimizer from the PyTorch library. # AdamW-Optimierer aus der PyTorch-Bibliothek importieren.

def get_adamw(parameters, learning_rate: float, weight_decay: float):  
    """  
    Get AdamW optimizer.  # Define a function to create and return an AdamW optimizer.  
    # Funktion definieren, um einen AdamW-Optimierer zu erstellen und zurückzugeben.  
    :param Iterable parameters: The parameters to optimize.  # Parameters for optimization are provided as input.  
    # Zu optimierende Parameter werden als Eingabe bereitgestellt.  
    :param float learning_rate: The learning rate.  # Set the learning rate for the optimizer.  
    # Lernrate für den Optimierer festlegen.  
    :param float weight_decay: The weight decay.  # Set the weight decay for the optimizer to prevent overfitting.  
    # Gewichtungsabnahme zur Vermeidung von Überanpassung festlegen.  
    :return: torch.optim.AdamW. The AdamW optimizer.  # Returns an AdamW optimizer instance.  
    # Gibt eine AdamW-Optimierer-Instanz zurück.  
    """  
    optimizer = AdamW(  
        params=parameters,  # Assign the provided parameters to the optimizer.  
        # Übergebene Parameter dem Optimierer zuweisen.  
        lr=learning_rate,  # Set the learning rate.  
        # Lernrate festlegen.  
        weight_decay=weight_decay,  # Apply weight decay for regularization.  
        # Gewichtungsabnahme zur Regularisierung anwenden.  
        eps=1e-4,  # Set the epsilon value for numerical stability.  
        # Epsilon-Wert für numerische Stabilität festlegen.  
    )  
    return optimizer  # Return the configured optimizer.  
    # Konfigurierten Optimierer zurückgeben.  

def get_adafactor(parameters, learning_rate, weight_decay):  
    """  
    Get Adafactor optimizer (Requires fairseq, pip install fairseq).  # Define a function to create and return an Adafactor optimizer.  
    # Funktion definieren, um einen Adafactor-Optimierer zu erstellen und zurückzugeben.  
    :param Iterable parameters: The parameters to optimize.  # Parameters for optimization are provided as input.  
    # Zu optimierende Parameter werden als Eingabe bereitgestellt.  
    :param float learning_rate: The learning rate.  # Set the learning rate for the optimizer.  
    # Lernrate für den Optimierer festlegen.  
    :param float weight_decay: The weight decay.  # Set the weight decay for the optimizer to prevent overfitting.  
    # Gewichtungsabnahme zur Vermeidung von Überanpassung festlegen.  
    :return: fairseq.optim.adafactor.Adafactor. The Adafactor optimizer.  # Returns an Adafactor optimizer instance.  
    # Gibt eine Adafactor-Optimierer-Instanz zurück.  
    """  
    try:  
        from fairseq.optim.adafactor import Adafactor  # Import Adafactor optimizer from the fairseq library.  
        # Adafactor-Optimierer aus der Fairseq-Bibliothek importieren.  
    except ImportError:  
        raise ImportError(  
            "You need to install fairseq to use Adafactor optimizer."  
            "Run `pip install fairseq`."  # Raise an error if fairseq is not installed and provide installation instructions.  
            # Fehler auslösen, wenn Fairseq nicht installiert ist, und Installationsanweisungen bereitstellen.  
        )  
    optimizer = Adafactor(  
        params=parameters,  # Assign the provided parameters to the optimizer.  
        # Übergebene Parameter dem Optimierer zuweisen.  
        scale_parameter=False,  # Disable parameter scaling.  
        # Parameterskalierung deaktivieren.  
        relative_step=False,  # Disable relative step size.  
        # Relative Schrittgröße deaktivieren.  
        warmup_init=False,  # Disable warmup initialization.  
        # Aufwärm-Initialisierung deaktivieren.  
        lr=learning_rate,  # Set the learning rate.  
        # Lernrate festlegen.  
        clip_threshold=1.0,  # Set the clipping threshold to avoid exploding gradients.  
        # Clipping-Schwelle festlegen, um explodierende Gradienten zu vermeiden.  
        weight_decay=weight_decay,  # Apply weight decay for regularization.  
        # Gewichtungsabnahme zur Regularisierung anwenden.  
    )  

    return optimizer  # Return the configured optimizer.  
    # Konfigurierten Optimierer zurückgeben.  
