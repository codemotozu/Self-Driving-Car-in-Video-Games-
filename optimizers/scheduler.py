"""
This script uses code from the following sources:
    - Hugginface Transformers: https://github.com/huggingface/transformers/blob/v4.19.2/src/transformers/optimization.py
"""
# English: This script is partially based on code from Hugging Face's Transformers library.
# Deutsch: Dieses Skript basiert teilweise auf Code aus der Transformers-Bibliothek von Hugging Face.

from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import math
from torch.optim import Optimizer
# English: Import necessary modules for learning rate schedulers and math operations.
# Deutsch: Importiert notwendige Module für Lernraten-Scheduler und mathematische Operationen.

def get_reducelronplateau(optimizer, patience: int = 2, verbose=True):
    """
    Reduce LR on Plateay scheduler.
    :param Optimizer optimizer: The optimizer to reduce the learning rate.
    :param int patience: The patience of the scheduler.
    :param bool verbose: If True, the scheduler will print the current learning rate
    :return: torch.optim.lr_scheduler.ReduceLROnPlateau. The scheduler.
    """
    # English: Defines a function to create a ReduceLROnPlateau scheduler with specified patience and verbosity.
    # Deutsch: Definiert eine Funktion, um einen ReduceLROnPlateau-Scheduler mit festgelegter Geduld und Ausgabe zu erstellen.

    scheduler = ReduceLROnPlateau(optimizer, "max", patience=patience, verbose=verbose)
    # English: Creates a scheduler that reduces the learning rate when a metric stops improving.
    # Deutsch: Erstellt einen Scheduler, der die Lernrate reduziert, wenn sich eine Metrik nicht mehr verbessert.

    return scheduler
    # English: Returns the created scheduler.
    # Deutsch: Gibt den erstellten Scheduler zurück.

def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """
    FROM https://github.com/huggingface/transformers/blob/v4.19.2/src/transformers/optimization.py#L233

    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # English: Defines a function for a linear learning rate schedule with a warmup phase.
    # Deutsch: Definiert eine Funktion für einen linearen Lernratenplan mit einer Aufwärmphase.

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # English: During the warmup phase, the learning rate increases linearly.
        # Deutsch: Während der Aufwärmphase steigt die Lernrate linear an.

        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )
        # English: After the warmup phase, the learning rate decreases linearly until it reaches 0.
        # Deutsch: Nach der Aufwärmphase sinkt die Lernrate linear, bis sie 0 erreicht.

    return LambdaLR(optimizer, lr_lambda, last_epoch)
    # English: Returns a LambdaLR scheduler using the defined learning rate function.
    # Deutsch: Gibt einen LambdaLR-Scheduler zurück, der die definierte Lernratenfunktion verwendet.

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    FROM https://github.com/huggingface/transformers/blob/v4.19.2/src/transformers/optimization.py#L233

    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # English: Defines a cosine learning rate schedule with a warmup phase and optional cycles.
    # Deutsch: Definiert einen kosinusförmigen Lernratenplan mit Aufwärmphase und optionalen Zyklen.

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # English: During warmup, the learning rate increases linearly.
        # Deutsch: Während der Aufwärmphase steigt die Lernrate linear an.

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # English: Progress represents the proportion of completed steps beyond warmup.
        # Deutsch: Fortschritt gibt den Anteil der abgeschlossenen Schritte nach der Aufwärmphase an.

        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )
        # English: The learning rate follows a cosine curve after warmup.
        # Deutsch: Nach der Aufwärmphase folgt die Lernrate einer Kosinuskurve.

    return LambdaLR(optimizer, lr_lambda, last_epoch)
    # English: Returns a LambdaLR scheduler with the cosine schedule function.
    # Deutsch: Gibt einen LambdaLR-Scheduler mit der Kosinus-Zeitplanfunktion zurück.

def get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    lr_end=1e-7,
    power=1.0,
    last_epoch=-1,
):
    """
    FROM https://github.com/huggingface/transformers/blob/v4.19.2/src/transformers/optimization.py#L233

    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        lr_end (`float`, *optional*, defaults to 1e-7):
            The end LR.
        power (`float`, *optional*, defaults to 1.0):
            Power factor.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # English: Defines a polynomial decay learning rate schedule with a warmup phase.
    # Deutsch: Definiert einen Lernratenplan mit polynomischem Abfall und einer Aufwärmphase.

    lr_init = optimizer.defaults["lr"]
    # English: Retrieves the initial learning rate from the optimizer.
    # Deutsch: Ruft die anfängliche Lernrate vom Optimizer ab.

    if not (lr_init > lr_end):
        raise ValueError(
            f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})"
        )
    # English: Ensures that the final learning rate is smaller than the initial learning rate.
    # Deutsch: Stellt sicher, dass die endgültige Lernrate kleiner als die anfängliche Lernrate ist.

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # English: During warmup, the learning rate increases linearly.
        # Deutsch: Während der Aufwärmphase steigt die Lernrate linear an.

        elif current_step > num_training_steps:
            return lr_end / lr_init
        # English: After all training steps, the learning rate stays at the end value.
        # Deutsch: Nach allen Trainingsschritten bleibt die Lernrate auf dem Endwert.

        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            # English: Applies polynomial decay to the learning rate after warmup.
            # Deutsch: Wendet polynomischen Abfall auf die Lernrate nach der Aufwärmphase an.

            return decay / lr_init
            # English: Returns the scaled learning rate.
            # Deutsch: Gibt die skalierte Lernrate zurück.

    return LambdaLR(optimizer, lr_lambda, last_epoch)
    # English: Returns a LambdaLR scheduler with the polynomial decay function.
    # Deutsch: Gibt einen LambdaLR-Scheduler mit der Funktion für den polynomischen Abfall zurück.
