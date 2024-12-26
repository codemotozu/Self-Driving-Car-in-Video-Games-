"""
Experimental segmentation module based on SegFormer.
Only intended for testing purposes.
Only supported in inference, may be part of the model in the future.
It uses too much GPU resources, not viable for training or real-time inference yet.
Nvidia pls launch faster GPUs :)

Requires the transformers library from huggingface to be installed (huggingface.co/transformers)
"""
# This block is a docstring explaining the purpose of the code.
# Dies ist ein Docstring, der den Zweck des Codes erklärt.

import torch
# Imports the PyTorch library for tensor computation and neural networks.
# Importiert die PyTorch-Bibliothek für Tensorberechnungen und neuronale Netzwerke.

from torch.nn import functional
# Imports functional API from PyTorch for various operations like interpolation.
# Importiert die funktionale API von PyTorch für verschiedene Operationen wie Interpolation.

from torchvision import transforms
# Imports the transforms module from torchvision for image transformations.
# Importiert das Modul 'transforms' von torchvision für Bildtransformationen.

import numpy as np
# Imports NumPy library for handling arrays.
# Importiert die NumPy-Bibliothek für die Arbeit mit Arrays.

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
# Imports feature extractor and segmentation model from Hugging Face's transformers library.
# Importiert den Feature Extractor und das Segmentierungsmodell aus der Hugging Face Transformers-Bibliothek.

from typing import List, Dict
# Imports List and Dict for type hinting in function signatures.
# Importiert List und Dict für die Typisierung in Funktionssignaturen.


def cityscapes_palette():
    """
    Returns the cityscapes palette.

    :return: List[List[int]] - The cityscapes palette.
    """
    # Defines a function that returns the Cityscapes color palette used for semantic segmentation.
    # Definiert eine Funktion, die die Cityscapes-Farbpalette für die semantische Segmentierung zurückgibt.
    return [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]
    # The function returns a list of RGB color values representing the Cityscapes color palette.
    # Die Funktion gibt eine Liste von RGB-Farbwerten zurück, die die Cityscapes-Farbpalette darstellen.


class SequenceResize(object):
    """Prepares the images for the model"""

    def __init__(self, size=(1024, 1024)):
        """
        INIT

        :param Tuple[int, int] size:  - The size of the output images.
        """
        # Constructor for initializing image size for resizing.
        # Konstruktor für die Initialisierung der Bildgröße zum Ändern der Größe.
        self.size = size

    def __call__(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Applies the transformation to the images.

        :param List[np.ndarray] images: - The images to transform.
        :return: List[np.ndarray] - The transformed images.
        """
        # Applies resizing to the given images.
        # Wendet die Größenänderung auf die angegebenen Bilder an.
        return functional.interpolate(
            images,
            size=self.size,
            mode="bilinear",
            align_corners=False,
        )
        # Uses bilinear interpolation to resize images to the desired size.
        # Verwendet bilineare Interpolation, um die Bilder auf die gewünschte Größe zu ändern.


class ToTensor(object):
    """Convert np.ndarray images to Tensors."""

    def __call__(self, images: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Applies the transformation to the sequence of images.

        :param List[np.ndarray] images: - The images to transform.
        :return: List[torch.Tensor] - The transformed images.
        """
        # Converts a sequence of NumPy images into PyTorch tensors.
        # Konvertiert eine Sequenz von NumPy-Bildern in PyTorch-Tensoren.
        image1, image2, image3, image4, image5 = (
            images[0],
            images[1],
            images[2],
            images[3],
            images[4],
        )

        # Swap color axis from HWC (height, width, channels) to CHW (channels, height, width).
        # Tauscht die Farbachse von HWC (Höhe, Breite, Kanäle) nach CHW (Kanäle, Höhe, Breite).
        image1 = image1.transpose((2, 0, 1)).astype(float)
        image2 = image2.transpose((2, 0, 1)).astype(float)
        image3 = image3.transpose((2, 0, 1)).astype(float)
        image4 = image4.transpose((2, 0, 1)).astype(float)
        image5 = image5.transpose((2, 0, 1)).astype(float)

        return [
            torch.from_numpy(image1),
            torch.from_numpy(image2),
            torch.from_numpy(image3),
            torch.from_numpy(image4),
            torch.from_numpy(image5),
        ]
        # Converts each NumPy image to a tensor and returns them as a list.
        # Konvertiert jedes NumPy-Bild in einen Tensor und gibt diese als Liste zurück.


class MergeImages(object):
    """Merges the images into one torch.Tensor"""

    def __call__(self, images: List[torch.tensor]) -> torch.tensor:
        """
        Applies the transformation to the sequence of images.

        :param List[torch.tensor] images: - The images to transform.
        :return: torch.Tensor - The transformed image.
        """
        # Merges multiple images into a single tensor.
        # Fässt mehrere Bilder zu einem einzigen Tensor zusammen.
        image1, image2, image3, image4, image5 = (
            images[0],
            images[1],
            images[2],
            images[3],
            images[4],
        )

        return torch.stack([image1, image2, image3, image4, image5])
        # Uses torch.stack to stack images along a new dimension, forming one tensor.
        # Verwendet torch.stack, um die Bilder entlang einer neuen Dimension zu stapeln und einen einzigen Tensor zu bilden.


class ImageSegmentation:
    """
    Class for performing image segmentation.
    """

    def __init__(
        self,
        device: torch.device,
        model_name: str = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    ):
        """
        INIT

        :param torch.device device: - The device to use.
        :param str model_name: - The name of the model to use (https://huggingface.co/models)
        """
        # Initializes the image segmentation class with the given device and model.
        # Initialisiert die Bildsegmentierungsklasse mit dem angegebenen Gerät und Modell.
        print(f"Loading feature extractor for {model_name}")
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
        print(f"Loading segmentation model for {model_name}")
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.device = device
        self.model = self.model.to(device=self.device)

        self.image_transforms = transforms.Compose(
            [ToTensor(), MergeImages(), SequenceResize()]
        )
        # Loads the pretrained SegFormer feature extractor and model.
        # Lädt den vortrainierten SegFormer-Feature-Extractor und das Modell.


    def add_segmentation(self, images: np.ndarray) -> np.ndarray:
        """
        Adds the segmentation to the images. The segmentation is added as a mask over the original images.

        :param np.ndarray images: - The images to add the segmentation to.
        :return: np.ndarray - The images with the segmentation added.
        """

        original_image_size = images[0].shape
        # Saves the original size of the image for resizing after segmentation.
        # Speichert die Originalgröße des Bildes für das spätere Ändern der Größe nach der Segmentierung.

        inputs = torch.vstack(
            [
                self.feature_extractor(images=image, return_tensors="pt")[
                    "pixel_values"
                ]
                for image in images
            ]
        ).to(device=self.device)
        # Prepares the input images by passing them through the feature extractor.
        # Bereitet die Eingabebilder vor, indem sie durch den Feature Extractor geleitet werden.

        outputs = self.model(inputs).logits.detach().cpu()
        # Passes the input through the model to get the output logits.
        # Leitet die Eingabe durch das Modell, um die Logits auszugeben.

        logits = functional.interpolate(
            outputs,
            size=(original_image_size[0], original_image_size[1]),
            mode="bilinear",
            align_corners=False,
        )
        # Resizes the output logits back to the original image size.
        # Ändert die Größe der Ausgabelogits auf die Originalgröße des Bildes.

        segmented_images = logits.argmax(dim=1)
        # Takes the highest value from the logits to determine the segmentation mask.
        # Nimmt den höchsten Wert aus den Logits, um die Segmentierungsmaske zu bestimmen.

        for image_no, seg in enumerate(segmented_images):
            color_seg = np.zeros(
                (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
            )  # height, width, 3
            # Creates a blank color segmentation image to apply the segmentation mask.
            # Erstellt ein leeres Farbsegmentierungsbild, um die Segmentierungsmaske anzuwenden.

            palette = np.array(cityscapes_palette())
            for label, color in enumerate(palette):
                color_seg[seg == label, :] = color
            # Uses the palette to color the segmented areas based on the class labels.
            # Verwendet die Farbpalette, um die segmentierten Bereiche basierend auf den Klassennamen zu färben.

            images[image_no] = images[image_no] * 0.5 + color_seg * 0.5
            # Blends the original image with the segmented color mask.
            # Mischt das Originalbild mit der segmentierten Farbmaske.

        return images
        # Returns the images with added segmentation.
        # Gibt die Bilder mit hinzugefügter Segmentierung zurück.
