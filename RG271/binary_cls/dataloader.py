import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        """Initialise a CustomDataset object. This class inherits from the torch Dataset
        class to create an object compatible with the torch DataLoader.
        Args:
            encodings: Tokenized inputs.
            labels: Sentiment label for encoding.
        """

        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """Allow usage of this class as an iterator.
        Args:
            idx: Desired index.
        """

        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        """Allow for usage of len() on an object of this class."""

        return len(self.labels)
