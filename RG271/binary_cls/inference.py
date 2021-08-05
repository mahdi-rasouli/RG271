import torch
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


class InferenceModel:
    """Helper class for loading a DistilBERT sentiment analysis model for inference purposes."""

    def __init__(self, input_path):
        """Initialise an InferenceModel.

        Args:
            input_path (str): Path to the PyTorch model to load.
        """

        self._model = self._load_model(input_path)
        self._tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def _load_model(self, input_path):
        """Loads a state dictionary into a DistilBERT model.

        Args:
            input_path (str): Path to the PyTorch model.

        Returns:
            PyTorch model in evaluation mode.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", state_dict=torch.load(input_path)
        )
        model.to(device)
        model.eval()

        return model

    def predict(self, input_str):
        """Apply model to an input string.

        Args:
            input_str (str): String to apply model to.

        Returns:
            TODO
        """

        if isinstance(input_str, str):
            input_str = [input_str]

        encodings = self._tokenizer(input_str, truncation=True, padding=True)

        with torch.no_grad():
            outputs = self._model(
                torch.tensor(encodings["input_ids"]),
                attention_mask=torch.tensor(encodings["attention_mask"]),
            )

            label_id = np.argmax(outputs[0].detach().cpu().numpy(), axis=1)[0].item()
            confidence = outputs.logits.softmax(dim=-1).detach().cpu().max().item()
            label = "Complaint" if label_id == 1 else "OK"

        return {"label_id": label_id, "label": label, "confidence": confidence}
