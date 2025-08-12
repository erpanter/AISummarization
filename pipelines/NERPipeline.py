import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertForTokenClassification
import json
from pathlib import Path

class BertNERPipeline:
    def __init__(self, model_name_or_path, label2id_path=None, id2label_path=None):
        # Get the directory of the current file
        base_dir = Path(__file__).resolve().parent.parent  # go up from /pipelines to project root

        # Default paths if not provided
        if label2id_path is None:
            label2id_path = base_dir / "data" / "labels" / "label2id.json"
        if id2label_path is None:
            id2label_path = base_dir / "data" / "labels" / "id2label.json"

        # Load label mappings from JSON
        with open(label2id_path, "r") as f:
            self.label2id = json.load(f)
        with open(id2label_path, "r") as f:
            self.id2label = json.load(f)
        
        # Convert keys of id2label to int (JSON saves keys as strings)
        self.id2label = {int(k): v for k, v in self.id2label.items()}
        
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
        self.model = BertForTokenClassification.from_pretrained(
            model_name_or_path,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.model.eval()

    def predict(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, is_split_into_words=False)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits  # [1, seq_len, num_labels]
        probs = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=2)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_labels = [self.id2label[pred] for pred in predictions[0].tolist()]
        predicted_scores = [probs[0, idx, pred].item() for idx, pred in enumerate(predictions[0].tolist())]

        results = []
        for token, label, score in zip(tokens, predicted_labels, predicted_scores):
            if token in ["[CLS]", "[SEP]"]:
                continue
            if token.startswith("##"):
                token = token[2:]
            results.append({
                "entity_group": label,
                "word": token,
                "score": score
            })
        return results