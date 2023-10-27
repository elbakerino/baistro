import torch
from baistro.config.config import AppConfig
from baistro.model_control.model_base import ModelBase
from transformers import AutoImageProcessor, AutoModelForImageClassification


class DitBaseModel(ModelBase):
    name = "microsoft/dit-base-finetuned-rvlcdip"
    url = "hugging"
    folder = f'{AppConfig.MODEL_DIR}/model-{name.replace("/", "_").lower()}'
    id = 'dit-base'
    tasks = ['image-classify']

    _model = None
    _processor = None

    def __init__(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

    @property
    def model(self):
        if not DitBaseModel._model:
            DitBaseModel._model = AutoModelForImageClassification.from_pretrained(self.folder)
        return DitBaseModel._model

    @property
    def processor(self):
        if not DitBaseModel._processor:
            DitBaseModel._processor = AutoImageProcessor.from_pretrained(self.folder)
        return DitBaseModel._processor

    def generate(self, image):
        # todo: turn image to grayscale for dit rvlcdip
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits

        predicted_class_idx = logits.argmax(-1).item()
        return 0, self.model.config.id2label[predicted_class_idx]

    @staticmethod
    def download():
        m = DitBaseModel
        tokenizer = AutoImageProcessor.from_pretrained(m.name)
        model = AutoModelForImageClassification.from_pretrained(m.name)
        tokenizer.save_pretrained(m.folder)
        model.save_pretrained(m.folder)
