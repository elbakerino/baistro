import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from baistro.config.config import AppConfig
from baistro.model_control.model_base import ModelBase


class QagEnBaseModel(ModelBase):
    name = "lmqg/flan-t5-base-squad-qag"
    url = "hugging"
    folder = f'{AppConfig.MODEL_DIR}/model-{name.replace("/", "_").lower()}'
    id = 'qag-en-base'
    tasks = ['qag']
    locale = ['en']

    _model = None
    _tokenizer = None

    def __init__(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        # self.tokenizer.to(device)

    @property
    def model(self):
        if not QagEnBaseModel._model:
            QagEnBaseModel._model = AutoModelForSeq2SeqLM.from_pretrained(self.folder)
        return QagEnBaseModel._model

    @property
    def tokenizer(self):
        if not QagEnBaseModel._tokenizer:
            QagEnBaseModel._tokenizer = AutoTokenizer.from_pretrained(self.folder)
        return QagEnBaseModel._tokenizer

    def generate_with_pipe(self, text):
        pipe = pipeline(
            "text2text-generation", model=self.model, tokenizer=self.tokenizer,
            # seems `80-90` or so is somehow the default max-length, increase if no answer was given/longer input etc.
            max_length=300,
        )
        return pipe(text)

    def generate(self, text):
        encoded_input = self.tokenizer([text], max_length=512, truncation=True, return_tensors="pt")
        tokens = len(encoded_input['input_ids'][0])
        with torch.no_grad():
            output = self.model.generate(
                **encoded_input,
                do_sample=True,
                top_k=0,
                top_p=0.24,
                # with `no_repeat_ngram_size` it forces new questions, but also won't separate consistently
                # no_repeat_ngram_size=4,
                # min_length=10,
                max_length=160,
            )
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return tokens, decoded_output

    @staticmethod
    def download():
        m = QagEnBaseModel
        tokenizer = AutoTokenizer.from_pretrained(m.name)
        model = AutoModelForSeq2SeqLM.from_pretrained(m.name)
        tokenizer.save_pretrained(m.folder)
        model.save_pretrained(m.folder)
