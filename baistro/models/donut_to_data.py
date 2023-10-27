import re

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from baistro.config.config import AppConfig
from baistro.model_control.model_base import ModelBase


class DonutToDataModel(ModelBase):
    name = "katanaml-org/invoices-donut-model-v1"
    url = "hugging"
    folder = f'{AppConfig.MODEL_DIR}/model-{name.replace("/", "_").lower()}'
    id = 'donut-to-data'
    tasks = ['image2text', 'image2data']

    _model = None
    _processor = None

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        # self.tokenizer.to(device)

    @property
    def model(self):
        if not DonutToDataModel._model:
            DonutToDataModel._model = VisionEncoderDecoderModel.from_pretrained(self.folder)
        return DonutToDataModel._model

    @property
    def processor(self):
        if not DonutToDataModel._processor:
            DonutToDataModel._processor = DonutProcessor.from_pretrained(self.folder)
        return DonutToDataModel._processor

    def generate(self, image):
        # prepare encoder inputs
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # prepare decoder inputs
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

        outputs = self.model.generate(
            pixel_values.to(self.device),
            decoder_input_ids=decoder_input_ids.to(self.device),
            max_length=self.model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token

        return 0, self.processor.token2json(sequence)

    @staticmethod
    def download():
        m = DonutToDataModel
        tokenizer = DonutProcessor.from_pretrained(m.name, resume_download=True)
        model = VisionEncoderDecoderModel.from_pretrained(m.name, resume_download=True)
        tokenizer.save_pretrained(m.folder)
        model.save_pretrained(m.folder)
