import re

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from baistro.config.config import AppConfig
from baistro.model_control.model_base import ModelBase


class DonutToTextModel(ModelBase):
    name = "naver-clova-ix/donut-base-finetuned-cord-v2"
    url = "hugging"
    folder = f'{AppConfig.MODEL_DIR}/model-{name.replace("/", "_").lower()}'
    id = 'donut-to-text'
    tasks = ['image2text']

    _model = None
    _processor = None

    def __init__(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        # self.tokenizer.to(device)

    @property
    def model(self):
        if not DonutToTextModel._model:
            DonutToTextModel._model = VisionEncoderDecoderModel.from_pretrained(self.folder)
        return DonutToTextModel._model

    @property
    def processor(self):
        if not DonutToTextModel._processor:
            DonutToTextModel._processor = DonutProcessor.from_pretrained(self.folder)
        return DonutToTextModel._processor

    def generate(self, image):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # prepare decoder inputs
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        print(f'donut-t-text {self.model.decoder.config.max_position_embeddings}')
        outputs = self.model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=self.model.decoder.config.max_position_embeddings,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token

        return 0, self.processor.token2json(sequence)

    @staticmethod
    def download():
        m = DonutToTextModel
        tokenizer = DonutProcessor.from_pretrained(m.name, resume_download=True)
        model = VisionEncoderDecoderModel.from_pretrained(m.name, resume_download=True)
        tokenizer.save_pretrained(m.folder)
        model.save_pretrained(m.folder)
