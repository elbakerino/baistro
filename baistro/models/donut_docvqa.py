import re

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from baistro.config.config import AppConfig
from baistro.model_control.model_base import ModelBase


class DonutDocvqaModel(ModelBase):
    name = "naver-clova-ix/donut-base-finetuned-docvqa"
    url = "hugging"
    folder = f'{AppConfig.MODEL_DIR}/model-{name.replace("/", "_").lower()}'
    id = 'donut-docvqa'
    tasks = ['qnli', 'vqa', 'docvqa']

    _model = None
    _processor = None

    def __init__(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        # self.tokenizer.to(device)

    @property
    def model(self):
        if not DonutDocvqaModel._model:
            DonutDocvqaModel._model = VisionEncoderDecoderModel.from_pretrained(self.folder)
        return DonutDocvqaModel._model

    @property
    def processor(self):
        if not DonutDocvqaModel._processor:
            DonutDocvqaModel._processor = DonutProcessor.from_pretrained(self.folder)
        return DonutDocvqaModel._processor

    def generate(self, question, image):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
        prompt = task_prompt.replace("{user_input}", question)
        decoder_input_ids = self.processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
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
        m = DonutDocvqaModel
        tokenizer = DonutProcessor.from_pretrained(m.name, resume_download=True)
        model = VisionEncoderDecoderModel.from_pretrained(m.name, resume_download=True)
        tokenizer.save_pretrained(m.folder)
        model.save_pretrained(m.folder)
