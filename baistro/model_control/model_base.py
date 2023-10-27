from typing import List, Optional


class ModelBase(object):
    id: str
    # a list of task ids supports
    # - ` ` question-answering natural language inference
    # - `vqa` visual question-answering
    # - `docvqa` document visual question-answering
    # - `qag` question-answer generation
    # - `image2text` image to text / markdown / data
    # - `image2data` image to structured data
    # - `image-classify` image classification
    # - `vector-text` dense vector space generation for texts
    # - `vector-code` dense vector space generation for sourcecode
    # - `vector-image` dense vector space generation for images
    tasks: List[str]
    # the model name (e.g. from huggingface)
    name: str
    # the local folder path
    folder: str
    # the url to the model description (or simply `hugging`)
    url: Optional[str] = None
    locale: Optional[List[str]] = None

    @staticmethod
    def download():
        pass


def model_url(model: ModelBase):
    return "https://huggingface.co/" + model.name if model.url == "hugging" else model.url
