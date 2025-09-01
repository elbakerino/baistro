import logging
from typing import Union, List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from tqdm import trange

logger = logging.getLogger(__name__)


class SentenceTransformerModelBase:

    def __init__(
        self,
        model_name_or_path: str | None = None,
        local_files_only: bool = False,
    ):
        self.transformer = SentenceTransformer(model_name_or_path, local_files_only=local_files_only)

    def save(self, path: str):
        return self.transformer.save(path)

    # a modified `encode`, with added tokens stats
    def encode_with_stats(
        self, sentences: Union[List[str], str],
        batch_size: int = 32,
        convert_to_tensor: bool = True,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = False,
    ):
        device = self.transformer.device
        tokens = 0

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-self.transformer._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=True):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            features = self.transformer.tokenize(sentences_batch)
            if 'input_ids' in features:
                tokens += sum(len(inp_ids) for inp_ids in features['input_ids'])
            elif 'pixel_values' in features:
                # todo: not sure if "tokens" should also be used for image models als stat,
                #       but the pixel-values seem to be the only "size metrics" here, except pixels-sizes of course
                tokens += sum(len(pixel_value) for pixel_values in features['pixel_values'] for pixel_value in pixel_values)
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.transformer.forward(features)

                embeddings = out_features['sentence_embedding']
                embeddings = embeddings.detach()
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return tokens, all_embeddings
