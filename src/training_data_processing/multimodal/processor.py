from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from .cross_attention import CrossAttentionFusion


class MultimodalProcessor:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print("Successfully loaded OpenCLIP model")

        feature_dim = self.model.visual.output_dim
        self.cross_attention_fusion = CrossAttentionFusion(
            feature_dim=feature_dim
        ).to(self.device)
        self.cross_attention_fusion.eval()

        self.fusion_methods = {
            "concat": self._concat_fusion,
            "additive": self._additive_fusion,
            "multiplicative": self._multiplicative_fusion,
            "cross_attention": self._cross_attention_fusion,
        }

    def extract_image_features(
        self, images: Union[Image.Image, List[Image.Image]]
    ) -> torch.Tensor:
        if isinstance(images, list):
            processed_images = torch.stack(
                [self.preprocess(image.convert("RGB")) for image in images]
            ).to(self.device)
        else:
            processed_images = self.preprocess(
                images.convert("RGB")
            ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(processed_images)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

        return image_features

    def extract_text_features(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        text_tokens = self.tokenizer(texts).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

        return text_features

    def compute_similarity(
        self, features1: Union[np.ndarray, torch.Tensor],
        features2: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        if torch.is_tensor(features1):
            features1 = features1.cpu().numpy()
        if torch.is_tensor(features2):
            features2 = features2.cpu().numpy()

        return cosine_similarity(features1, features2)

    def align_modalities(
        self,
        images: Union[Image.Image, List[Image.Image]],
        texts: Union[str, List[str]],
    ) -> np.ndarray:
        image_features = self.extract_image_features(images)
        text_features = self.extract_text_features(texts)
        return self.compute_similarity(image_features, text_features)

    def _concat_fusion(self, image_features, text_features):
        if torch.is_tensor(image_features):
            image_features = image_features.cpu().numpy()
        if torch.is_tensor(text_features):
            text_features = text_features.cpu().numpy()

        return np.concatenate([image_features, text_features], axis=1)

    def _additive_fusion(self, image_features, text_features):
        if torch.is_tensor(image_features):
            image_features = image_features.cpu().numpy()
        if torch.is_tensor(text_features):
            text_features = text_features.cpu().numpy()

        return image_features + text_features

    def _multiplicative_fusion(self, image_features, text_features):
        if torch.is_tensor(image_features):
            image_features = image_features.cpu().numpy()
        if torch.is_tensor(text_features):
            text_features = text_features.cpu().numpy()

        return image_features * text_features

    def _cross_attention_fusion(self, image_features, text_features):
        min_dim = min(image_features.size(-1), text_features.size(-1))
        image_features = image_features[:, :min_dim]
        text_features = text_features[:, :min_dim]

        with torch.no_grad():
            fused_features = self.cross_attention_fusion(
                image_features, text_features
            )

        return fused_features.cpu().numpy()

    def multimodal_fusion(
        self,
        images: Union[Image.Image, List[Image.Image]],
        texts: Union[str, List[str]],
        method: str = "concat",
    ) -> np.ndarray:
        image_features = self.extract_image_features(images)
        text_features = self.extract_text_features(texts)

        min_dim = min(image_features.size(-1), text_features.size(-1))
        image_features = image_features[:, :min_dim]
        text_features = text_features[:, :min_dim]

        fusion_func = self.fusion_methods.get(method, self._concat_fusion)
        return fusion_func(image_features, text_features)

    def visualize_similarity(
        self,
        similarity_matrix: np.ndarray,
        images: List[Image.Image],
        texts: List[str],
        output_path: Optional[Union[str, Path]] = None,
    ):
        figure, axis = plt.subplots(figsize=(10, 8))
        image = axis.imshow(similarity_matrix, cmap="viridis", aspect="auto")

        axis.set_xticks(range(len(texts)))
        axis.set_yticks(range(len(images)))
        axis.set_xticklabels(texts, rotation=45, ha="right")
        axis.set_yticklabels(
            [f"Image {index + 1}" for index in range(len(images))]
        )

        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label("Similarity Score")

        for row in range(len(images)):
            for column in range(len(texts)):
                axis.text(
                    column,
                    row,
                    f"{similarity_matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                )

        axis.set_title("Image-Text Similarity Matrix")
        figure.tight_layout()

        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=150)
        else:
            plt.show()

        return figure

    def save_features(self, output_path: Union[str, Path], **arrays) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
