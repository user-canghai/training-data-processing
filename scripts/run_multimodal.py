import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多模态数据处理")
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)

    from training_data_processing.multimodal import MultimodalProcessor

    processor = MultimodalProcessor(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
    )

    random_generator = np.random.default_rng(args.seed)
    sample_images = []
    for _ in range(3):
        image_array = random_generator.integers(
            0, 255, size=(224, 224, 3), dtype=np.uint8
        )
        sample_images.append(Image.fromarray(image_array))

    sample_texts = [
        "A beautiful landscape",
        "A cute cat",
        "A modern city",
    ]

    print("=== 模态对齐 ===")
    similarity_matrix = processor.align_modalities(sample_images, sample_texts)
    print("相似度矩阵:")
    print(similarity_matrix)

    figure_path = args.output_dir / "figures/similarity_matrix.png"
    processor.visualize_similarity(
        similarity_matrix, sample_images, sample_texts, output_path=figure_path
    )
    print(f"相似度矩阵图: {figure_path}")

    print("\n=== 模态融合 ===")
    fusion_results = {}
    for method in [
        "concat",
        "additive",
        "multiplicative",
        "cross_attention",
    ]:
        try:
            fused_features = processor.multimodal_fusion(
                sample_images[:1],
                sample_texts[:1],
                method=method,
            )
            fusion_results[method] = fused_features
            print(f"{method} 融合后特征维度: {fused_features.shape}")
        except Exception as error:
            print(f"{method} 融合出错: {error}")

    print("\n=== 特征相似度计算 ===")
    text_features = processor.extract_text_features(["A beautiful landscape"])
    similar_text_features = processor.extract_text_features(
        ["A beautiful landscape with mountains and a lake"]
    )
    random_text_features = processor.extract_text_features(
        ["A completely different topic"]
    )

    similar_score = processor.compute_similarity(
        text_features, similar_text_features
    )[0][0]
    random_score = processor.compute_similarity(
        text_features, random_text_features
    )[0][0]
    print(f"相似文本相似度: {similar_score:.3f}")
    print(f"不同文本相似度: {random_score:.3f}")

    print("\n=== 交叉注意力融合详细分析 ===")
    image_feat = processor.extract_image_features(sample_images[:1])
    text_feat = processor.extract_text_features(sample_texts[:1])

    try:
        concat_result = processor.multimodal_fusion(
            sample_images[:1], sample_texts[:1], method="concat"
        )
        cross_attention_result = processor.multimodal_fusion(
            sample_images[:1], sample_texts[:1], method="cross_attention"
        )

        print(f"拼接融合结果维度: {concat_result.shape}")
        print(f"交叉注意力融合结果维度: {cross_attention_result.shape}")

        if concat_result.ndim > 1 and concat_result.shape[1] >= 512:
            concat_image_part = concat_result[:, :512]
            concat_text_part = concat_result[:, 512:]

            image_similarity = cosine_similarity(
                image_feat.cpu().numpy(), concat_image_part
            )[0][0]
            text_similarity = cosine_similarity(
                text_feat.cpu().numpy(), concat_text_part
            )[0][0]

            print(
                "拼接融合中图像特征与原始图像特征相似度: "
                f"{image_similarity:.3f}"
            )
            print(
                "拼接融合中文本特征与原始文本特征相似度: "
                f"{text_similarity:.3f}"
            )
    except Exception as error:
        print(f"融合测试出错: {error}")

    feature_path = args.output_dir / "features/multimodal_fusion.npz"
    processor.save_features(
        feature_path,
        image_features=image_feat.cpu().numpy(),
        text_features=text_feat.cpu().numpy(),
        similarity_matrix=similarity_matrix,
        **fusion_results,
    )
    print(f"融合特征: {feature_path}")


if __name__ == "__main__":
    main()
