import numpy as np
import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert (
            self.head_dim * num_heads == feature_dim
        ), "特征维度不能被头数整除"

        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, image_features, text_features):
        batch_size = image_features.size(0)

        query = self.q_proj(image_features).view(
            batch_size, self.num_heads, self.head_dim
        )
        key = self.k_proj(text_features).view(
            batch_size, self.num_heads, self.head_dim
        )
        value = self.v_proj(text_features).view(
            batch_size, self.num_heads, self.head_dim
        )

        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(
            self.head_dim
        )
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attended_features = torch.matmul(attention_weights, value)
        attended_features = attended_features.contiguous().view(
            batch_size, self.feature_dim
        )

        output = self.out_proj(attended_features)
        output = self.layer_norm(output + image_features)

        query2 = self.q_proj(text_features).view(
            batch_size, self.num_heads, self.head_dim
        )
        key2 = self.k_proj(image_features).view(
            batch_size, self.num_heads, self.head_dim
        )
        value2 = self.v_proj(image_features).view(
            batch_size, self.num_heads, self.head_dim
        )

        scores2 = torch.matmul(query2, key2.transpose(-2, -1)) / np.sqrt(
            self.head_dim
        )
        attention_weights2 = torch.softmax(scores2, dim=-1)
        attention_weights2 = self.dropout(attention_weights2)

        attended_features2 = torch.matmul(attention_weights2, value2)
        attended_features2 = attended_features2.contiguous().view(
            batch_size, self.feature_dim
        )

        output2 = self.out_proj(attended_features2)
        output2 = self.layer_norm(output2 + text_features)

        return 0.7 * output + 0.3 * output2
