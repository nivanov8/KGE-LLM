from enum import Enum

class EmbeddingType(Enum):
    CLS_TOKEN_EMBEDDING = "cls_token_embedding"
    MEAN_POOLING_EMBEDDING = "mean_pooling_embedding"
