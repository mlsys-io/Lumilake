# Per-model parameter counts (billions). Used to estimate memory usage
# and to choose an appropriate batch size for training or inference.
MODEL_SIZE = {
    "stabilityai/stable-diffusion-xl-base-1.0": 3.5,
    "stabilityai/stable-diffusion-3.5-medium": 2.5,
    "stabilityai/stable-diffusion-3.5-large": 8.1,
    "meta-llama/Llama-3.2-3B-Instruct": 3.0,
    "Tongyi-MAI/Z-Image-Turbo": 6.0,
    "BAAI/bge-small-en-v1.5": 0.033,
    "BAAI/bge-reranker-v2-m3": 0.568,
    "cross-encoder/ms-marco-MiniLM-L6-v2": 0.0227,
}
