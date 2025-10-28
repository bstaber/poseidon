"""Simple script to load a pretrained ScOT model and print the number of parameters."""

from poseidon.scOT.model import ScOT, ScOTConfig

MODEL_MAP = {
    "T": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [4, 4, 4, 4],
        "embed_dim": 48,
    },
    "S": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 48,
    },
    "B": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 96,
    },
    "L": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 192,
    },
}

model_size = "L"
model_config = ScOTConfig(
    image_size=128,
    patch_size=MODEL_MAP[model_size]["patch_size"],
    num_channels=1,
    num_out_channels=1,
    embed_dim=MODEL_MAP[model_size]["embed_dim"],
    depths=MODEL_MAP[model_size]["depths"],
    num_heads=MODEL_MAP[model_size]["num_heads"],
    skip_connections=MODEL_MAP[model_size]["skip_connections"],
    window_size=MODEL_MAP[model_size]["window_size"],
    mlp_ratio=MODEL_MAP[model_size]["mlp_ratio"],
    qkv_bias=True,
    hidden_dropout_prob=0.0,  # default
    attention_probs_dropout_prob=0.0,  # default
    drop_path_rate=0.0,
    hidden_act="gelu",
    use_absolute_embeddings=False,
    initializer_range=0.02,
    layer_norm_eps=1e-5,
    p=1,
    channel_slice_list_normalized_loss=None,
    residual_model="convnext",
    use_conditioning=False,
    learn_residual=False,
)

model = ScOT.from_pretrained(
    f"camlab-ethz/Poseidon-{model_size}",
    # config=model_config,
    # ignore_mismatched_sizes=True,
)

num_params = 0
for p in model.parameters():
    num_params += p.numel()
print(f"Number of parameters (millions) in the model: {num_params / 1_000_000:.2f}M")
