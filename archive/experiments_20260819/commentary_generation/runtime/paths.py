"""Read-only asset paths for the historical SoccerMaster commentary experiment."""

import os
from pathlib import Path


ASSET_ROOT_ENV = "SOCCERMASTER_COMMENTARY_ASSET_ROOT"
_asset_root_text = os.environ.get(ASSET_ROOT_ENV)

if _asset_root_text:
    ASSET_ROOT = Path(_asset_root_text)
    if not ASSET_ROOT.is_absolute():
        raise ValueError(f"{ASSET_ROOT_ENV} must be an absolute path")
    ASSET_LAYOUT = "nas_bundle_v1"
    LLAMA_ROOT = ASSET_ROOT / "models/Meta-Llama-3-8B-Instruct"
    BERT_ROOT = ASSET_ROOT / "models/bert-base-uncased"
    SIGLIP2_ROOT = ASSET_ROOT / "models/siglip2-large-patch16-512"
    VISUAL_BACKBONE = (
        ASSET_ROOT / "checkpoints/visual_backbone_epoch_19/backbone.pt"
    )
    GENERATION_CHECKPOINT = (
        ASSET_ROOT / "checkpoints/commentary_epoch_11/model_save_11.pth"
    )
    WORD_WORLD = ASSET_ROOT / "metadata/words_world/match_time.pkl"
    TRAIN_ANNOTATIONS = (
        ASSET_ROOT / "metadata/MatchTime/classification_train.json"
    )
    TEST_ANNOTATIONS = (
        ASSET_ROOT / "metadata/MatchTime/classification_test.json"
    )
else:
    ASSET_ROOT = None
    ASSET_LAYOUT = "historical_remote_home"
    LLAMA_ROOT = Path("/remote-home/share/huggingface/Meta-Llama-3-8B-Instruct")
    BERT_ROOT = Path("/remote-home/share/huggingface/bert-base-uncased")
    SIGLIP2_ROOT = Path(
        "/remote-home/haolinyang/sports/Soccer-Backbone/"
        "pretrained_models/google/siglip2-large-patch16-512"
    )
    VISUAL_BACKBONE = Path(
        "/remote-home/haolinyang/sports/Soccer-Backbone/outputs/"
        "pretrain_large_512_multitask_aug_consine_part_temporal_early_"
        "freeze_text_lr_5e-5_cap_cls_weight_1_extra_7000/epoch_19/backbone.pt"
    )
    GENERATION_CHECKPOINT = Path(
        "/remote-home/haolinyang/sports/dirty_code/UniSoccer/output/"
        "large_512_multitask_w_1_epoch_19_train_matchtime_eval_matchtime_"
        "half_lr_bf16/model_save_11.pth"
    )
    WORD_WORLD = Path(
        "/remote-home/haolinyang/sports/UniSoccer/words_world/match_time.pkl"
    )
    TRAIN_ANNOTATIONS = Path(
        "/remote-home/haolinyang/sports/UniSoccer/train_data/"
        "video_clip_json/MatchTime/classification_train.json"
    )
    TEST_ANNOTATIONS = Path(
        "/remote-home/haolinyang/sports/UniSoccer/train_data/"
        "video_clip_json/MatchTime/classification_test.json"
    )

ANONYMIZATION_TOKENS = (
    "[PLAYER]",
    "[TEAM]",
    "[COACH]",
    "[REFEREE]",
    "([TEAM])",
)
TOKENIZER_INITIAL_LENGTH = 128256
TOKENIZER_FINAL_LENGTH = 128261
LEGACY_PADDING_TOKEN_ID = 128001
