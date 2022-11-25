from typing import Dict, Optional

from lightning.app.storage import Drive


def get_kwargs(
    pretrained_model_name_or_path: str,
    drive: Optional[Drive] = None,
) -> Dict[str, str]:
    kwargs = {
        "revision": "fp16",
        "use_auth_token": "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",
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
    }

    if drive and drive.list() == ["model.pt"]:
        drive.get("model.pt", overwrite=True)
        kwargs = {"pretrained_model_name_or_path": "./model.pt"}

    return kwargs
