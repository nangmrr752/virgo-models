"""Whether a Virgo hearing model was trained on every language (finetune_stt.py --multilingual writes
virgo_hearing.json into it). Kept separate so training can ask without loading the speech server."""
import json
import os


def multilingual(model):
    try:
        path = os.path.join(model, "virgo_hearing.json")
        if not os.path.isfile(path):
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(model, "virgo_hearing.json")
        return bool(json.load(open(path, encoding="utf-8")).get("multilingual"))
    except Exception:
        return False
