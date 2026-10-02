"""Whether Virgo-1.0-Bayon fits next to Virgo-1.0-Angkor on this machine's GPUs (the server notebooks and
scripts/serve_rented_gpu.py ask before downloading it). Rough 4-bit sizes; hearing + voice need ~7 GB."""
import json

SIZES_GB = {27: 17.0, 12: 7.5, 4: 3.5}
ANGKOR_GB = 7.5
SPEECH_GB = 7.0  # Whisper hearing + VoxCPM2/Kokoro voice


def model_size(repo):
    """12, 27 or 4 (billions of parameters), from the model's adapter_config.json (base model name)."""
    from huggingface_hub import hf_hub_download

    try:
        base = json.load(open(hf_hub_download(repo, "adapter_config.json"))).get("base_model_name_or_path", "")
    except Exception:  # a whole model (TPU-trained): read its size from the name
        base = repo
    name = base.lower()
    return 27 if "27b" in name else 4 if "4b" in name else 12


def bayon_fits(repo):
    import torch

    gpus = [torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(torch.cuda.device_count())]
    if not gpus:
        return False
    need = SIZES_GB[model_size(repo)]
    if len(gpus) > 1:  # Bayon and the voice on the last GPU, Angkor and hearing on the first
        return gpus[-1] >= need + 5.5
    return gpus[0] >= ANGKOR_GB + need + SPEECH_GB
