# Virgo's models and their names

Standard names: **`Virgo-{Name}-{Version}-{Size or Type}`**, like other open models
(`Llama-3.1-8B-Instruct`, `whisper-large-v3`). STT = speech to text (hearing), TTS = text to speech (voice).

| Model | Standard name | Old name (still works) | Trained with |
|---|---|---|---|
| Chat, Gemma 3 27B | **Virgo-Bayon-1.0-27B** | Virgo-1.0-Bayon | `train_local.sh bayon 27b` |
| Chat, Gemma 3 12B | **Virgo-Bayon-1.0-12B** | Virgo-1.0-Bayon-12B | `train_local.sh bayon 12b` |
| Chat, Gemma 3 4B | **Virgo-Bayon-1.0-4B** | Virgo-1.0-Bayon-4B | `train_local.sh bayon 4b` |
| Chat, Gemma 3 12B | **Virgo-Angkor-1.0-12B** | Virgo-1.0-Angkor-12B | `train_local.sh angkor 12b` |
| Chat, Gemma 3 4B | **Virgo-Angkor-1.0-4B** | Virgo-1.0-Angkor | `train_local.sh angkor 4b` |
| Hearing (Khmer) | **Virgo-Angkor-1.0-STT** | Virgo-1.0-Angkor-Hearing | `train_local.sh hearing` |
| Voice | **Virgo-Angkor-1.0-TTS** | Virgo-1.0-Angkor-Voice | `train_local.sh voice` |
| Bayon's voice | **Virgo-Bayon-1.0-TTS** | Virgo-1.0-Bayon-Voice | `train_local.sh voice bayon` |
| Live voice's voice | **Virgo-Bakong-2.0-TTS** | Virgo-Bakong-2.0-Voice | `train_local.sh voice bakong` |
| Training data | **Virgo-Data-1.0** (dataset) | Virgo-1.0-Angkor-Data | `scripts/distill_gemma.py` |

**Virgo-Bakong-2.0** is the live voice product (hearing + Bayon/Angkor + voice), named on the website.

## Renaming your Hugging Face repos

```bash
python scripts/names.py                 # shows what would be renamed (changes nothing)
python scripts/names.py rename --yes    # renames them; Hugging Face redirects the old names
```

Everything here (training, the server, the notebooks) tries the standard name first and then the old one,
so it works before, during and after the rename. New training runs upload to the old name until you rename.
The list of names lives in `scripts/names.py`.
