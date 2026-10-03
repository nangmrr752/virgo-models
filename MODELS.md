# Virgo's models and their names

The platform is **Virgo**, so model names don't repeat it: **`{Name}-{Version}-{Size or Type}`**, kept in
Virgo's own Hugging Face organization: **`<org>/{Name}-{Version}-{Size or Type}`**, e.g. `virgoai/Bayon-1.0-27B`.
STT = speech to text (hearing), TTS = text to speech (voice).

| Model | Name | Older names (still work) | Trained with |
|---|---|---|---|
| Chat, Gemma 3 27B | **Bayon-1.0-27B** | Virgo-1.0-Bayon | `train_local.sh bayon 27b` |
| Chat, Gemma 3 12B | **Bayon-1.0-12B** | Virgo-1.0-Bayon-12B | `train_local.sh bayon 12b` |
| Chat, Gemma 3 4B | **Bayon-1.0-4B** | Virgo-1.0-Bayon-4B | `train_local.sh bayon 4b` |
| Chat, Gemma 3 12B | **Angkor-1.0-12B** | Virgo-1.0-Angkor-12B | `train_local.sh angkor 12b` |
| Chat, Gemma 3 4B | **Angkor-1.0-4B** | Virgo-1.0-Angkor | `train_local.sh angkor 4b` |
| Hearing (Khmer) | **Angkor-1.0-STT** | Virgo-1.0-Angkor-Hearing | `train_local.sh hearing` |
| Voice | **Angkor-1.0-TTS** | Virgo-1.0-Angkor-Voice | `train_local.sh voice` |
| Bayon's voice | **Bayon-1.0-TTS** | Virgo-1.0-Bayon-Voice | `train_local.sh voice bayon` |
| Live voice's voice | **Bakong-2.0-TTS** | Virgo-Bakong-2.0-Voice | `train_local.sh voice bakong` |
| Training data (dataset) | **Data-1.0** | Virgo-1.0-Angkor-Data | `scripts/distill_gemma.py` |

**Bakong-2.0** is also the live voice product's name on the website (hearing + Bayon/Angkor + voice).

## Moving your models to Virgo's organization

1. On Hugging Face: your avatar → **New Organization**, pick a free name (e.g. `virgoai`), keep repos private.
2. Add it to `.env`: `VIRGO_HF_ORG=virgoai`
3. Move the models (a dry run first):

```bash
python scripts/names.py                 # shows what would move (changes nothing)
python scripts/names.py rename --yes    # moves and renames them; Hugging Face redirects the old names
```

Everything here (training, the server, the notebooks) looks in Virgo's organization first, then on your
account, then under the older names, so it works before, during and after the move. The list of names
lives in `scripts/names.py`.
