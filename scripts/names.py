"""Virgo's model names, in one place: the standard Virgo-{Name}-{Version}-{Size or Type} names, and the old
names each one replaces (Hugging Face keeps a redirect from an old name after a rename).

    python scripts/names.py                 # what would be renamed on your Hugging Face account (nothing changes)
    python scripts/names.py rename --yes    # rename them (old links keep working)
    python scripts/names.py resolve Virgo-Bayon-1.0-TTS   # the name to use now: the new one, else an old one

Scripts call resolve() so they work before, during and after the rename.
"""
import sys

# standard name: [old names]
NAMES = {
    # chat
    "Virgo-Angkor-1.0-4B": ["Virgo-1.0-Angkor"],
    "Virgo-Angkor-1.0-12B": ["Virgo-1.0-Angkor-12B"],
    "Virgo-Bayon-1.0-27B": ["Virgo-1.0-Bayon"],
    "Virgo-Bayon-1.0-12B": ["Virgo-1.0-Bayon-12B"],
    "Virgo-Bayon-1.0-4B": ["Virgo-1.0-Bayon-4B"],
    # hearing (speech to text)
    "Virgo-Angkor-1.0-STT": ["Virgo-1.0-Angkor-Hearing"],
    # voices (text to speech)
    "Virgo-Angkor-1.0-TTS": ["Virgo-1.0-Angkor-Voice"],
    "Virgo-Bayon-1.0-TTS": ["Virgo-1.0-Bayon-Voice"],
    "Virgo-Bakong-2.0-TTS": ["Virgo-Bakong-2.0-Voice"],
}
DATASETS = {"Virgo-Data-1.0": ["Virgo-1.0-Angkor-Data"]}


def _exists(api, repo, repo_type="model"):
    try:
        return api.repo_exists(repo, repo_type=repo_type)
    except Exception:
        return False


def resolve(name, me=None, api=None, repo_type=None):
    """`me/name` if it exists, else the first old name that exists, else `me/name` (where new ones go)."""
    from huggingface_hub import HfApi

    api = api or HfApi()
    me = me or api.whoami()["name"]
    table = DATASETS if (repo_type == "dataset" or name in DATASETS) else NAMES
    kind = "dataset" if table is DATASETS else "model"
    for candidate in [name, *table.get(name, [])]:
        if _exists(api, f"{me}/{candidate}", kind):
            return f"{me}/{candidate}"
    return f"{me}/{name}"


def plan(api, me):
    moves = []
    for table, kind in ((NAMES, "model"), (DATASETS, "dataset")):
        for new, olds in table.items():
            if _exists(api, f"{me}/{new}", kind):
                continue
            old = next((o for o in olds if _exists(api, f"{me}/{o}", kind)), None)
            if old:
                moves.append((f"{me}/{old}", f"{me}/{new}", kind))
    return moves


def main():
    from huggingface_hub import HfApi

    api = HfApi()
    if len(sys.argv) > 2 and sys.argv[1] == "resolve":
        print(resolve(sys.argv[2], api=api))
        return
    me = api.whoami()["name"]
    moves = plan(api, me)
    if not moves:
        print("✅ Everything already has its standard name.")
        return
    for old, new, kind in moves:
        print(f"  {old}  →  {new}" + ("  (dataset)" if kind == "dataset" else ""))
    if sys.argv[1:3] != ["rename", "--yes"]:
        print("\nNothing changed. To rename: python scripts/names.py rename --yes  (old links keep redirecting)")
        return
    for old, new, kind in moves:
        api.move_repo(from_id=old, to_id=new, repo_type=kind)
        print("✅ Renamed", old, "→", new)


if __name__ == "__main__":
    main()
