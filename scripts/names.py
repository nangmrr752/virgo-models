"""Virgo's model names, in one place. The platform is Virgo, so models are named {Name}-{Version}-{Size or Type}
and live in Virgo's own Hugging Face organization: <VIRGO_HF_ORG>/Bayon-1.0-27B (e.g. virgoai/Bayon-1.0-27B).
Without VIRGO_HF_ORG they stay on your own account (<you>/Bayon-1.0-27B).

    python scripts/names.py                       # what would move or be renamed (nothing changes)
    python scripts/names.py rename --yes          # move/rename them (Hugging Face redirects old names)
    python scripts/names.py resolve Bayon-1.0-TTS # the repo to use now: the standard one, else an old one

Every script calls resolve()/candidates(), so they work before, during and after the move.
"""
import os
import sys

# standard name: [older names, newest first]
NAMES = {
    # chat
    "Bayon-1.0-27B": ["Virgo-Bayon-1.0-27B", "Virgo-1.0-Bayon"],
    "Bayon-1.0-12B": ["Virgo-Bayon-1.0-12B", "Virgo-1.0-Bayon-12B"],
    "Bayon-1.0-4B": ["Virgo-Bayon-1.0-4B", "Virgo-1.0-Bayon-4B"],
    "Angkor-1.0-12B": ["Virgo-Angkor-1.0-12B", "Virgo-1.0-Angkor-12B"],
    "Angkor-1.0-4B": ["Virgo-Angkor-1.0-4B", "Virgo-1.0-Angkor"],
    # hearing (speech to text)
    "Angkor-1.0-STT": ["Virgo-Angkor-1.0-STT", "Virgo-1.0-Angkor-Hearing"],
    "Angkor-1.1-STT": [],
    # voices (text to speech)
    "Angkor-1.0-TTS": ["Virgo-Angkor-1.0-TTS", "Virgo-1.0-Angkor-Voice"],
    "Bayon-1.0-TTS": ["Virgo-Bayon-1.0-TTS", "Virgo-1.0-Bayon-Voice"],
    "Bakong-2.0-TTS": ["Virgo-Bakong-2.0-TTS", "Virgo-Bakong-2.0-Voice"],
}
DATASETS = {"Data-1.0": ["Virgo-Data-1.0", "Virgo-1.0-Angkor-Data"]}


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def owners(api=None):
    """Where Virgo's repos are looked for: Virgo's organization first (VIRGO_HF_ORG), then your account."""
    api = api or _api()
    me = api.whoami()["name"]
    org = os.environ.get("VIRGO_HF_ORG", "").strip()
    return ([org] if org and org != me else []) + [me]


def home(api=None):
    """Where new repos go: Virgo's organization, else your account."""
    return owners(api)[0]


def _table(name):
    return (DATASETS, "dataset") if name in DATASETS else (NAMES, "model")


def candidates(name, api=None):
    """Every repo id this model may be at, best first: standard name in the org, then on your account,
    then the older names."""
    table, _ = _table(name)
    names = [name, *table.get(name, [])]
    return [f"{owner}/{n}" for n in names for owner in owners(api)]


def _exists(api, repo, kind):
    try:
        return api.repo_exists(repo, repo_type=kind)
    except Exception:
        return False


def resolve(name, api=None, need_file=None):
    """The first repo in candidates() that exists (and has `need_file`, if given), else where it goes new."""
    api = api or _api()
    _, kind = _table(name)
    for repo in candidates(name, api):
        if not _exists(api, repo, kind):
            continue
        if need_file:
            try:
                if not api.file_exists(repo, need_file, repo_type=kind):
                    continue
            except Exception:
                continue
        return repo
    return f"{home(api)}/{name}"


def plan(api):
    moves = []
    for table, kind in ((NAMES, "model"), (DATASETS, "dataset")):
        for name in table:
            target = f"{home(api)}/{name}"
            if _exists(api, target, kind):
                continue
            found = next((r for r in candidates(name, api)[1:] if _exists(api, r, kind)), None)
            if found:
                moves.append((found, target, kind))
    return moves


def main():
    api = _api()
    if len(sys.argv) > 2 and sys.argv[1] == "resolve":
        print(resolve(sys.argv[2], api))
        return
    org = os.environ.get("VIRGO_HF_ORG", "").strip()
    print("Virgo's models go to:", home(api), "(set VIRGO_HF_ORG to Virgo's organization)" if not org else "")
    moves = plan(api)
    if not moves:
        print("✅ Everything already has its standard name and place.")
        return
    for old, new, kind in moves:
        print(f"  {old}  →  {new}" + ("  (dataset)" if kind == "dataset" else ""))
    if sys.argv[1:3] != ["rename", "--yes"]:
        print("\nNothing changed. To move them: python scripts/names.py rename --yes  (old links keep redirecting)")
        return
    for old, new, kind in moves:
        api.move_repo(from_id=old, to_id=new, repo_type=kind)
        print("✅", old, "→", new)


if __name__ == "__main__":
    main()
