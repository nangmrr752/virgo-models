"""Talk to Virgo 1.0 in the terminal (base model + the trained adapter).

    python chat/chat.py
    python chat/chat.py --adapter chat/out/virgo-1.0-chat-lora
"""
import argparse
import os

from virgo_chat import VirgoChat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="google/gemma-3-4b-it")
    p.add_argument("--adapter", default="chat/out/virgo-1.0-chat-lora")
    args = p.parse_args()
    bot = VirgoChat(args.base, args.adapter if os.path.isdir(args.adapter) else None)
    history = []
    print("Virgo 1.0 — type your message (Ctrl+C to quit).")
    while True:
        text = input("\nYou: ").strip()
        if not text:
            continue
        history.append({"role": "user", "content": text})
        reply = bot.reply(history)
        history.append({"role": "assistant", "content": reply})
        print(f"Virgo: {reply}")


if __name__ == "__main__":
    main()
