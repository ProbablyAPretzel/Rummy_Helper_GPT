import json
import time
from pathlib import Path

import pyautogui

CONFIG_PATH = Path("screen_config.json")

def main():
    print("You have 3 seconds to switch to the game window...")
    time.sleep(3)

    img = pyautogui.screenshot()
    img_path = Path("latest_screenshot.png")
    img.save(img_path)
    print(f"Saved screenshot to {img_path.resolve()}")

    print("\nNow open latest_screenshot.png and note pixel coords.")
    print("We’ll store two rectangles:")
    print("  1) hand_row: bounding box around the whole 14-tile row on your board")
    print("  2) top_discard: bounding box around the discard tile")

    def ask_rect(name):
        print(f"\nEnter rectangle for {name} as: left top right bottom")
        s = input("> ").strip().split()
        if len(s) != 4:
            raise ValueError("Need 4 integers: left top right bottom")
        l, t, r, b = map(int, s)
        if r <= l or b <= t:
            raise ValueError("Invalid rectangle")
        return {"left": l, "top": t, "right": r, "bottom": b}

    cfg = {
        "hand_row": ask_rect("hand_row"),
        "top_discard": ask_rect("top_discard"),
    }

    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    print(f"\nWrote {CONFIG_PATH.resolve()}")
    print("Next step: we’ll split hand_row into 14 equal slots and classify each slot.")

if __name__ == "__main__":
    main()
