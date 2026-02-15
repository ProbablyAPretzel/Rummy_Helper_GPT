from engine import Tile, recommend_turn_start, recommend_after_drawing_deck

COLOR_MAP = {"R": 0, "B": 1, "K": 2, "Y": 3}

def parse_tile(tok: str) -> Tile:
    tok = tok.strip().upper()
    if tok == "J":
        return Tile(-1, -1)
    c = tok[0]
    n = int(tok[1:])
    if c not in COLOR_MAP:
        raise ValueError(f"Bad color '{c}'. Use R/B/K/Y or J.")
    if not (1 <= n <= 13):
        raise ValueError(f"Bad number '{n}'. Must be 1..13.")
    return Tile(COLOR_MAP[c], n)

def parse_hand(line: str):
    toks = [t for t in line.strip().split() if t]
    tiles = [parse_tile(t) for t in toks]
    return tiles

def fmt_tile(tile: Tile) -> str:
    if tile.color == -1 and tile.number == -1:
        return "J"
    inv = {0:"R", 1:"B", 2:"K", 3:"Y"}
    return f"{inv[tile.color]}{tile.number}"

def main():
    print("Enter your 14-tile hand (e.g. 'R1 R2 R3 ... Y13'):")
    hand14 = parse_hand(input("> "))
    if len(hand14) != 14:
        raise ValueError(f"Expected 14 tiles, got {len(hand14)}")

    print("Enter top discard (e.g. 'R8' or 'J'):")
    top_discard = parse_tile(input("> "))

    # If you discarded a joker earlier in the round, set this True.
    # For now we default False; you can edit or add a prompt later.
    joker_discarded_earlier = False

    out = recommend_turn_start(hand14, top_discard, joker_discarded_earlier=joker_discarded_earlier)
    print("\nTurn start recommendation:")
    print(out)

    if out["draw"]["action"] == "DRAW_DECK":
        print("\nYou chose/you must draw from deck. Enter the drawn tile:")
        drawn = parse_tile(input("> "))
        action = recommend_after_drawing_deck(hand14, drawn, joker_discarded_earlier=joker_discarded_earlier)
        print("\nAfter drawing recommendation:")
        # Make it human-readable
        if action["action"] == "CLOSE":
            print(f"CLOSE. Free tile: {fmt_tile(action['free_tile'])}. Score: {action['score']}")
        else:
            print(f"DISCARD: {fmt_tile(action['discard_tile'])} (index {action['discard_index']})")
            print("Info:", action["info"])
    else:
        # If TAKE, recommend_turn_start includes after_take
        after_take = out["after_take"]
        print("\nAfter taking discard recommendation:")
        if after_take["action"] == "CLOSE":
            print(f"CLOSE. Free tile: {fmt_tile(after_take['free_tile'])}. Score: {after_take['score']}")
        else:
            print(f"DISCARD: {fmt_tile(after_take['discard_tile'])} (index {after_take['discard_index']})")
            print("Info:", after_take["info"])

if __name__ == "__main__":
    main()