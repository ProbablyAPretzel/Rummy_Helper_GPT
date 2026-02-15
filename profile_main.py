import cProfile, pstats
from engine import Tile, recommend_turn_start, recommend_after_drawing_deck

hand_14 = [
    Tile(0,1), Tile(0,2), Tile(0,3),
    Tile(1,5), Tile(2,5), Tile(3,5),
    Tile(0,7), Tile(0,8), Tile(0,9),
    Tile(1,10), Tile(2,10), Tile(3,10),
    Tile(0,12), Tile(0,13)
]
top_discard = Tile(1,4)

def run_once():
    recommend_turn_start(hand_14, top_discard)
    recommend_after_drawing_deck(hand_14, Tile(1,6))

if __name__ == "__main__":
    prof = cProfile.Profile()
    prof.enable()
    for _ in range(50):
        run_once()
    prof.disable()

    stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
    stats.print_stats(30)
