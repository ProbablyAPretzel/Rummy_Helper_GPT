import timeit

setup = """
from engine import Tile, recommend_turn_start, recommend_after_drawing_deck

hand_14 = [
    Tile(0,1), Tile(0,2), Tile(0,3),
    Tile(1,5), Tile(2,5), Tile(3,5),
    Tile(0,7), Tile(0,8), Tile(0,9),
    Tile(1,10), Tile(2,10), Tile(3,10),
    Tile(0,12), Tile(0,13)
]
top_discard = Tile(1,4)
drawn = Tile(1,6)

def run_once():
    recommend_turn_start(hand_14, top_discard)
    recommend_after_drawing_deck(hand_14, drawn)

# warm-up
for _ in range(200):
    run_once()
"""

# repeat a few times, see stability
for k in range(5):
    result = timeit.timeit("run_once()", setup=setup, number=2000)
    print(f"Trial {k+1}: 2000 runs: {result:.6f}s, avg {(result/2000)*1000:.3f} ms/run")
