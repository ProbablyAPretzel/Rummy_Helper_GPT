from time import perf_counter
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

# Warm-up (fills DP caches, meld caches, etc.)
for _ in range(2000):
    run_once()

N = 20000
t0 = perf_counter()
for _ in range(N):
    run_once()
t1 = perf_counter()

avg_ms = (t1 - t0) / N * 1000
print(f"Average: {avg_ms:.3f} ms/run over {N} runs")
