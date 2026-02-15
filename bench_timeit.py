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

# warm-up
for _ in range(50):
    run_once()

times = []
N = 1000
for _ in range(N):
    t0 = perf_counter()
    run_once()
    t1 = perf_counter()
    times.append((t1 - t0) * 1000)  # ms

times.sort()
def pct(p):
    return times[int(p/100 * (N-1))]

print(f"min: {times[0]:.3f} ms")
print(f"p50: {pct(50):.3f} ms")
print(f"p90: {pct(90):.3f} ms")
print(f"p99: {pct(99):.3f} ms")
print(f"max: {times[-1]:.3f} ms")
