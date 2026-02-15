from engine import Tile, recommend_turn_start, recommend_after_drawing_deck

hand_14 = [
    Tile(0,1), Tile(0,2), Tile(0,3),
    Tile(1,5), Tile(2,5), Tile(3,5),
    Tile(0,7), Tile(0,8), Tile(0,9),
    Tile(1,10), Tile(2,10), Tile(3,10),
    Tile(0,12), Tile(0,13)
]
top_discard = Tile(1,4)


print(recommend_turn_start(hand_14, top_discard))

# Example: if you drew from deck and got Blue 6:
print(recommend_after_drawing_deck(hand_14, Tile(1,6)))

from time import perf_counter

t0 = perf_counter()
print(recommend_turn_start(hand_14, top_discard))
print(recommend_after_drawing_deck(hand_14, Tile(1,6)))
t1 = perf_counter()
print("Decision time (ms):", (t1 - t0) * 1000)
