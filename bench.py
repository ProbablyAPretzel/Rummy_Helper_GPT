from engine import Tile, debug_best_discard

R=0; B=1; K=2; Y=3
J = Tile(-1,-1)

hand15 = [
    Tile(R,3), Tile(B,3), J, Tile(R,6), Tile(R,7), Tile(R,9),
    Tile(Y,7), Tile(Y,8), Tile(Y,10), Tile(Y,11),
    Tile(K,11), Tile(B,11), Tile(B,12), Tile(R,12),
    Tile(B,5),   # drawn tile
]

debug_best_discard(hand15, top_k=12)
