from engine import Tile, recommend_take_or_draw, recommend_action, recommend_turn_start, recommend_after_drawing_deck

R=0; B=1; K=2; Y=3
J = Tile(-1,-1)

def t(c,n): return Tile(c,n)

def assert_true(cond, msg):
    if not cond:
        raise AssertionError("FAIL: " + msg)

def assert_eq(a, b, msg):
    if a != b:
        raise AssertionError(f"FAIL: {msg} (expected {b}, got {a})")


def run_strategy_tests():
    # 1) "Useless discard shouldn't be taken"
    hand14 = [
        t(R,1), t(R,2), t(R,3),
        t(B,5), t(K,5), t(Y,5),
        t(R,7), t(R,8), t(R,9),
        t(B,10), t(K,10), t(Y,10),
        t(R,12), t(R,13),
    ]
    top_discard = t(B,4)
    d = recommend_take_or_draw(hand14, top_discard)

    # TEMP: stop here so we can read the debug output

    d = recommend_take_or_draw(hand14, top_discard)
    assert_eq(d["action"], "DRAW_DECK", "Should DRAW when discard doesn't meaningfully improve post-discard position")
    assert d["baseline_score"] >= d["take_score"] - 1e-9 or (d["take_score"] - d["baseline_score"]) < 5.0

    # 2) Discard joker cannot be taken (even if it would create a strong close)
    base14 = [
        t(R, 1), t(R, 2), t(R, 3), t(R, 4),
        t(B, 4), t(B, 5), t(B, 6), t(B, 7),
        t(R, 3), t(K, 3), t(Y, 3),
        t(R, 6), t(K, 6), t(Y, 6),
    ]

    top_discard = J  # joker tile
    d = recommend_take_or_draw(base14, top_discard, joker_discarded_earlier=False)
    assert_eq(d["action"], "DRAW_DECK", "Should DRAW_DECK because you can't take a discarded joker")

    # 3) After drawing a tile, if a high-value close exists, recommend_action should CLOSE
    # Make a known 500 minors close with free joker.
    hand15 = base14 + [J]  # joker free -> x2 on 500? (but base is 500; free joker doubles to 1000)
    a = recommend_action(hand15, joker_discarded_earlier=False)
    from engine import best_close_plan, evaluate_close
    print("DEBUG hand15:", hand15)
    print("DEBUG best_close_plan:", best_close_plan(hand15))
    print("DEBUG evaluate_close:", evaluate_close(hand15))
    print("DEBUG recommend_action:", a)

    assert_eq(a["action"], "CLOSE", "Should CLOSE when a big close is available")

    # 4) If no close is available, recommend_action should DISCARD
    # Make a messy 15 where partition isn't possible.
    hand15 = [
        t(R, 1), t(B, 2), t(K, 3),
        t(Y, 4), t(R, 6), t(B, 7),
        t(K, 9), t(Y, 10), t(R, 12),
        t(B, 13), t(K, 1), t(Y, 2),
        t(R, 5), t(B, 8), t(K, 11),
    ]
    a = recommend_action(hand15)
    assert_eq(a["action"], "DISCARD", "Should DISCARD when no close exists")
    assert_true("discard_index" in a and "discard_tile" in a, "DISCARD should include discard_index and discard_tile")


    # 5) Turn-start wrapper: structure + required fields
    hand14 = [
        t(R, 1), t(R, 2), t(R, 3),
        t(B, 5), t(K, 5), t(Y, 5),
        t(R, 7), t(R, 8), t(R, 9),
        t(B, 10), t(K, 10), t(Y, 10),
        t(R, 12), t(R, 13),
    ]
    out = recommend_turn_start(hand14, t(B, 4))

    assert_eq(out.get("stage"), "TURN_START", "turn_start should set stage=TURN_START")
    assert_true(isinstance(out.get("draw"), dict), "turn_start should include a draw decision dict")
    assert_true("action" in out["draw"], "draw decision should include action")
    assert_true(isinstance(out.get("note"), str) and len(out["note"]) > 0, "turn_start should include a note string")

    act = out["draw"]["action"]
    assert_true(act in ("TAKE_DISCARD", "DRAW_DECK"), "turn_start action must be TAKE_DISCARD or DRAW_DECK")

    # If it takes, it must include the chosen discard after taking
    if act == "TAKE_DISCARD":
        assert_true("discard_index_after_take" in out["draw"], "TAKE should include discard_index_after_take")
        assert_true("discard_tile_after_take" in out["draw"], "TAKE should include discard_tile_after_take")
        assert_true("take_score" in out["draw"] and "baseline_score" in out["draw"],
                    "TAKE should include scores for transparency")
    else:
        # If it draws, it should still include scores (if your implementation provides them)
        assert_true("take_score" in out["draw"] and "baseline_score" in out["draw"],
                    "DRAW should include scores for transparency")

    # 6) Discard: prefer throwing an isolated tile rather than breaking a strong near-run
    # Hand has 6R-7R (strong), plus some melds; one tile is isolated (13Y).
    # After drawing a random tile, bot should avoid discarding 6R or 7R if there’s a clear junk tile.
    hand15 = [
        t(R,6), t(R,7),          # near-run we want to keep
        t(B,5), t(K,5), t(Y,5),  # set
        t(B,10), t(K,10), t(Y,10),  # set
        t(R,1), t(R,2), t(R,3),  # run
        t(B,8), t(K,9),          # semi-junk
        t(Y,13),                 # very isolated junk
        t(B,1),                  # filler
    ]
    a = recommend_action(hand15)
    assert_eq(a["action"], "DISCARD", "Should DISCARD here (no guaranteed close assumed)")
    discarded = a["discard_tile"]
    assert_true(discarded not in [t(R,6), t(R,7)], "Should not break 6R-7R near-run if junk exists")

    # 7) Take/Draw: prefer taking discard that extends a near-run
    # You have 6R-7R and the discard is 8R, which strongly improves near-run structure.
    hand14 = [
        t(R,6), t(R,7),
        t(B,5), t(K,5), t(Y,5),
        t(B,10), t(K,10), t(Y,10),
        t(R,1), t(R,2), t(R,3),
        t(B,8), t(K,9),
        t(Y,13),
    ]
    top_discard = t(R,8)
    d = recommend_take_or_draw(hand14, top_discard)
    assert_eq(d["action"], "TAKE_DISCARD", "Should TAKE discard that extends a strong near-run")

    # 8) Regression: avoid overvaluing 12-13 adjacency; don't discard set-seed like K6 here
    hand14 = [
        t(Y, 13), t(R, 13), t(B, 13),
        t(B, 1), t(K, 1), t(K, 3), t(B, 3),
        t(Y, 2), t(Y, 4), t(R, 8),
        t(B, 5), t(R, 5), t(R, 6), t(K, 6),
    ]
    drawn = t(Y, 12)
    a = recommend_after_drawing_deck(hand14, drawn)

    assert_eq(a["action"], "DISCARD", "Should discard after drawing when no close")
    assert_true(a["discard_tile"] != t(K, 6), "Should not discard K6 here")

    print("All strategy tests passed.")

if __name__ == "__main__":
    run_strategy_tests()
