# tests.py
from engine import (
    Tile,
    is_valid_run,
    is_valid_set,
    best_partition_jokers_used,
    evaluate_close,
)

R = 0  # red
B = 1  # blue
K = 2  # black
Y = 3  # yellow
J = Tile(-1, -1)

def t(c, n):
    return Tile(c, n)

def j():
    return Tile(-1, -1)

def assert_true(cond, msg):
    if not cond:
        raise AssertionError("FAIL: " + msg)

def assert_false(cond, msg):
    if cond:
        raise AssertionError("FAIL: " + msg)

def assert_eq(a, b, msg):
    if a != b:
        raise AssertionError(f"FAIL: {msg} (expected {b}, got {a})")

def test_wrap_rule():
    # Allowed: wrap is terminal only, e.g. 11-12-13-1
    ok, _ = is_valid_run([t(R, 11), t(R, 12), t(R, 13), t(R, 1)])
    assert_true(ok, "11-12-13-1 should be valid (wrap terminal)")

    # Not allowed: wrap then continue
    ok, _ = is_valid_run([t(R, 12), t(R, 13), t(R, 1), t(R, 2)])
    assert_false(ok, "12-13-1-2 should NOT be valid (cannot continue after wrap)")

    ok, _ = is_valid_run([t(R, 13), t(R, 1), t(R, 2)])
    assert_false(ok, "13-1-2 should NOT be valid (wrap can't continue)")

    # Normal runs
    ok, _ = is_valid_run([t(R, 1), t(R, 2), t(R, 3)])
    assert_true(ok, "1-2-3 should be valid")

def test_4_tile_melds():
    # 4-tile run
    ok, _ = is_valid_run([t(B, 4), t(B, 5), t(B, 6), t(B, 7)])
    assert_true(ok, "4-5-6-7 same color should be a valid 4-run")

    # 4-tile set (all same number, unique colors)
    ok, _ = is_valid_set([t(R, 9), t(B, 9), t(K, 9), t(Y, 9)])
    assert_true(ok, "9 set across 4 colors should be a valid 4-set")

def test_full_partition_sanity():
    # This 14-tile set should NOT fully partition (your earlier example after wrap fix)
    tiles_14 = [
        t(R,1), t(R,2), t(R,3),
        t(B,5), t(K,5), t(Y,5),
        t(R,7), t(R,8), t(R,9),
        t(B,10), t(K,10), t(Y,10),
        t(R,12), t(R,13),
    ]
    res = best_partition_jokers_used(tiles_14)
    assert_true(res is None, "This 14-tile set must NOT fully partition")

def test_scoring_normal_and_joker_multiplier():
    # 14 melded tiles + 1 free tile (Y13) => normal close 250
    hand15 = [
        t(R, 1), t(R, 2), t(R, 3),
        t(B, 4), t(B, 5), t(B, 6), t(B, 7),
        t(R, 9), t(B, 9), t(K, 9),
        t(R, 12), t(B, 12), t(K, 12), t(Y, 12),
        t(Y, 13),   # free non-joker
    ]

    score = evaluate_close(hand15, joker_discarded_earlier=False)
    assert_eq(score, 250, "Normal close should score 250")

    # Same hand but free is joker => x2 => 500
    hand15_jfree = hand15[:-1] + [J]
    score = evaluate_close(hand15_jfree, joker_discarded_earlier=False)

    from engine import score_close

    joker_idxs = [i for i, tt in enumerate(hand15_jfree) if tt.is_joker()]
    print("Joker indices:", joker_idxs)
    for i in joker_idxs:
        s = score_close(hand15_jfree, i, joker_discarded_earlier=False)
        print("score_close with joker free_index", i, "->", s)

    assert_eq(score, 500, "Normal close with free joker should be 250x2 = 500")

    # Free joker + earlier discarded joker => x4 => 1000
    score = evaluate_close(hand15_jfree, joker_discarded_earlier=True)
    assert_eq(score, 1000, "Normal close with free joker + earlier discard should be 250x4 = 1000")

def test_scoring_big_minors_500():
    # Make a Minors big close: all non-free tiles are 1..7 (jokers allowed, but none used here).
    # 14 melded tiles: Run(4) + Run(4) + Set(3) + Set(3) = 14
    # Runs within 1..7:
    # 1-2-3-4 red (4)
    # 4-5-6-7 blue (4)
    # Sets:
    # 3 (red/black/yellow) (3)
    # 6 (red/black/yellow) (3)
    # Free: 13 yellow (free tile can be anything)
    hand15 = [
        t(R,1), t(R,2), t(R,3), t(R,4),
        t(B,4), t(B,5), t(B,6), t(B,7),
        t(R,3), t(K,3), t(Y,3),
        t(R,6), t(K,6), t(Y,6),
        t(Y,13),
    ]
    score = evaluate_close(hand15, joker_discarded_earlier=False)
    assert_eq(score, 500, "Minors big close should score 500")

def test_scoring_monocolor_1000():
    # Monocolor close (all non-free tiles same color): base 1000.
    # 14 melded tiles all red:
    # Run(7): 1-2-3-4-5-6-7 red
    # Run(7): 7-8-9-10-11-12-13 red  (note: duplicates are allowed only if your tile set has duplicates;
    #                                  in your game there are duplicates; your engine allows duplicates because Tile equality includes both fields.)
    # Free: 5 blue
    #
    # IMPORTANT: This assumes your game has duplicates per color/number (it does in remi pe  tabla),
    # and your engine currently allows duplicates.
    hand15 = [
        t(R,1), t(R,2), t(R,3), t(R,4), t(R,5), t(R,6), t(R,7),
        t(R,7), t(R,8), t(R,9), t(R,10), t(R,11), t(R,12), t(R,13),
        t(B,5),
    ]
    score = evaluate_close(hand15, joker_discarded_earlier=False)
    assert_eq(score, 1000, "Monocolor close should score 1000")

def run_all():
    test_wrap_rule()
    test_4_tile_melds()
    test_full_partition_sanity()
    test_scoring_normal_and_joker_multiplier()
    test_scoring_big_minors_500()
    test_scoring_monocolor_1000()
    print("All tests passed.")

if __name__ == "__main__":
    run_all()
