from dataclasses import dataclass

@dataclass(frozen=True, eq=True)
class Tile:
    color: int
    number: int

    def is_joker(self):
        return self.color == -1 and self.number == -1

INF = 10**9

def popcount(x: int) -> int:
    return x.bit_count()


def count_jokers_in_melds(tiles):
    """
    Count how many jokers are in the given set of tiles.
    This is used for applying joker penalties.
    """
    return sum(1 for t in tiles if t.is_joker())

_VALID_MELD_CACHE = {}

def _meld_key(tiles):
    # Order-independent key
    return tuple(sorted((t.color, t.number) for t in tiles))

def is_valid_set(tiles):
    key = ("set", _meld_key(tiles))
    if key in _VALID_MELD_CACHE:
        return _VALID_MELD_CACHE[key]

    if len(tiles) < 3:
        res = (False, 0)
        _VALID_MELD_CACHE[key] = res
        return res

    jokers = [t for t in tiles if t.is_joker()]
    non_jokers = [t for t in tiles if not t.is_joker()]

    if not non_jokers:
        res = (False, 0)  # can't form set from only jokers
        _VALID_MELD_CACHE[key] = res
        return res

    numbers = {t.number for t in non_jokers}
    colors = {t.color for t in non_jokers}

    if len(numbers) > 1:
        res = (False, 0)
        _VALID_MELD_CACHE[key] = res
        return res

    if len(colors) != len(non_jokers):
        res = (False, 0)
        _VALID_MELD_CACHE[key] = res
        return res

    if len(non_jokers) + len(jokers) > 4:
        res = (False, 0)
        _VALID_MELD_CACHE[key] = res
        return res

    res = (True, len(jokers))
    _VALID_MELD_CACHE[key] = res
    return res


def is_valid_run(tiles):
    key = ("run", _meld_key(tiles))
    if key in _VALID_MELD_CACHE:
        return _VALID_MELD_CACHE[key]

    if len(tiles) < 3:
        res = (False, 0)
        _VALID_MELD_CACHE[key] = res
        return res

    jokers = [t for t in tiles if t.is_joker()]
    non_jokers = [t for t in tiles if not t.is_joker()]

    if not non_jokers:
        res = (False, 0)
        _VALID_MELD_CACHE[key] = res
        return res

    colors = {t.color for t in non_jokers}
    if len(colors) != 1:
        res = (False, 0)
        _VALID_MELD_CACHE[key] = res
        return res

    nums = sorted(t.number for t in non_jokers)
    j = len(jokers)

    def jokers_needed_for_linear(seq):
        need = 0
        for a, b in zip(seq, seq[1:]):
            if b <= a:
                return None
            need += (b - a - 1)
        return need

    # Case 1: normal run (no wrap)
    need = jokers_needed_for_linear(nums)
    if need is not None and need <= j:
        res = (True, j)
        _VALID_MELD_CACHE[key] = res
        return res

    # Case 2: wrap allowed ONLY at the very end: ... 11,12,13,1
    if 1 in nums:
        tail = [n for n in nums if n != 1]
        if tail:
            tail_need = jokers_needed_for_linear(tail)
            if tail_need is not None:
                extend_need = 13 - tail[-1]
                if extend_need >= 0:
                    total_need = tail_need + extend_need
                    if total_need <= j:
                        res = (True, j)
                        _VALID_MELD_CACHE[key] = res
                        return res

    res = (False, 0)
    _VALID_MELD_CACHE[key] = res
    return res


from itertools import combinations

def best_partition_jokers_used(tiles, memo=None):
    """
    Returns:
      - None if tiles cannot be fully partitioned into valid melds
      - an int = minimum number of jokers used inside melds across all valid partitions
    """
    if memo is None:
        memo = {}

    tiles = tuple(sorted(tiles, key=lambda t: (t.color, t.number)))

    if tiles in memo:
        return memo[tiles]

    if not tiles:
        return 0

    first = tiles[0]
    best = None

    for size in range(3, len(tiles) + 1):
        for combo in combinations(tiles, size):
            if first not in combo:
                continue

            valid_set, jokers_set = is_valid_set(combo)
            valid_run, jokers_run = is_valid_run(combo)

            if not (valid_set or valid_run):
                continue

            jokers_in_this_meld = jokers_set if valid_set else jokers_run

            remaining = list(tiles)
            for t in combo:
                remaining.remove(t)

            sub = best_partition_jokers_used(remaining, memo)
            if sub is None:
                continue

            total = jokers_in_this_meld + sub
            if best is None or total < best:
                best = total

    memo[tiles] = best
    return best

def can_fully_partition(tiles):
    """
    True iff ALL tiles can be partitioned into valid melds (no leftovers).
    """
    return best_partition_jokers_used(tiles) is not None


def evaluate_close(hand, joker_discarded_earlier=False):
    """
    Returns the best achievable close score for this 15-tile hand,
    or None if no valid close exists.

    IMPORTANT: Uses score_close directly so multipliers/penalties are never dropped.
    """
    if len(hand) != 15:
        return None

    best = None
    for i in range(15):
        s = score_close(hand, i, joker_discarded_earlier=joker_discarded_earlier)
        if s is None:
            continue
        if best is None or s > best:
            best = s
    return best



def _is_minors(nonfree_tiles):
    # All non-free tiles must be 1..7 (jokers allowed but don't help satisfy range)
    for t in nonfree_tiles:
        if t.is_joker():
            continue
        if not (1 <= t.number <= 7):
            return False
    return True

def _is_majors(nonfree_tiles):
    # All non-free tiles must be 8..13 or 1
    for t in nonfree_tiles:
        if t.is_joker():
            continue
        if not (t.number == 1 or 8 <= t.number <= 13):
            return False
    return True

def _is_bicolor(nonfree_tiles):
    colors = {t.color for t in nonfree_tiles if not t.is_joker()}
    return len(colors) == 2

def _is_monocolor(nonfree_tiles):
    colors = {t.color for t in nonfree_tiles if not t.is_joker()}
    return len(colors) == 1 and len(colors) > 0




def score_close(hand, free_index, joker_discarded_earlier=False):
    if len(hand) != 15:
        return None

    free_tile = hand[free_index]
    nonfree = hand[:free_index] + hand[free_index+1:]

    # Only allow a valid close if the 14 cards are fully partitionable
    if not can_fully_partition(nonfree):
        return None

    jokers_used_in_melds = best_partition_jokers_used(nonfree)

    # Determine base close category
    if _is_monocolor(nonfree):
        base = 1000
    elif _is_minors(nonfree) or _is_majors(nonfree) or _is_bicolor(nonfree):
        base = 500
    else:
        base = 250

    # Apply joker penalties:
    base_after_penalty = base - 50 * jokers_used_in_melds

    # Apply multiplier:
    mult = 1
    if free_tile.is_joker():
        mult = 4 if joker_discarded_earlier else 2

    return base_after_penalty * mult


def max_meld_coverage_fast(tiles):
    """
    Fast max coverage using dp_cov instead of recursion.
    """
    n = len(tiles)
    full = (1 << n) - 1
    _, _, dp_cov = get_hand_dp(tiles, allow_joker_melds=False)
    return dp_cov[full]


def hand_strength(hand, joker_discarded_earlier=False):
    """
    Fast strength eval using DP.
    """
    if len(hand) == 15:
        plan = best_close_plan(hand, joker_discarded_earlier=joker_discarded_earlier)
        if plan is not None:
            return {"can_close": True, "close_score": plan["score"], "coverage": 14, "unmatched": 1}

        cov = max_meld_coverage_fast(hand)
        return {"can_close": False, "close_score": None, "coverage": cov, "unmatched": 15 - cov}

    # len == 14 (pre-draw / post-discard board)
    cov = max_meld_coverage_fast(hand)
    return {"can_close": False, "close_score": None, "coverage": cov, "unmatched": 14 - cov}


def best_discard(hand, joker_discarded_earlier=False):
    """
    Returns (index, tile, evaluation_dict) for the best discard.
    Assumes len(hand) == 15.

    Uses DP coverage (no jokers inside melds in normal play) with a correct
    original-index -> dp-index mapping for masks.
    """
    assert len(hand) == 15
    full = (1 << 15) - 1

    # Get dp tables AND mapping
    _tiles_dp, orig_to_dp, _dp_min, dp_cov = get_hand_dp(hand, allow_joker_melds=False, need_index_map=True)

    best = None

    for i in range(15):
        dp_i = orig_to_dp[i]                      # <-- critical fix
        rem_mask = full ^ (1 << dp_i)

        cov = dp_cov[rem_mask]
        unmatched = 14 - cov

        hand14 = hand[:i] + hand[i+1:]

        pot = hand_potential(hand14)
        near = near_meld_score(hand14, cov=cov)

        big_mode = (
            pot["dominant_color_sum"] >= 12 or
            pot["top2_color_sum"] >= 12 or
            pot["minor_tiles"] >= 12 or
            pot["major_tiles"] >= 12
        )

        pot_top2  = pot["top2_color_sum"] if big_mode else 0
        pot_dom   = pot["dominant_color_sum"] if big_mode else 0
        pot_minor = pot["minor_tiles"] if big_mode else 0
        pot_major = pot["major_tiles"] if big_mode else 0

        prefer_discard_nonjoker = 1 if not hand[i].is_joker() else 0

        score = (
            -unmatched,
            cov,
            prefer_discard_nonjoker,
            near,
            pot_top2,
            pot_dom,
            pot_minor,
            pot_major,
        )

        if best is None or score > best["score"]:
            best = {
                "index": i,
                "tile": hand[i],
                "score": score,
                "coverage": cov,
                "unmatched": unmatched,
            }

    return best["index"], best["tile"], {"coverage": best["coverage"], "unmatched": best["unmatched"]}

def debug_best_discard(hand15, top_k=10, joker_discarded_earlier=False):
    """
    Prints the top discard candidates with their scoring breakdown.
    """
    assert len(hand15) == 15
    full = (1 << 15) - 1
    _, _dp_min, dp_cov = get_hand_dp(hand15, allow_joker_melds=False)

    rows = []
    for i in range(15):
        rem_mask = full ^ (1 << i)
        cov = dp_cov[rem_mask]
        unmatched = 14 - cov
        hand14 = hand15[:i] + hand15[i+1:]
        pot = hand_potential(hand14)
        near = near_meld_score(hand14, cov=cov)

        big_mode = (
                pot["dominant_color_sum"] >= 12 or
                pot["top2_color_sum"] >= 12 or
                pot["minor_tiles"] >= 12 or
                pot["major_tiles"] >= 12
        )

        # Only use pot metrics if we're genuinely close to a big close.
        pot_top2 = pot["top2_color_sum"] if big_mode else 0
        pot_dom = pot["dominant_color_sum"] if big_mode else 0
        pot_minor = pot["minor_tiles"] if big_mode else 0
        pot_major = pot["major_tiles"] if big_mode else 0

        score = (
            -unmatched,
            cov,
            near,  # near-meld progress matters now
            pot_top2,  # big-close direction ONLY when close
            pot_dom,
            pot_minor,
            pot_major,
        )

        rows.append({
            "i": i,
            "discard": hand15[i],
            "cov": cov,
            "unmatched": unmatched,
            "top2": pot["top2_color_sum"],
            "dom": pot["dominant_color_sum"],
            "minor": pot["minor_tiles"],
            "major": pot["major_tiles"],
            "near": near,
            "score": score,
        })

    rows.sort(key=lambda r: r["score"], reverse=True)

    print(f"Top {top_k} discard candidates:")
    for r in rows[:top_k]:
        print(
            f"i={r['i']:2d} discard={r['discard']}  "
            f"cov={r['cov']:2d} unmatch={r['unmatched']:2d}  "
            f"top2={r['top2']:2d} dom={r['dom']:2d}  "
            f"minor={r['minor']:2d} major={r['major']:2d} near={r['near']:4.1f}  "
            f"score={r['score']}"
        )


def best_close_plan(hand, joker_discarded_earlier=False):
    if len(hand) != 15:
        return None

    full = (1 << 15) - 1

    # IMPORTANT: get index mapping so DP bit positions match original hand indices
    _tiles_dp, orig_to_dp, dp_min, _dp_cov = get_hand_dp(
        hand,
        allow_joker_melds=True,     # allow jokers in melds, then apply your policy filter below
        need_index_map=True
    )

    best = None

    for i in range(15):
        dp_i = orig_to_dp[i]                 # map original index -> dp index
        nonfree_mask = full ^ (1 << dp_i)    # remove the FREE tile in DP-space

        jokers_used = dp_min[nonfree_mask]
        if jokers_used >= INF:
            continue

        free_tile = hand[i]
        nonfree_tiles = [hand[j] for j in range(15) if j != i]

        # Determine base category from actual 14 tiles
        if _is_monocolor(nonfree_tiles):
            base = 1000
        elif _is_minors(nonfree_tiles) or _is_majors(nonfree_tiles) or _is_bicolor(nonfree_tiles):
            base = 500
        else:
            base = 250

        base_after_penalty = base - 50 * jokers_used

        mult = 1
        if free_tile.is_joker():
            mult = 4 if joker_discarded_earlier else 2

        score = base_after_penalty * mult

        # Your personal policy: avoid joker(s) inside melds unless resulting close is > 500
        if jokers_used > 0 and score <= 500:
            continue

        if best is None or score > best["score"]:
            best = {"free_index": i, "free_tile": free_tile, "score": score}

    return best



def recommend_action_after_draw(hand, joker_discarded_earlier=False):
    """
    Assumes it's your turn AFTER you've drawn (so you have 15 tiles).
    Recommends either:
      - CLOSE (with best free tile)
      - DISCARD (best discard choice)
    """
    plan = best_close_plan(hand, joker_discarded_earlier=joker_discarded_earlier)
    if plan is not None:
        return {
            "action": "CLOSE",
            "free_index": plan["free_index"],
            "free_tile": plan["free_tile"],
            "score": plan["score"],
        }

    idx, tile, info = best_discard(hand, joker_discarded_earlier=joker_discarded_earlier)
    return {
        "action": "DISCARD",
        "discard_index": idx,
        "discard_tile": tile,
        "info": info,
    }

def hand_potential(hand):
    """
    Returns heuristic proximity measures to big closes:
      - minor_tiles (count toward minors)
      - major_tiles (count toward majors)
      - top2_color_sum (dominant + second color)
      - dominant_color_sum (largest single color group)
      - total_tiles
    """
    from collections import Counter

    non_jokers = [t for t in hand if not t.is_joker()]
    jokers = [t for t in hand if t.is_joker()]

    color_counts = Counter(t.color for t in non_jokers)

    # Numeric categories
    minor_ok = [t for t in non_jokers if 1 <= t.number <= 7]
    major_ok = [t for t in non_jokers if t.number == 1 or 8 <= t.number <= 13]

    # Combine jokers with both categories
    minor_tiles = len(minor_ok) + len(jokers)
    major_tiles = len(major_ok) + len(jokers)

    # Color dominance
    sorted_colors = sorted(color_counts.values(), reverse=True)
    top = sorted_colors + [0, 0]
    top2_color_sum = top[0] + top[1] + len(jokers)
    dominant_color_sum = top[0] + len(jokers)

    return {
        "minor_tiles": minor_tiles,
        "major_tiles": major_tiles,
        "top2_color_sum": top2_color_sum,
        "dominant_color_sum": dominant_color_sum,
        "total_tiles": len(hand),
    }

def recommend_action(hand, joker_discarded_earlier=False):
    """
    Recommends CLOSE / DISCARD based on heuristic potential.
    """
    plan = best_close_plan(hand, joker_discarded_earlier=joker_discarded_earlier)
    strength = hand_strength(hand, joker_discarded_earlier=joker_discarded_earlier)
    pot = hand_potential(hand)

    # If strong close exists (>= 1000), just close
    if plan is not None and plan["score"] >= 1000:
        return {"action": "CLOSE", **plan}

    # If 500+ close exists AND hand is NOT trending toward even better
    if plan is not None and plan["score"] >= 500:
        if not (
            pot["dominant_color_sum"] >= 13 or
            pot["top2_color_sum"] >= 12 or
            pot["minor_tiles"] >= 12 or
            pot["major_tiles"] >= 12
        ):
            return {"action": "CLOSE", **plan}

    # If only simple close (250) and no real big potential → close
    if plan is not None and plan["score"] == 250:
        if not (
            pot["minor_tiles"] >= 12 or
            pot["major_tiles"] >= 12 or
            pot["top2_color_sum"] >= 12
        ):
            return {"action": "CLOSE", **plan}

    # Otherwise, recommend discard
    idx, tile, info = best_discard(hand, joker_discarded_earlier=joker_discarded_earlier)
    return {"action": "DISCARD", "discard_index": idx, "discard_tile": tile, "info": info}

def evaluate_with_added_tile(hand, new_tile, joker_discarded_earlier=False):
    """
    Simulates what the hand would look like *after drawing* or *taking a discard*,
    before deciding what to discard.

    Returns a dict with:
      - strength info
      - best close score if available
      - best potential
    """
    new_hand = hand + [new_tile]
    strength = hand_strength(new_hand, joker_discarded_earlier=joker_discarded_earlier)
    potential = hand_potential(new_hand)
    best_close = best_close_plan(new_hand, joker_discarded_earlier=joker_discarded_earlier)

    return {
        "hand": new_hand,
        "strength": strength,
        "potential": potential,
        "best_close": best_close
    }

def recommend_take_or_draw(hand14, top_discard, joker_discarded_earlier=False):
    """
    Decide whether to TAKE the top discard or DRAW from the deck.

    Rules:
    0) Cannot take a discarded joker in this ruleset -> always DRAW_DECK.
    1) If taking the discard allows an immediate valid close -> TAKE.
    2) Otherwise, take only if it improves the post-discard position by a meaningful margin.
    """
    # Rule 0: cannot take discarded joker
    if top_discard is not None and top_discard.is_joker():
        return {
            "action": "DRAW_DECK",
            "reason": "Cannot take a discarded joker in this ruleset",
            "take_score": None,
            "baseline_score": None,
        }

    # Evaluate taking the discard for immediate close
    take_eval = evaluate_with_added_tile(hand14, top_discard, joker_discarded_earlier)
    if take_eval["best_close"] is not None:
        return {
            "action": "TAKE_DISCARD",
            "reason": "Taking discard allows immediate valid close"
        }

    baseline_score = position_score_14(hand14, joker_discarded_earlier=joker_discarded_earlier)

    di, dtile, resulting14, take_score = apply_best_discard_and_score(
        hand14=hand14,
        top_discard=top_discard,
        joker_discarded_earlier=joker_discarded_earlier
    )

    IMPROVEMENT_MARGIN = 5.0

    if take_score >= baseline_score + IMPROVEMENT_MARGIN:
        return {
            "action": "TAKE_DISCARD",
            "reason": "Taking discard improves post-discard position",
            "discard_index_after_take": di,
            "discard_tile_after_take": dtile,
            "take_score": take_score,
            "baseline_score": baseline_score,
        }

    return {
        "action": "DRAW_DECK",
        "reason": "Discard does not improve post-discard position",
        "take_score": take_score,
        "baseline_score": baseline_score,
    }




# Debug helper — test full partition logic directly
def debug_can_fully_partition(tiles):
    """
    Print partition attempt results so we can see what's going on.
    """
    from pprint import pprint

    jokers_used = best_partition_jokers_used(tiles)

    print("Tiles:", [(t.color, t.number) for t in tiles])
    print("Can fully partition (None means NO):", jokers_used)

    if jokers_used is not None:
        print("Jokers used in partition:", jokers_used)
    else:
        print("No full partition found.")

def position_score_14(hand14, joker_discarded_earlier=False):
    cov = max_meld_coverage_fast(hand14)
    pot = hand_potential(hand14)
    near = near_meld_score(hand14, cov=cov)

    return (
        cov * 10
        + pot["minor_tiles"]
        + pot["major_tiles"]
        + pot["top2_color_sum"]
        + pot["dominant_color_sum"]
        + near * 0.5
    )



def apply_best_discard_and_score(hand15=None, joker_discarded_earlier=False, hand14=None, top_discard=None):
    """
    Computes the best discard after you have 15 tiles, then scores the resulting 14-tile position.

    Supports TWO calling styles (backwards compatible):
      A) apply_best_discard_and_score(hand15, joker_discarded_earlier=False)
      B) apply_best_discard_and_score(hand14=..., top_discard=..., joker_discarded_earlier=False)

    Returns: (discard_index, discard_tile, resulting14, resulting_score)
    """
    # If called with hand14 + top_discard, build the 15-tile hand
    if hand14 is not None:
        if top_discard is None:
            raise ValueError("apply_best_discard_and_score: top_discard is required when using hand14=")
        hand15 = list(hand14) + [top_discard]

    if hand15 is None:
        raise ValueError("apply_best_discard_and_score: must provide hand15 or hand14+top_discard")

    if len(hand15) != 15:
        raise ValueError(f"apply_best_discard_and_score: expected 15 tiles, got {len(hand15)}")

    # Pick best discard on the 15-tile hand
    di, dtile, _info = best_discard(hand15, joker_discarded_earlier=joker_discarded_earlier)

    # Resulting 14-tile position after discarding
    resulting14 = hand15[:di] + hand15[di+1:]

    # Score that 14-tile position (normal-play scoring)
    resulting_score = position_score_14(resulting14, joker_discarded_earlier=joker_discarded_earlier)

    return di, dtile, resulting14, resulting_score


def recommend_turn_start(hand14, top_discard, joker_discarded_earlier=False):
    """
    Returns a dict with stage TURN_START decision: TAKE_DISCARD or DRAW_DECK.

    Rule: You cannot take a discarded joker in this game.
    """
    # If top discard is a joker, you MUST draw from deck
    if top_discard is not None and top_discard.is_joker():
        return {
            "stage": "TURN_START",
            "draw": {
                "action": "DRAW_DECK",
                "reason": "Cannot take a discarded joker in this ruleset",
            },
            "note": "If you draw from deck, call recommend_after_drawing_deck(hand14, drawn_tile)."
        }

    draw = recommend_take_or_draw(hand14, top_discard, joker_discarded_earlier=joker_discarded_earlier)
    return {
        "stage": "TURN_START",
        "draw": draw,
        "note": "If you draw from deck, call recommend_after_drawing_deck(hand14, drawn_tile)."
    }


def recommend_after_taking_discard(hand14, top_discard, joker_discarded_earlier=False):
    """
    If you take the top discard, you now have 15 tiles.
    This returns the full best action: CLOSE or DISCARD (and which tile).
    """
    hand15 = hand14 + [top_discard]
    return recommend_action(hand15, joker_discarded_earlier=joker_discarded_earlier)

def recommend_after_drawing_deck(hand14, drawn_tile, joker_discarded_earlier=False):
    """
    After drawing from deck, you have 15 tiles (known drawn tile).
    Returns CLOSE or DISCARD recommendation.
    """
    hand15 = hand14 + [drawn_tile]
    return recommend_action(hand15, joker_discarded_earlier=joker_discarded_earlier)

def recommend_turn_start(hand14, top_discard, joker_discarded_earlier=False):
    """
    Turn-start advisor:
      - Decide TAKE_DISCARD vs DRAW_DECK.
      - If TAKE_DISCARD, also returns after_take recommendation immediately.
      - If DRAW_DECK, tells you to call recommend_after_drawing_deck() once you see the drawn tile.
    """
    draw = recommend_take_or_draw(hand14, top_discard, joker_discarded_earlier=joker_discarded_earlier)

    if draw["action"] == "TAKE_DISCARD":
        after_take = recommend_after_taking_discard(hand14, top_discard, joker_discarded_earlier=joker_discarded_earlier)
        return {"stage": "TURN_START", "draw": draw, "after_take": after_take}

    return {
        "stage": "TURN_START",
        "draw": draw,
        "note": "If you draw from deck, call recommend_after_drawing_deck(hand14, drawn_tile)."
    }

def generate_melds_for_hand(hand_tiles, allow_joker_melds=True):
    """
    Returns list of meld descriptors:
      (mask, jokers_used_in_meld, size)
    where mask is over indices in hand_tiles.
    """
    n = len(hand_tiles)
    melds = []
    for mask in range(1, 1 << n):
        k = popcount(mask)
        if k < 3:
            continue
        tiles = [hand_tiles[i] for i in range(n) if (mask >> i) & 1]
        if not allow_joker_melds and any(t.is_joker() for t in tiles):
            continue
        ok_set, jok_set = is_valid_set(tiles)
        if ok_set:
            melds.append((mask, jok_set, k))
            continue
        ok_run, jok_run = is_valid_run(tiles)
        if ok_run:
            melds.append((mask, jok_run, k))
    return melds

def dp_min_jokers_exact_cover(n, melds):
    """
    dp_min[mask] = min jokers used to cover exactly mask using disjoint melds
    """
    full = (1 << n) - 1
    dp = [INF] * (1 << n)
    dp[0] = 0
    for mask in range(1 << n):
        cur = dp[mask]
        if cur == INF:
            continue
        for m, jok, _size in melds:
            if mask & m:
                continue
            nm = mask | m
            val = cur + jok
            if val < dp[nm]:
                dp[nm] = val
    return dp

def dp_max_coverage(n, melds):
    """
    dp_cov[mask] = max tiles that can be covered by melds inside mask (leftovers allowed)
    """
    dp = [0] * (1 << n)
    for mask in range(1 << n):
        # option: drop one tile (leave it uncovered)
        if mask:
            lsb = mask & -mask
            dp[mask] = max(dp[mask], dp[mask ^ lsb])

        # option: use a meld fully contained in mask
        for m, _jok, size in melds:
            if (m & mask) == m:
                dp[mask] = max(dp[mask], dp[mask ^ m] + size)
    return dp

def _hand_key(hand_tiles):
    # ORDER-DEPENDENT: required because DP masks depend on index positions
    return tuple((t.color, t.number) for t in hand_tiles)



_DP_CACHE = {}

_HAND_DP_CACHE = {}

def _hand_key(hand_tiles):
    # order-independent multiset key
    return tuple(sorted((t.color, t.number) for t in hand_tiles))

def get_hand_dp(hand_tiles, allow_joker_melds=False, need_index_map=False):
    """
    Returns DP tables for this multiset of tiles, computed in a canonical sorted order.

    If need_index_map=True, also returns orig_to_dp: list mapping original indices -> dp indices,
    so you can build masks correctly when removing tiles by original index.

    Returns:
      - if need_index_map=False:
          (tiles_dp, dp_min, dp_cov)
      - if need_index_map=True:
          (tiles_dp, orig_to_dp, dp_min, dp_cov)
    """
    key = (_hand_key(hand_tiles), bool(allow_joker_melds))
    if key in _HAND_DP_CACHE:
        tiles_dp, dp_min, dp_cov = _HAND_DP_CACHE[key]
    else:
        # canonical order the DP is built on
        tiles_dp = sorted(list(hand_tiles), key=lambda t: (t.color, t.number))
        # build meld list based on tiles_dp order
        melds = generate_melds_for_hand(tiles_dp, allow_joker_melds=allow_joker_melds)

        n = len(tiles_dp)
        dp_cov = dp_max_coverage(n, melds)
        dp_min = dp_min_jokers_exact_cover(n, melds)

        _HAND_DP_CACHE[key] = (tiles_dp, dp_min, dp_cov)

    if not need_index_map:
        return tiles_dp, dp_min, dp_cov

    # Build mapping from ORIGINAL index -> DP index (handles duplicates safely)
    from collections import defaultdict, deque

    # queues of dp indices for each (color, number)
    buckets = defaultdict(list)
    for j, t in enumerate(tiles_dp):
        buckets[(t.color, t.number)].append(j)
    queues = {k: deque(v) for k, v in buckets.items()}

    orig_to_dp = []
    for t in hand_tiles:
        dq = queues.get((t.color, t.number))
        if not dq:
            raise RuntimeError("get_hand_dp mapping error: tile multiset mismatch")
        orig_to_dp.append(dq.popleft())

    return tiles_dp, orig_to_dp, dp_min, dp_cov


def min_jokers_to_cover_mask(hand15, cover_mask):
    _, dp_min, _ = get_hand_dp(hand15)
    val = dp_min[cover_mask]
    return None if val >= INF else val

def near_meld_score(hand, cov=None):
    """
    Coverage-aware near-structure score.

    Big idea:
      - Early/mid: building new melds and set-to-run conversions can be good.
      - When cov >= 12 (you already have ~4 melds), prioritize LIPITURI:
          * extending existing run segments (same color)
          * adding missing colors to sets
        and downweight "new 3-meld seeds" that compete for the same tiles.

    cov: completed-meld coverage on this hand (14 tiles), from DP (no jokers in melds).
    """
    from collections import defaultdict, Counter

    non_jokers = [t for t in hand if not t.is_joker()]
    jokers = sum(1 for t in hand if t.is_joker())

    # If caller doesn't pass cov, estimate stage from hand size only (safe fallback)
    stage_lipituri = (cov is not None and cov >= 12)

    score = 0.0

    # Count exact tile duplicates (because you have 2 copies in the deck)
    tile_counts = Counter((t.color, t.number) for t in non_jokers)

    # ---- A) Set strength + "missing color" lipituri ----
    by_num = defaultdict(list)
    for t in non_jokers:
        by_num[t.number].append(t.color)

    for num, colors_list in by_num.items():
        colors = set(colors_list)
        uniq = len(colors)

        # Base set strength
        if uniq == 2:
            score += 5
        elif uniq == 3:
            score += 12
        elif uniq >= 4:
            score += 18

        # LIPITURA for sets: if you have 3 colors of a number, the 4th color is a strong extender.
        # This is always good and does NOT create the "no free card" trap.
        if uniq == 3:
            score += 4  # represent "one missing color exists"

    # ---- B) Runs within each color: reward EXTENDING existing segments more than starting new ones ----
    by_color = defaultdict(set)
    for t in non_jokers:
        by_color[t.color].add(t.number)

    def longest_consecutive_segment(nums_set):
        """Longest run length in a set of numbers, ignoring wrap for simplicity."""
        if not nums_set:
            return 0
        best = 1
        nums = sorted(nums_set)
        cur = 1
        for a, b in zip(nums, nums[1:]):
            if b == a + 1:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
        return best

    for c, s in by_color.items():
        nums = sorted(s)
        sset = set(nums)

        longest = longest_consecutive_segment(sset)

        # Weighting:
        # - If we're in lipituri stage, only reward adjacency strongly when this color already has a real segment.
        #   Otherwise adjacency like B4-B5 in an otherwise "dead" color shouldn't look amazing.
        adj_w = 4.0 if (not stage_lipituri or longest >= 3) else 1.0
        gap_w = 2.0 if (not stage_lipituri or longest >= 3) else 0.5

        # Adjacent pairs + 1-gap pairs
        for x in nums:
            if x + 1 in sset:
                score += adj_w
            if x + 2 in sset:
                score += gap_w

        # Length-weighted bridge (m-1 and m+1 exist)
        def consec_left(x):
            k = 0
            while (x - k) in sset:
                k += 1
            return k

        def consec_right(x):
            k = 0
            while (x + k) in sset:
                k += 1
            return k

        for m in range(2, 13):
            if (m not in sset) and (m - 1 in sset) and (m + 1 in sset):
                left_len = consec_left(m - 1)
                right_len = consec_right(m + 1)

                base = 6.0 + 3.0 * (left_len + right_len)
                # In lipituri stage, bridges in "dead" colors are less valuable
                if stage_lipituri and longest < 3:
                    base *= 0.5

                score += base

        # Terminal wrap hints (very small)
        if 12 in sset and 13 in sset:
            score += 2.0
        if 13 in sset and 1 in sset:
            score += 0.5

    # ---- C) Set-to-run synergy (ONLY when it doesn't fake-progress) ----
    # This is the part that was overestimating B4 in your example.
    # We now only grant it if:
    #   - not in lipituri stage, OR
    #   - it is genuinely feasible without destroying value:
    #       * you have a 4-color set (can break and still have 3-set), OR
    #       * you have a duplicate of the exact color+number tile (two copies in hand)
    if not stage_lipituri:
        synergy_scale = 1.0
    else:
        synergy_scale = 0.0  # default OFF in lipituri stage

    if synergy_scale > 0:
        set_size_by_num = {num: len(set(colors)) for num, colors in by_num.items()}
        colors_by_num = {num: set(colors) for num, colors in by_num.items()}

        for t in non_jokers:
            n = t.number
            c = t.color

            for base in (n - 1, n + 1):
                if 1 <= base <= 13:
                    cols = colors_by_num.get(base, set())
                    sz = set_size_by_num.get(base, 0)

                    if sz >= 3 and c in cols:
                        # Feasibility: either 4-set or duplicate tile exists for that base tile
                        has_dup_base_tile = tile_counts[(c, base)] >= 2
                        feasible = (sz >= 4) or has_dup_base_tile
                        if feasible:
                            score += 10.0 * synergy_scale

    # ---- D) Jokers: small flexibility bonus (we avoid joker-in-melds in DP, but joker as free is valuable) ----
    score += jokers * 4.0

    return float(score)

