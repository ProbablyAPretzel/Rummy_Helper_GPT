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
    Fast close evaluation: returns best close score or None.
    """
    if len(hand) != 15:
        return None
    plan = best_close_plan(hand, joker_discarded_earlier=joker_discarded_earlier)
    return None if plan is None else plan["score"]


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
    Uses DP coverage and strategy tiebreakers (big-potential + near-meld).
    """
    assert len(hand) == 15
    full = (1 << 15) - 1

    # DP coverage table for THIS 15-tile hand
    _, _dp_min, dp_cov = get_hand_dp(hand, allow_joker_melds=False)

    best = None

    for i in range(15):
        rem_mask = full ^ (1 << i)
        cov = dp_cov[rem_mask]
        unmatched = 14 - cov

        # Build the 14-tile hand after discarding i
        hand14 = hand[:i] + hand[i+1:]

        # Tiebreakers: keep big-close potential + keep good near-meld shape
        pot = hand_potential(hand14)
        near = near_meld_score(hand14)

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

        if best is None or score > best["score"]:
            best = {
                "index": i,
                "tile": hand[i],
                "score": score,
                "coverage": cov,
                "unmatched": unmatched,
                "near": near,
            }

    return best["index"], best["tile"], {"coverage": best["coverage"], "unmatched": best["unmatched"]}

def debug_best_discard(hand15, top_k=10, joker_discarded_earlier=False):
    """
    Prints the top discard candidates with their scoring breakdown.
    """
    assert len(hand15) == 15
    full = (1 << 15) - 1
    _, _dp_min, dp_cov = get_hand_dp(hand15)

    rows = []
    for i in range(15):
        rem_mask = full ^ (1 << i)
        cov = dp_cov[rem_mask]
        unmatched = 14 - cov
        hand14 = hand15[:i] + hand15[i+1:]
        pot = hand_potential(hand14)
        near = near_meld_score(hand14)

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
    _, dp_min, _ = get_hand_dp(hand)

    best = None
    for i in range(15):
        nonfree_mask = full ^ (1 << i)
        jokers_used = dp_min[nonfree_mask]
        if jokers_used >= INF:
            continue

        free_tile = hand[i]

        # base category from the actual 14 tiles (build list)
        nonfree_tiles = [hand[j] for j in range(15) if j != i]
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
    Decide whether to take the top discard or draw from deck.
    Phase-1 heuristic:
      - Take if it enables a good close (>=500), or if it improves your best post-discard position.
    """
    IMPROVEMENT_MARGIN = 5.0

    # Baseline score if we DON'T take discard (proxy for "draw deck")
    baseline_score = position_score_14(hand14, joker_discarded_earlier=joker_discarded_earlier)

    # Simulate taking discard
    hand15_take = hand14 + [top_discard]

    # If taking discard enables an immediate close, that's a strong reason to take
    close_plan = best_close_plan(hand15_take, joker_discarded_earlier=joker_discarded_earlier)
    if close_plan is not None:
        # In Phase 1, only auto-take if it's 500+ (or joker-multiplied).
        # 250 closes are often not worth forcing via discard-take.
        if close_plan["score"] >= 500:
            return {
                "action": "TAKE_DISCARD",
                "reason": f"Taking discard enables close for {close_plan['score']}",
                "close_plan": close_plan
            }

    # Otherwise evaluate best discard after taking
    di, dtile, resulting14, take_score = apply_best_discard_and_score(
        hand15_take,
        joker_discarded_earlier=joker_discarded_earlier
    )

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
    near = near_meld_score(hand14)

    return (
        cov * 10
        + pot["minor_tiles"]
        + pot["major_tiles"]
        + pot["top2_color_sum"]
        + pot["dominant_color_sum"]
        + near * 0.5
    )



def apply_best_discard_and_score(hand15, joker_discarded_earlier=False):
    """
    Given 15 tiles, choose best discard and return:
      (discard_index, discard_tile, resulting_hand14, score14)
    """
    idx, tile, _ = best_discard(hand15, joker_discarded_earlier=joker_discarded_earlier)
    hand14 = hand15[:idx] + hand15[idx+1:]
    score14 = position_score_14(hand14, joker_discarded_earlier=joker_discarded_earlier)
    return idx, tile, hand14, score14

def recommend_turn_start(hand14, top_discard, joker_discarded_earlier=False):
    """
    Turn-start advisor:
      - Decide TAKE_DISCARD vs DRAW_DECK.
      - If TAKE_DISCARD, also recommend the discard (or close) immediately.
      - If DRAW_DECK, returns draw recommendation and you call recommend_action() after you see the drawn tile.
    """
    draw_decision = recommend_take_or_draw(hand14, top_discard, joker_discarded_earlier=joker_discarded_earlier)

    if draw_decision["action"] == "TAKE_DISCARD":
        hand15 = hand14 + [top_discard]
        after_take = recommend_action(hand15, joker_discarded_earlier=joker_discarded_earlier)
        return {
            "stage": "TURN_START",
            "draw": draw_decision,
            "after_take": after_take
        }

    return {
        "stage": "TURN_START",
        "draw": draw_decision,
        "note": "If you draw from deck, call recommend_action(hand15) after you see the drawn tile."
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
        ok_set, jok_set = is_valid_set(tiles)
        if ok_set:
            melds.append((mask, jok_set, k))
            continue
        if not allow_joker_melds and any(t.is_joker() for t in combo):
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

def get_hand_dp(hand_tiles, allow_joker_melds=True):

    """
    Returns (melds, dp_min, dp_cov) for this exact tile list.
    """
    key = (_hand_key(hand_tiles), allow_joker_melds)
    if key in _DP_CACHE:
        return _DP_CACHE[key]
    meld_masks, meld_jokers_used = generate_melds_for_hand(hand_tiles, allow_joker_melds=allow_joker_melds)

    n = len(hand_tiles)
    dp_min = dp_min_jokers_exact_cover(n, melds)
    dp_cov = dp_max_coverage(n, melds)
    _DP_CACHE[key] = (melds, dp_min, dp_cov)
    return melds, dp_min, dp_cov

def min_jokers_to_cover_mask(hand15, cover_mask):
    _, dp_min, _ = get_hand_dp(hand15)
    val = dp_min[cover_mask]
    return None if val >= INF else val

def near_meld_score(hand):
    """
    Heuristic score for 'almost melds'.
    Does NOT require tiles to be in valid melds.
    Higher = more promising structure.
    """
    from collections import Counter, defaultdict

    non_jokers = [t for t in hand if not t.is_joker()]
    jokers = sum(1 for t in hand if t.is_joker())

    score = 0

    # A) Pairs/triples toward sets (same number)
    by_num = defaultdict(list)
    for t in non_jokers:
        by_num[t.number].append(t)

    for num, tiles in by_num.items():
        # unique colors matter for sets
        uniq_colors = len(set(t.color for t in tiles))
        if uniq_colors == 2:
            score += 6   # a pair is useful
        elif uniq_colors == 3:
            score += 14  # one away from 4-set or strong set base
        elif uniq_colors >= 4:
            score += 20  # already a full 4-set exists (also helps coverage anyway)

    # B) Close runs: count adjacency within each color
    by_color = defaultdict(list)
    for t in non_jokers:
        by_color[t.color].append(t.number)

    for c, nums in by_color.items():
        nums = sorted(set(nums))

        # score adjacent pairs and 2-step gaps (e.g., 4 & 6 suggests 5)
        s = set(nums)
        for x in nums:
            if x + 1 in s:
                score += 4  # adjacent is strong
            if x + 2 in s:
                score += 2  # one-gap is weaker but relevant

        # Big bonus for a 1-gap bridge (e.g. 8 and 10 means 9 completes)
        # This is especially valuable when it connects longer segments.
        def consec_left(x):
            k = 0
            while (x - k) in s:
                k += 1
            return k  # how many consecutive numbers ending at x

        def consec_right(x):
            k = 0
            while (x + k) in s:
                k += 1
            return k  # how many consecutive numbers starting at x

        # Length-weighted bridge: bigger bonus if it connects longer segments
        for m in range(2, 13):  # 2..12 possible middle
            if (m not in s) and (m - 1 in s) and (m + 1 in s):
                left_len = consec_left(m - 1)  # e.g. 7,8 -> left_len=2 at x=8
                right_len = consec_right(m + 1)  # e.g. 10,11 -> right_len=2 at x=10
                score += 6 + 3 * (left_len + right_len)

        # terminal wrap adjacency (12-13) suggests 1 as cap; (13-1) is NOT allowed to continue,
        # but 12-13 still matters because it can end with 1.
        if 12 in s and 13 in s:
            score += 3
        if 13 in s and 1 in s:
            score += 1  # weaker because your wrap is terminal

    # C) Jokers slightly boost near-meld flexibility
    score += jokers * 5

    return score

