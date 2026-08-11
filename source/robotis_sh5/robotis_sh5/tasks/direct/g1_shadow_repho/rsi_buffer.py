"""Direction-split RSI buffer mechanics, as pure tensor functions.

Lifted out of the env so they can be tested without a simulator. The env's fingerprint turned out to
be too noisy to settle anything — two runs of IDENTICAL code moved `col0_max` by 17.6 and
`start_frame_mean` by 28.6 — so the buffer logic is verified here instead, against reference
implementations, where the answer is exact.

Layout everywhere below:

    cache  (D, F, S, X)   D=2 directions, F frames, S slots, X = _STATE_DIM
    occ    (D, F, S)      bool, slot holds something
    cret   (D, F, S)      discounted return, the tiebreak
    clen   (D, F, S)      cross-buffer survival length (-inf = empty)

Direction 0 is FORWARD, 1 is BACKWARD. Frames are in ORIGINAL clip time in every buffer; the caller
mirrors episode time and negates velocities before getting here, so a state written by one direction
is directly readable by the other. That is what makes the cross transfer a plain copy.

Two scores come out of one rollout (RePHO / InterMimic, SupMat Alg 2):
    self  = |t_end - t|     how much further it got            -> its own direction's buffer
    cross = |t - t_start|   how far it had come                -> the OPPOSITE direction's
Which one matters depends on the direction the next episode will leave that state in.
"""

import torch


def group_argmax(key: torch.Tensor, score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per distinct key, the index of its highest-scoring entry.

    Ties resolve to the EARLIEST entry, matching the `argmax` of the Python loop this replaces —
    with a survival-length score exact ties are common, and taking the last instead silently swaps
    which candidate's return and state get stored (227/400 random cases diverged before this was
    fixed). Returns (unique_keys, winner_index_into_the_inputs).
    """
    keys, inv = torch.unique(key, return_inverse=True)
    o1 = torch.argsort(score, descending=True, stable=True)
    o2 = torch.argsort(inv[o1], stable=True)
    idx_sorted, inv_sorted = o1[o2], inv[o1][o2]
    pos = torch.arange(inv_sorted.numel(), device=key.device)
    first = torch.zeros(keys.numel(), dtype=torch.long, device=key.device)
    first.scatter_(0, inv_sorted.flip(0), pos.flip(0))          # reversed scatter: earliest wins
    return keys, idx_sorted[first]


def self_write(cache, occ, cret, dirs, frames, score, ret, state, margin, ret_ratio=1.0):
    """Commit candidates into their OWN direction's buffer. Returns the number written.

    Per (direction, frame) the best candidate is compared against the worst LEARNED slot; slot 0 is
    the retarget reference and its state is never replaced (RePHO seeds it and only writes 1..).
    Acceptance follows RePHO 1870: length decides and the return only has to clear `ret_ratio` times
    the incumbent's, while a clear win on length (`margin`) overrides the return outright — otherwise
    a long-surviving rollout could be blocked forever by an incumbent scored under an older policy.
    An empty slot always takes.

    `ret_ratio` is RePHO's `ratio = max((53250 - curr_epoch)/1000, 0)` (intermimic.py:1868). It starts
    around 0.25 (curr_epoch is initialised to 53001) and anneals to 0, i.e. the return bar is weak
    early and gone later, leaving length alone in charge. A fixed 1.0 — "the return must not be worse"
    — is STRICTER than RePHO ever gets, and stays strict exactly when the incumbent's return is most
    stale.
    """
    D, F, S, _ = cache.shape
    keys, win = group_argmax(dirs * F + frames, score)          # composite key: one pass over both
    kd, kf = keys // F, keys % F
    if S > 1:
        occ_l = occ[kd, kf, 1:]
        sc_l = torch.where(occ_l, cache[kd, kf, 1:, 0], torch.full_like(cache[kd, kf, 1:, 0], -float("inf")))
        k = sc_l.argmin(dim=1) + 1
        beat = torch.where(
            occ_l.all(dim=1),
            ((score[win] > sc_l.min(dim=1).values) & (ret[win] >= cret[kd, kf, k] * ret_ratio))
            | (score[win] > sc_l.min(dim=1).values + margin),
            torch.ones(keys.numel(), dtype=torch.bool, device=cache.device))
    else:
        k = torch.zeros_like(kf)
        beat = score[win] > cache[kd, kf, 0, 0]
    wi = beat.nonzero(as_tuple=True)[0]
    if not wi.numel():
        return 0
    d_w, f_w, k_w, c_w = kd[wi], kf[wi], k[wi], win[wi]
    cache[d_w, f_w, k_w] = state[c_w]
    cache[d_w, f_w, k_w, 0] = score[c_w]
    occ[d_w, f_w, k_w] = True
    cret[d_w, f_w, k_w] = ret[c_w]
    return int(wi.numel())


def cross_write(ccache, clen, cocc, dirs, frames, score, state):
    """Stage candidates into the WRITING direction's cross buffer (SupMat Alg 2, lines 33-38).

    Same fan-out as self_write, but the rule is a bare `L_new > L_min`: the cross buffer carries no
    return, so there is nothing to tiebreak on and no margin.
    """
    F = clen.shape[1]
    keys, win = group_argmax(dirs * F + frames, score)
    kd, kf = keys // F, keys % F
    k = clen[kd, kf].argmin(dim=1)
    beat = score[win] > clen[kd, kf, k]
    wi = beat.nonzero(as_tuple=True)[0]
    if not wi.numel():
        return 0
    d_w, f_w, k_w, c_w = kd[wi], kf[wi], k[wi], win[wi]
    ccache[d_w, f_w, k_w] = state[c_w]
    clen[d_w, f_w, k_w] = score[c_w]
    cocc[d_w, f_w, k_w] = True
    return int(wi.numel())


def cross_import(cache, occ, cret, ccache, clen, cocc, reserved, margin, floor, ratio, penalty):
    """Inter-direction update (SupMat Alg 3, lines 41-52).

    Direction d takes the OPPOSITE direction's best cross entry into a RESERVED slot — RePHO's
    "slot 3 is reserved for inter-direction states", which its released code implements by merging
    into `ref_reward[:, 2:]`. The gate is deliberately stricter than an intra-direction write
    (intermimic.py:1058): a clear margin over the incumbent, an absolute floor, and it must beat this
    direction's OWN best slots by a ratio. The imported score is then docked by `penalty` — a state
    the other direction reached is weaker evidence about this direction than one we reached ourselves.

    Returns the number imported.
    """
    n = 0
    for d in (0, 1):
        src = 1 - d
        L_in, best = clen[src].max(dim=1)                       # (F,) best cross entry per frame
        ok = (cocc[src].any(dim=1)
              & (L_in > cache[d, :, reserved, 0] + margin)
              & (L_in > floor)
              & (L_in > cache[d, :, :reserved, 0].max(dim=1).values * ratio))
        wi = ok.nonzero(as_tuple=True)[0]
        if not wi.numel():
            continue
        cache[d, wi, reserved] = ccache[src, wi, best[wi]]
        cache[d, wi, reserved, 0] = (L_in[wi] - penalty).clamp(min=0.0)
        cret[d, wi, reserved] = 0.0                             # no return travels with it
        occ[d, wi, reserved] = True
        n += int(wi.numel())
    return n


def slot_probs(cache, occ, dirs, frames, floor):
    """(n, S) slot lottery at the given (direction, frame) pairs.

        p_k proportional to (L_k - floor).clamp(min=1) * occupied_k

    RePHO's rule (intermimic.py:1381-1391): the min=1 clamp IS the floor, so every occupied slot
    keeps mass while a long-surviving one dominates in proportion to how much longer it lasted.
    Slot 0's score stays 0, so it sits at the clamp and a learned slot at L=25 outweighs it 19:1.
    A frame with no occupied slot falls back to slot 0 — multinomial on an all-zero row returns an
    out-of-range index and faults the cache gather two lines later.
    """
    u = occ[dirs, frames]
    sc = (cache[dirs, frames, :, 0] - floor).clamp(min=1.0) * u
    dead = sc.sum(1, keepdim=True) <= 0
    if bool(dead.any()):
        z = torch.zeros_like(sc)
        z[:, 0] = 1.0
        sc = torch.where(dead, z, sc)
    return sc / sc.sum(1, keepdim=True).clamp(min=1e-6)
