"""Monte Carlo simulation of particle trajectories.

Ports of TrackArt's ``sim/`` MATLAB module:

* ``sim_simulateStates``       - two-state Markov switching
* ``sim_simulateDomains``      - random circular domain map
* ``sim_simulateCoordinates``  - 1particle1D / Switch / Domains trajectories
* ``sim_reduceCoordinates``    - decimate micro-step coordinates
* ``sim_fracprob``             - theoretical fractions and switch probabilities
* ``sim_avgGrpSize``           - average group size helper

The output ``coordinates`` array follows the TrackArt convention::

    col 1  dataset id (always 1 for simulated data)
    col 2  1
    col 3  track number
    col 4  frame number
    col 5  x [um]
    col 6  y [um]

(an extra 7th column, holding the diffusion state, is appended in Domains
mode so the actual fraction can be computed afterwards).
"""

from __future__ import annotations

import numpy as np


def sim_fracprob(k21, k12, tau):
    """Theoretical fractions F1, F2 and switch probabilities p21, p12."""
    from .util import roundsd

    F1 = roundsd(k21 / (k12 + k21), 2)
    F2 = roundsd(k12 / (k12 + k21), 2)
    p12 = roundsd(k12 * tau, 2)
    p21 = roundsd(k21 * tau, 2)
    return F1, F2, p21, p12


def sim_simulateStates(N, k21, k12, tau):
    """Simulate a two-state Markov chain of length N.

    state 0 = population 1, state 1 = population 2.
    Returns a 1-D int array of length N (plus a stats dict).
    """
    p12 = k12 * tau  # probability to switch 1 -> 2
    p21 = k21 * tau  # probability to switch 2 -> 1

    states = np.zeros(N, dtype=int)
    if k12 + k21 > 0:
        states[0] = int(np.random.binomial(1, k12 / (k12 + k21)))
    else:
        states[0] = 0

    for i in range(1, N):
        if states[i - 1] == 0:
            states[i] = int(np.random.binomial(1, p12))
        else:
            states[i] = 1 - int(np.random.binomial(1, p21))

    stats = {
        "p12": p12,
        "p21": p21,
        "k12": k12,
        "k21": k21,
        "F1": k21 / (k12 + k21) if k12 + k21 > 0 else 0.0,
        "F1real": np.sum(states == 0) / N,
        "F2": k12 / (k12 + k21) if k12 + k21 > 0 else 0.0,
        "F2real": np.sum(states == 1) / N,
    }
    return states, stats


def sim_simulateDomains(sz, diameter, c, variation):
    """Simulate a square binary map of overlapping circular domains.

    sz        - image size in pixels
    diameter  - domain diameter in pixels (already scaled to image)
    c         - crowding parameter / concentration (0..1)
    variation - relative size variation of the domains

    Returns a 2-D int array of 0/1 (1 = domain).
    """
    sz = int(sz)
    rad = diameter / 2.0
    domains = np.zeros((sz, sz), dtype=np.uint8)

    yy, xx = np.mgrid[0:sz, 0:sz]  # y first index like MATLAB's find + domains(x,y)

    c_sim = 0.0
    while c > c_sim:
        xc = int(np.ceil(np.random.uniform(0, sz))) - 1
        yc = int(np.ceil(np.random.uniform(0, sz))) - 1
        r = rad + rad * np.random.randn() * variation
        r2 = r * r
        d = (xx - xc) ** 2 + (yy - yc) ** 2 <= r2
        domains[d] = 1
        c_sim = np.sum(domains) / float(sz * sz)
    return domains


def sim_reduceCoordinates(coordinates, factor):
    """Keep every ``factor``-th frame and renumber frames 1..k per track."""
    coordinates = np.asarray(coordinates, dtype=float)
    frames = coordinates[:, 3]  # frame number (4th column)
    x = np.remainder(frames - 1, factor)
    red = coordinates[x == 0, :]
    if len(red) == 0:
        return red
    k = len(np.unique(red[:, 3]))
    for tr in np.unique(red[:, 0]):
        red[red[:, 0] == tr, 3] = np.arange(1, k + 1)
    return red


def _simulate_1particle1D(particle_count, N, tau, D, f, random_start, sim_range):
    """1 particle 1D: each particle diffuses with one fixed D."""
    k = np.sqrt(D * 2 * tau)
    k1 = np.sqrt(D[0] * 2 * tau)
    k2 = np.sqrt(D[1] * 2 * tau) if len(D) > 1 and D[1] != 0 else None

    if k2 is None:
        f = 1.0

    coords = np.zeros((N, 2))
    tracks = []
    n1 = int(round(particle_count * f))
    for i in range(particle_count):
        if i < n1:
            kk = k1
        else:
            kk = k2 if k2 is not None else k1
        x = np.cumsum(kk * np.random.randn(N))
        y = np.cumsum(kk * np.random.randn(N))
        if random_start:
            x = x + np.random.uniform(-sim_range / 4, sim_range / 4)
            y = y + np.random.uniform(-sim_range / 4, sim_range / 4)
        tracks.append((x, y))

    actual_f1 = n1 / particle_count
    return tracks, actual_f1


def _simulate_switch(particle_count, N, tau, D, k21, k12, random_start, sim_range):
    """Switch: every particle switches between two states."""
    k1 = np.sqrt(D[0] * 2 * tau) if D[0] != 0 else 0.0
    k2 = np.sqrt(D[1] * 2 * tau) if len(D) > 1 and D[1] != 0 else None

    all_states = []
    tracks = []
    for i in range(particle_count):
        dx = np.random.randn(N)
        dy = np.random.randn(N)
        if k2 is not None:
            states, _ = sim_simulateStates(N, k21, k12, tau)
            all_states.append(states)
            dx[states == 0] = dx[states == 0] * k1
            dy[states == 0] = dy[states == 0] * k1
            dx[states == 1] = dx[states == 1] * k2
            dy[states == 1] = dy[states == 1] * k2
        else:
            dx = dx * k1
            dy = dy * k1
            all_states.append(np.zeros(N, dtype=int))
        x = np.cumsum(dx)
        y = np.cumsum(dy)
        if random_start:
            x = x + np.random.uniform(-sim_range / 4, sim_range / 4)
            y = y + np.random.uniform(-sim_range / 4, sim_range / 4)
        tracks.append((x, y))

    if all_states and k2 is not None:
        states_all = np.concatenate(all_states)
        actual_f1 = np.sum(states_all == 0) / states_all.size
    else:
        actual_f1 = 1.0
    return tracks, actual_f1


def _simulate_domains(particle_count, N, tau, D, k21, k12, domains, sim_range):
    """Domains: diffusion on a pixel grid with a binary domain map."""
    domains_res = domains.shape[0]
    k1 = np.sqrt(D[0] * 2 * tau) * (domains_res / sim_range)
    k2 = np.sqrt(D[1] * 2 * tau) * (domains_res / sim_range)

    p12 = k12 * tau
    p21 = k21 * tau

    tracks = []
    all_states = []
    # If neither state can be entered (all k's or D's zero) no seed position
    # is ever accepted - match the original, but bail out with a warning
    # instead of looping forever.
    can_seed_inside = D[1] != 0 and k12 != 0
    can_seed_outside = D[0] != 0 and k21 != 0

    for i in range(particle_count):
        # seed the particle (not on the border)
        seed_attempts = 0
        while True:
            seed_attempts += 1
            new_x = int(round(np.random.uniform(0.25 * domains_res, 0.75 * domains_res)))
            new_y = int(round(np.random.uniform(0.25 * domains_res, 0.75 * domains_res)))
            # MATLAB: round() can land on 0 or domains_res -> clamp to valid index
            new_x = min(max(new_x, 0), domains_res - 1)
            new_y = min(max(new_y, 0), domains_res - 1)
            if domains[new_y, new_x]:
                if can_seed_inside:
                    state = 2
                    break
            else:
                if can_seed_outside:
                    state = 1
                    break
            if seed_attempts >= 1_000_000:
                raise ValueError(
                    "Cannot seed particle: with the given D and rate constants "
                    "no valid starting state exists (e.g. k21 = k12 = 0).")

        xs = np.zeros(N, dtype=float)
        ys = np.zeros(N, dtype=float)
        states = np.zeros(N, dtype=int)
        xs[0] = new_x
        ys[0] = new_y
        states[0] = state

        max_retries = 2_000_000  # guard against pathological trap loops
        for j in range(N - 1):
            x = int(xs[j])
            y = int(ys[j])
            if domains[y, x]:  # inside domain -> k2
                attempts = 0
                while True:
                    attempts += 1
                    new_x = int(x + round(float(np.random.randn() * k2)))
                    new_y = int(y + round(float(np.random.randn() * k2)))
                    if 0 < new_x < domains_res and 0 < new_y < domains_res:
                        if domains[new_y, new_x] == 0:
                            if np.random.binomial(1, p21) and D[0] != 0:
                                states[j + 1] = 1
                                break
                        else:
                            states[j + 1] = 2
                            break
                    if attempts >= max_retries:
                        new_x, new_y = x, y
                        states[j + 1] = 2
                        break
            else:  # outside domain -> k1
                attempts = 0
                while True:
                    attempts += 1
                    new_x = int(x + round(float(np.random.randn() * k1)))
                    new_y = int(y + round(float(np.random.randn() * k1)))
                    if 0 < new_x < domains_res and 0 < new_y < domains_res:
                        if domains[new_y, new_x] == 1:
                            if np.random.binomial(1, p12) and D[1] != 0:
                                states[j + 1] = 2
                                break
                        else:
                            states[j + 1] = 1
                            break
                    if attempts >= max_retries:
                        new_x, new_y = x, y
                        states[j + 1] = 1
                        break
            xs[j + 1] = new_x
            ys[j + 1] = new_y

        # convert from pixels to um
        tracks.append((xs * (sim_range / domains_res),
                       ys * (sim_range / domains_res)))
        all_states.append(states)

    if all_states:
        states_all = np.concatenate(all_states)
        actual_f1 = np.sum(states_all == 1) / np.sum(states_all != 0)
    else:
        actual_f1 = 1.0
    return tracks, actual_f1


def sim_simulateCoordinates(particle_count, N, tau, D, f, k21, k12, sig0,
                            sim_range, random_start, method, domains=None,
                            keep_states=False):
    """Simulate particle trajectories.

    Parameters
    ----------
    particle_count : int
    N : int
        number of (micro) steps per particle
    tau : float
        time per step in seconds
    D : array_like of 2 diffusion coefficients [um^2/s]
    f : float
        fraction of population 1 (only used by 1particle1D)
    k21, k12 : float
        switching rates (1/s) - already scaled to the micro-step time by the
        caller
    sig0 : float
        static localisation uncertainty [um]
    sim_range : float
        simulation range [um]
    random_start : bool
    method : str
        one of '1particle1D', 'Switch', 'Domains'
    domains : 2-D 0/1 array (required for Domains mode)

    Returns
    -------
    coordinates : array (N*particle_count, 7) with the TrackArt layout and an
        extra 7th column holding the state in Domains/Switch mode (0 otherwise)
    stats : dict with ``actualF1``
    """
    particle_count = int(particle_count)
    N = int(N)
    D = np.asarray(D, dtype=float).ravel()

    sim_coordinates = np.zeros((N * particle_count, 6))
    sim_coordinates[:, 1] = 1  # dataset id
    # Column 1 is the track (dataset) id: 1p1D writes the track number there,
    # Switch/Domains increment a running counter (which equals the track
    # number since tracks are generated in order).  Column 3 (index 2) holds
    # the track number, column 4 (index 3) the frame number.

    if method == "1particle1D":
        tracks, actual_f1 = _simulate_1particle1D(
            particle_count, N, tau, D, f, random_start, sim_range)
        for i, (x, y) in enumerate(tracks, start=1):
            start = (i - 1) * N
            sim_coordinates[start:start + N, 0] = i
            sim_coordinates[start:start + N, 2] = i
            sim_coordinates[start:start + N, 3] = np.arange(1, N + 1)
            sim_coordinates[start:start + N, 4] = x
            sim_coordinates[start:start + N, 5] = y

    elif method == "Switch":
        tracks, actual_f1 = _simulate_switch(
            particle_count, N, tau, D, k21, k12, random_start, sim_range)
        free_track = 0
        for i, (x, y) in enumerate(tracks, start=1):
            track_length = len(x)
            sim_coordinates[free_track:free_track + track_length, 2] = i
            sim_coordinates[free_track:free_track + track_length, 0] = (
                int(np.max(sim_coordinates[:, 0])) + 1)
            sim_coordinates[free_track:free_track + track_length, 3] = np.arange(1, track_length + 1)
            sim_coordinates[free_track:free_track + track_length, 4] = x
            sim_coordinates[free_track:free_track + track_length, 5] = y
            free_track += track_length

    elif method == "Domains":
        if domains is None:
            raise ValueError("Domains mode requires a domain map")
        tracks, actual_f1 = _simulate_domains(
            particle_count, N, tau, D, k21, k12, domains, sim_range)
        free_track = 0
        state_row = 0
        for i, (x, y) in enumerate(tracks, start=1):
            track_length = len(x)
            sim_coordinates[free_track:free_track + track_length, 2] = i
            sim_coordinates[free_track:free_track + track_length, 0] = (
                int(np.max(sim_coordinates[:, 0])) + 1)
            sim_coordinates[free_track:free_track + track_length, 3] = np.arange(1, track_length + 1)
            sim_coordinates[free_track:free_track + track_length, 4] = x
            sim_coordinates[free_track:free_track + track_length, 5] = y
            free_track += track_length
            state_row += track_length

    else:
        raise ValueError(f"Unknown simulation method: {method}")

    # add localisation error
    if sig0 > 0:
        sim_coordinates[:, 4:6] = sim_coordinates[:, 4:6] + sig0 * np.random.randn(
            sim_coordinates.shape[0], 2)

    stats = {"actualF1": actual_f1}
    return sim_coordinates, stats


def sim_avgGrpSize(s, g):
    """Average size of groups of value g in a sequence s (sim_avgGrpSize.m)."""
    s = np.asarray(s)
    not_g = g + 1
    s_prev = np.concatenate([[not_g], s[:-1]])
    s_next = np.concatenate([s[1:], [not_g]])
    idx_first = np.where((s_prev != g) & (s == g))[0]
    idx_last = np.where((s_next != g) & (s == g))[0]
    if len(idx_first) == 0:
        return np.nan
    return float(np.mean(idx_last - idx_first + 1))
