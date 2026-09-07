import math

AXES = {
    "L": {
        "usy": (0.0, 0.5, 0.866025),
        "shx": (1.0, 0.0, 0.0),
        "ely": (0.0, 1.0, 0.0),
        "elx": (1.0, 0.0, 0.0),
    },
    "R": {
        "usy": (0.0, 0.5, -0.866025),
        "shx": (1.0, 0.0, 0.0),
        "ely": (0.0, 1.0, 0.0),
        "elx": (1.0, 0.0, 0.0),
    },
}

REST_UPPER = {
    "L": (0.0, 0.381, 0.049),
    "R": (0.0, -0.381, 0.049),
}

REST_FORE = {
    "L": (0.0, 0.188, -0.013),
    "R": (0.0, -0.188, -0.013),
}

LIMITS = {
    "L": {
        "usy": (-1.9635, 1.9635),
        "shx": (-1.39626, 1.74533),
        "ely": (0.0, 3.14159),
        "elx": (0.0, 2.35619),
    },
    "R": {
        "usy": (-1.9635, 1.9635),
        "shx": (-1.74533, 1.39626),
        "ely": (0.0, 3.14159),
        "elx": (-2.35619, 0.0),
    },
}

STEP = 1e-4
ITERATIONS = 24
DAMPING = 0.35


def normalise(vector):
    length = math.sqrt(sum(component * component for component in vector))
    if length < 1e-9:
        return (0.0, 0.0, 0.0)
    return tuple(component / length for component in vector)


def rotate(vector, axis, angle):
    axis = normalise(axis)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dot = sum(a * b for a, b in zip(axis, vector))
    cross = (
        axis[1] * vector[2] - axis[2] * vector[1],
        axis[2] * vector[0] - axis[0] * vector[2],
        axis[0] * vector[1] - axis[1] * vector[0],
    )
    return tuple(
        vector[i] * cos_a + cross[i] * sin_a + axis[i] * dot * (1.0 - cos_a)
        for i in range(3)
    )


def clamp(value, bounds):
    return max(bounds[0], min(bounds[1], value))


def upper_direction(side, usy, shx):
    vector = REST_UPPER[side]
    vector = rotate(vector, AXES[side]["shx"], shx)
    vector = rotate(vector, AXES[side]["usy"], usy)
    return normalise(vector)


def fore_direction(side, usy, shx, ely, elx):
    vector = REST_FORE[side]
    vector = rotate(vector, AXES[side]["elx"], elx)
    vector = rotate(vector, AXES[side]["ely"], ely)
    vector = rotate(vector, AXES[side]["shx"], shx)
    vector = rotate(vector, AXES[side]["usy"], usy)
    return normalise(vector)


def _error(current, target):
    return sum((c - t) ** 2 for c, t in zip(current, target))


SEED_GRID = 3


def _seeds(side, seed):
    limits = LIMITS[side]
    yield seed
    for i in range(SEED_GRID):
        usy = limits["usy"][0] + (limits["usy"][1] - limits["usy"][0]) * i / (SEED_GRID - 1)
        for j in range(SEED_GRID):
            shx = limits["shx"][0] + (limits["shx"][1] - limits["shx"][0]) * j / (SEED_GRID - 1)
            yield (usy, shx)


def solve_upper(side, target, seed=(0.0, 0.0)):
    target = normalise(target)
    if target == (0.0, 0.0, 0.0):
        return seed, 1.0

    best = None
    best_error = float("inf")
    for candidate in _seeds(side, seed):
        result, error = _descend_upper(side, target, candidate)
        if error < best_error:
            best, best_error = result, error
            if best_error < 2e-3:
                break
    return best, best_error


def _descend_upper(side, target, seed):
    usy, shx = seed
    limits = LIMITS[side]
    error = _error(upper_direction(side, usy, shx), target)

    for _ in range(ITERATIONS):
        base = upper_direction(side, usy, shx)
        d_usy = upper_direction(side, usy + STEP, shx)
        d_shx = upper_direction(side, usy, shx + STEP)

        grad_usy = sum(2.0 * (base[i] - target[i]) * (d_usy[i] - base[i]) / STEP
                       for i in range(3))
        grad_shx = sum(2.0 * (base[i] - target[i]) * (d_shx[i] - base[i]) / STEP
                       for i in range(3))

        step = DAMPING
        improved = False
        for _ in range(6):
            candidate_usy = clamp(usy - step * grad_usy, limits["usy"])
            candidate_shx = clamp(shx - step * grad_shx, limits["shx"])
            candidate_error = _error(
                upper_direction(side, candidate_usy, candidate_shx), target
            )
            if candidate_error < error:
                usy, shx, error = candidate_usy, candidate_shx, candidate_error
                improved = True
                break
            step *= 0.5
        if not improved or error < 1e-8:
            break

    return (usy, shx), math.sqrt(max(0.0, error))


COARSE_STEPS = 12
REFINE_ROUNDS = 8


WARM_TOLERANCE = 4.0e-3


def _refine_fore(side, target, usy, shx, ely, elx, error, ely_span, elx_span):
    limits = LIMITS[side]
    for _ in range(REFINE_ROUNDS + 3):
        improved = False
        for d_ely in (-ely_span, 0.0, ely_span):
            for d_elx in (-elx_span, 0.0, elx_span):
                if d_ely == 0.0 and d_elx == 0.0:
                    continue
                candidate_ely = clamp(ely + d_ely, limits["ely"])
                candidate_elx = clamp(elx + d_elx, limits["elx"])
                candidate = _error(
                    fore_direction(side, usy, shx, candidate_ely, candidate_elx), target
                )
                if candidate < error:
                    error, ely, elx = candidate, candidate_ely, candidate_elx
                    improved = True
        if not improved:
            ely_span *= 0.5
            elx_span *= 0.5
    return ely, elx, error


def solve_fore(side, target, usy, shx, seed=(0.0, 0.0), allow_grid=True):
    target = normalise(target)
    if target == (0.0, 0.0, 0.0):
        return seed, 1.0

    limits = LIMITS[side]
    ely_low, ely_high = limits["ely"]
    elx_low, elx_high = limits["elx"]

    best_ely, best_elx = seed
    best_error = _error(fore_direction(side, usy, shx, best_ely, best_elx), target)

    warm = _refine_fore(side, target, usy, shx, best_ely, best_elx, best_error,
                        (ely_high - ely_low) * 0.12, (elx_high - elx_low) * 0.12)
    best_ely, best_elx, best_error = warm
    if best_error <= WARM_TOLERANCE or not allow_grid:
        return (best_ely, best_elx), math.sqrt(max(0.0, best_error))

    for i in range(COARSE_STEPS + 1):
        ely = ely_low + (ely_high - ely_low) * i / COARSE_STEPS
        for j in range(COARSE_STEPS + 1):
            elx = elx_low + (elx_high - elx_low) * j / COARSE_STEPS
            error = _error(fore_direction(side, usy, shx, ely, elx), target)
            if error < best_error:
                best_error, best_ely, best_elx = error, ely, elx

    ely_span = (ely_high - ely_low) / COARSE_STEPS
    elx_span = (elx_high - elx_low) / COARSE_STEPS
    for _ in range(REFINE_ROUNDS):
        ely_span *= 0.5
        elx_span *= 0.5
        improved = False
        for d_ely in (-ely_span, 0.0, ely_span):
            for d_elx in (-elx_span, 0.0, elx_span):
                if d_ely == 0.0 and d_elx == 0.0:
                    continue
                ely = clamp(best_ely + d_ely, limits["ely"])
                elx = clamp(best_elx + d_elx, limits["elx"])
                error = _error(fore_direction(side, usy, shx, ely, elx), target)
                if error < best_error:
                    best_error, best_ely, best_elx = error, ely, elx
                    improved = True
        if not improved and best_error < 1e-8:
            break

    return (best_ely, best_elx), math.sqrt(max(0.0, best_error))


def solve_arm(side, upper_target, fore_target, seed=None):
    seed = seed or {}
    (usy, shx), upper_error = solve_upper(
        side, upper_target, seed.get("upper", (0.0, 0.0))
    )
    if fore_target is None:
        return {"usy": usy, "shx": shx}, upper_error, 1.0
    (ely, elx), fore_error = solve_fore(
        side, fore_target, usy, shx, seed.get("fore", (0.0, 0.0))
    )
    return {"usy": usy, "shx": shx, "ely": ely, "elx": elx}, upper_error, fore_error
