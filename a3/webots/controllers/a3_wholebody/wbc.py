import os

import numpy as np
import osqp
import scipy.sparse as sp

ARM_JOINTS = ("LArmUsy", "LArmShx", "LArmEly", "LArmElx",
              "RArmUsy", "RArmShx", "RArmEly", "RArmElx")
TORSO_JOINTS = ("BackLbz", "BackMby", "BackUbx")
HEAD_JOINTS = ("NeckAy",)
LEG_JOINTS = ("LLegUhz", "LLegMhx", "LLegLhy", "LLegKny", "LLegUay", "LLegLax",
              "RLegUhz", "RLegMhx", "RLegLhy", "RLegKny", "RLegUay", "RLegLax")

W_COM = float(os.environ.get("A3_W_COM", "120.0"))
W_ARM = float(os.environ.get("A3_W_ARM", "25.0"))
W_TORSO = float(os.environ.get("A3_W_TORSO", "25.0"))
W_LEG = float(os.environ.get("A3_W_LEG", "3.5"))
W_POSTURE = 0.05
REGULARISATION = 0.02

COM_GAIN = float(os.environ.get("A3_COM_GAIN", "6.0"))
# Free zone: moving the arms *must* move the centre of mass. Pulling it
# back to a fixed point fights the imitation for no benefit, so the CoM
# term only engages once the error leaves this band.
COM_DEADBAND = float(os.environ.get("A3_COM_DEADBAND", "0.045"))
COM_MARGIN = 0.030
USE_COM_CONSTRAINT = os.environ.get("A3_COM_CONSTRAINT", "1") == "1"


class WholeBodyQP:
    """One quadratic program per tick.

    Tracking the human is the objective; keeping the centre of mass inside the
    shrunken support polygon is a hard constraint. When the two conflict the
    solver gives up imitation fidelity, never balance.
    """

    def __init__(self, model, dt, rate_limit=1.8):
        self.model = model
        self.dt = dt
        self.rate_limit = rate_limit
        self.names = model.names
        self.n = len(self.names)
        self.index = model.index

        self._solver = None
        self._shape = None
        self.status = "init"
        self.com_saturated = False

    def _rows(self, joints):
        return [self.index[name] for name in joints if name in self.index]

    def solve(self, angles, com_error, com_setpoint_error, polygon, targets,
              posture, com_jacobian, free_joints=()):
        """angles: measured joint angles. com_error: current CoM minus setpoint,
        expressed in the body frame (x forward, y left). polygon: (x_lo, x_hi,
        y_lo, y_hi) of the support region relative to the same setpoint."""

        blocks = []
        rhs = []
        weights = []

        jac = com_jacobian[:2, :]
        excess = np.zeros(2)
        for axis in range(2):
            value = float(com_error[axis])
            if value > COM_DEADBAND:
                excess[axis] = value - COM_DEADBAND
            elif value < -COM_DEADBAND:
                excess[axis] = value + COM_DEADBAND
        desired = -COM_GAIN * excess * self.dt
        blocks.append(jac)
        rhs.append(desired)
        weights.append(W_COM)

        for joints, weight in ((ARM_JOINTS, W_ARM), (TORSO_JOINTS, W_TORSO),
                               (HEAD_JOINTS, W_TORSO), (LEG_JOINTS, W_LEG)):
            rows = []
            values = []
            for name in joints:
                if name not in targets or name not in self.index:
                    continue
                row = np.zeros(self.n)
                row[self.index[name]] = 1.0
                rows.append(row)
                values.append(targets[name] - angles.get(name, 0.0))
            if rows:
                blocks.append(np.asarray(rows))
                rhs.append(np.asarray(values))
                weights.append(weight)

        rows = []
        values = []
        for name, value in posture.items():
            if name not in self.index:
                continue
            row = np.zeros(self.n)
            row[self.index[name]] = 1.0
            rows.append(row)
            values.append(value - angles.get(name, 0.0))
        if rows:
            blocks.append(np.asarray(rows))
            rhs.append(np.asarray(values))
            weights.append(W_POSTURE)

        hessian = REGULARISATION * np.eye(self.n)
        gradient = np.zeros(self.n)
        for block, target, weight in zip(blocks, rhs, weights):
            hessian += weight * (block.T @ block)
            gradient -= weight * (block.T @ target)

        rate = self.rate_limit * self.dt
        lower = np.full(self.n, -rate)
        upper = np.full(self.n, rate)
        # Joints driven straight from the reference are pinned to the value the
        # caller already applied, so the balance solution cannot fight them.
        for name in free_joints:
            i = self.index.get(name)
            if i is None:
                continue
            step = targets.get(name, angles.get(name, 0.0)) - angles.get(name, 0.0)
            step = max(-rate, min(rate, step))
            lower[i] = upper[i] = step
        for i, name in enumerate(self.names):
            low, high = self.model.limits[name]
            current = angles.get(name, 0.0)
            lower[i] = max(lower[i], low - current)
            upper[i] = min(upper[i], high - current)
            if lower[i] > upper[i]:
                lower[i] = upper[i] = 0.0

        # The centre of mass cannot be forced back inside the polygon within a
        # single tick -- the rate limit allows only ~0.014 rad per joint. Asking
        # for it makes the program infeasible. Instead the constraint says: never
        # move the CoM further outside than it already is. Recovery comes from
        # the weighted tracking term, which is always solvable.
        x_lo, x_hi, y_lo, y_hi = polygon
        com_body = np.asarray(com_setpoint_error[:2]) + np.asarray(com_error[:2])
        bounds_lo = np.array([x_lo + COM_MARGIN, y_lo + COM_MARGIN])
        bounds_hi = np.array([x_hi - COM_MARGIN, y_hi - COM_MARGIN])

        step_cap = self.rate_limit * self.dt * 0.6
        com_lower = np.empty(2)
        com_upper = np.empty(2)
        for axis in range(2):
            low = bounds_lo[axis] - com_body[axis]
            high = bounds_hi[axis] - com_body[axis]
            if low > high:
                self.com_saturated = True
                centre = 0.5 * (bounds_lo[axis] + bounds_hi[axis]) - com_body[axis]
                low, high = min(0.0, centre), max(0.0, centre)
            com_lower[axis] = max(low, -step_cap)
            com_upper[axis] = min(high, step_cap)
            if com_lower[axis] > com_upper[axis]:
                com_lower[axis] = com_upper[axis] = 0.0

        if USE_COM_CONSTRAINT:
            constraint = sp.csc_matrix(np.vstack([np.eye(self.n), jac]))
            constraint_lo = np.concatenate([lower, com_lower])
            constraint_hi = np.concatenate([upper, com_upper])
        else:
            constraint = sp.csc_matrix(np.eye(self.n))
            constraint_lo = lower
            constraint_hi = upper

        hessian = sp.csc_matrix((hessian + hessian.T) / 2.0)
        shape = (constraint.shape, hessian.nnz, constraint.nnz)

        if self._solver is None or self._shape != shape:
            self._solver = osqp.OSQP()
            self._solver.setup(hessian, gradient, constraint,
                               constraint_lo, constraint_hi,
                               verbose=False, eps_abs=1e-4, eps_rel=1e-4,
                               max_iter=250, polish=False)
            self._shape = shape
        else:
            self._solver.update(Px=hessian.data, q=gradient,
                                Ax=constraint.data,
                                l=constraint_lo, u=constraint_hi)

        result = self._solver.solve()
        self.status = str(result.info.status)
        if result.x is None or not np.all(np.isfinite(result.x)):
            return np.zeros(self.n)
        return np.clip(result.x, lower, upper)
