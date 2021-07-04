"""
Dynamics of multi-quadrotors with slung load.
Reference: Taeyoun Lee, "Geometric Control of Quadrotor UAVs Transporting
a Cable-Suspended Rigid Body," IEEE TCST, 2018.
Note
1. Rotation matrix(dcm) in the paper and code is the one used in Robotics,
which means the dcm should be transposed to use ``rot.angle2dcm`` package.
"""
import numpy as np
from numpy.linalg import inv, pinv
from numpy import pi, cos, sin
from scipy.linalg import block_diag

import fym
from fym import BaseEnv, BaseSystem, Sequential
from fym.utils.rot import sph2cart2

from utils import hat, angle2R
import config


cfg = config.load().dynamics


class Load(BaseEnv):
    def __init__(self):
        super().__init__()

        # Set random states
        states = cfg.load.initStates
        # bounds = cfg.load.initBounds
        # states.pos = np.vstack([np.random.uniform(*pb) for pb in bounds.position])
        # states.R = angle2R(*(np.random.uniform(*ab) for ab in bounds.attitude))

        self.pos = BaseSystem(states.pos)
        self.vel = BaseSystem(states.vel)
        self.R = BaseSystem(states.R)
        self.Omega = BaseSystem(states.Omega)

        # Set physical properties
        self.mass = cfg.load.physicalProperties.mass
        self.J = cfg.load.physicalProperties.J
        self.cg = cfg.load.physicalProperties.cg
        self.size = cfg.load.physicalProperties.size

    def set_dot(self, acc, ang_acc):
        self.pos.dot = self.vel.state
        self.vel.dot = acc
        self.R.dot = self.R.state @ hat(self.Omega.state)
        self.Omega.dot = ang_acc


class Link(BaseEnv):
    def __init__(self, load, uvec, length, avec):
        super().__init__(name="link")
        states = cfg.link.initStates
        self.uvec = BaseSystem(uvec)
        self.omega = BaseSystem(states.omega)

        self.load = load

        self.length = length
        self.anchor = load.size * avec - load.cg

        self.rho = self.anchor
        self.hatrho = hat(self.anchor)

    def set_dot(self, ang_acc):
        self.uvec.dot = hat(self.omega.state).dot(self.uvec.state)
        self.omega.dot = ang_acc


class Quadrotor(BaseEnv):
    def __init__(self, link):
        super().__init__(name="quad")
        self.R = BaseSystem(cfg.quadrotor.initStates.R)
        self.Omega = BaseSystem(cfg.quadrotor.initStates.Omega)

        self.link = link

        # Set physical properties
        self.mass = cfg.quadrotor.physicalProperties.mass
        self.J = cfg.quadrotor.physicalProperties.J

        self.invJ = inv(self.J)

        self.I = np.eye(3)
        self.e3 = np.vstack((0., 0., 1.))

    @property
    def pos(self):
        link = self.link
        load = link.load
        pos = (load.pos.state
               + load.R.state @ link.anchor
               - link.uvec.state * link.length)
        return pos

    def set_dot(self, moment):
        Omega = self.Omega.state
        self.R.dot = self.R.state @ hat(Omega)
        self.Omega.dot = self.invJ @ (moment - hat(Omega) @ self.J.dot(Omega))

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, f):
        self._f = f
        self._u = - f * self.R.state @ self.e3
        q = self.link.uvec.state
        self._qqT = q @ q.T
        self._upar = self._qqT @ self._u
        self._uper = (self.I - self._qqT) @ self._u

    @property
    def u(self):
        return self._u

    @property
    def qqT(self):
        return self._qqT

    @property
    def upar(self):
        return self._upar

    @property
    def uper(self):
        return self._uper

#     @property
#     def tension(self):
#         q = self.link.uvec.state
#         omega = self.link.omega.state
#         tension = self.upar - self.mass * q * (
#             self.link.length * omega.T @ omega + q.T @ a)
#         return tension


class MultiQuadSlungLoad(fym.BaseEnv):
    def __init__(self):
        super().__init__()
        N = cfg.MQSL.quads.num

        # Load system
        self.load = Load()

        if cfg.MQSL.links.autoInit:
            params = cfg.MQSL.links.autoInitParams

            lengths = [params.length] * N

            avecs = [np.vstack((cos(i*2*pi/N), sin(i*2*pi/N), 0))
                     for i in range(N)]

            if params.azimuth == "equal":
                azs = [pi + i*2*pi/N for i in range(N)]
            else:
                NotImplementedError

            if not isinstance(params.elevation, (str, list)):
                els = [params.elevation for i in range(N)]
            else:
                NotImplementedError
        else:
            bounds = cfg.link.initBounds
            az, el = (np.random.uniform(*ub) for ub in bounds.uvec)

        uvecs = [sph2cart2(1, az, el) for az, el in zip(azs, els)]

        self.links = fym.Sequential(
            *(Link(self.load, uvec, length, avec)
              for uvec, length, avec in zip(uvecs, lengths, avecs)))

        for name, link in self.links.systems_dict.items():
            fym.parser.update(cfg.MQSL.links, {
                name: {
                    "length": link.length,
                    "anchor": link.anchor,
                }
            })

        # Quadrotor systems
        self.quads = fym.Sequential(
            *(Quadrotor(link) for link in self.links.systems))

        self.N = N
        self.g = cfg.gravity
        self.e3 = np.vstack((0., 0., 1.))
        self.I = np.eye(3)

    def set_dot(self, t, fs, Ms):
        R0 = self.load.R.state
        Omega0 = self.load.Omega.state

        J0 = self.load.J
        hatOmega0 = hat(Omega0)
        hatOmega02 = hatOmega0 @ hatOmega0
        e3 = self.e3

        LHS = block_diag(self.load.mass * self.I, J0)
        RHS = np.zeros((6, 1))
        RHS[3:6, :] = - hatOmega0 @ J0 @ Omega0

        for i, quad in enumerate(self.quads.systems):
            quad.f = fs[i]
            quad.set_dot(Ms[i])

            m = quad.mass
            qqT = quad.qqT
            link = quad.link
            omega = link.omega.state
            hr = link.hatrho

            LHS[:3, :3] += m * qqT
            LHS[:3, 3:6] += - m * qqT @ R0 @ hr
            LHS[3:6, :3] += m * hr @ R0.T @ qqT
            LHS[3:6, 3:6] += - m * hr @ R0.T @ qqT @ R0 * hr

            rhs1 = (quad.upar
                    - m * link.length * omega.T @ omega * link.uvec.state
                    - m * qqT @ R0 @ hatOmega02 @ link.rho)
            RHS[:3, :] += rhs1
            RHS[3:6, :] += hr @ R0.T @ rhs1

        load_accs = inv(LHS) @ RHS
        x0ddot, Omega0dot = load_accs[:3] + self.g * e3, load_accs[3:6]

        self.load.set_dot(x0ddot, Omega0dot)

        for quad in self.quads.systems:
            link = quad.link
            hq = hat(link.uvec.state)

            a = load_accs[:3] + R0 @ (
                - link.hatrho @ Omega0dot + hatOmega02 @ link.rho)
            Omegadot = hq @ (a - quad.uper / quad.mass) / link.length

            link.set_dot(Omegadot)

#             q = link.uvec.state
#             omega = link.omega.state
#             tension = quad.upar - quad.mass * q * (
#                 link.length * omega.T @ omega + q.T @ a)

#             if tension.T @ (-q) >= 0:
#                 breakpoint()

#         if t > 3.62:
#             breakpoint()

#         if tension.T @ (-q) < 0:
#             breakpoint()

    def collision_check(self):
        pass


if __name__ == "__main__":
    class Env(BaseEnv):
        def __init__(self):
            super().__init__(dt=0.01, max_t=10)
            self.plant = MultiQuadSlungLoad()
            self.N = cfg.MQSL.quads.num
            self.logger = fym.Logger("data.h5")

        def step(self):
            *_, done = self.update()
            return done

        def set_dot(self, t):
            fs = [0] * self.N
            Ms = [np.zeros((3, 1))] * self.N
            self.plant.set_dot(t, fs, Ms)

            return dict(t=t, **self.plant.observe_dict())

    env = Env()
    while True:
        env.render()
        done = env.step()
        if done:
            break

    env.close()
