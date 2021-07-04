import numpy as np
from numpy.linalg import inv
from numpy import cos, sin, tan, pi

import fym
from fym.utils.rot import sph2cart2

from dynamics import Load, Link, Quadrotor
import dynamics
from utils import hat, R2angle
import config

cfg = config.load().env


class MultiQuadSlungLoad(fym.BaseEnv):
    def __init__(self):
        super().__init__(**fym.parser.decode(cfg.kwargs))
        N = cfg.quads.num

        # Load system
        self.load = Load()

        # Link systems
        if cfg.links.type == "identical":
            lengths = [cfg.links.singleLength] * N
        else:
            raise NotImplementedError

        if cfg.links.anchorPlacement == "equal":
            avecs = [np.vstack((cos(i*2*pi/N), sin(i*2*pi/N), 0))
                     for i in range(N)]
        else:
            raise NotImplementedError

        self.links = fym.Sequential(
            *(Link(self.load, length, avec)
              for length, avec in zip(lengths, avecs)))

        # Quadrotor systems
        self.quads = fym.Sequential(
            *(Quadrotor(link) for link in self.links.systems))

        self.N = N
        self.g = cfg.gravity
        self.e3 = np.vstack((0., 0., 1.))
        self.I = np.eye(3)

        self.Ke = cfg.controller.Ke
        self.Ks = cfg.controller.Ks
        self.cb = cfg.controller.chatteringBound
        self.mu = cfg.controller.maxUncertainty
        self.iscollision = False

#     def reset(self, des, fixed_init=False):
#         load_pos_des, load_att_des, _ = des
#         super().reset()
#         if fixed_init:
#             self.load.pos.state = self.cfg.load.pos_init
#             self.load.dcm.state = self.cfg.load.dcm_init

#             uvec_init = np.vstack((0., 0., -1.))
#             for link in self.links.systems:
#                 link.uvec.state = uvec_init
#         obs = self.observe(load_pos_des, load_att_des)
#         return obs

    def set_dot(self, t, quad_att_des, f_des):
        R0 = self.load.R.state
        Omega0 = self.load.omega.state

        J0 = self.load.J
        hatOmega0 = hat(Omega0)
        hatOmega02 = hatOmega0 @ hatOmega0
        e3 = self.e3

        LHS = np.diag([self.load.mass * self.I, J0])
        RHS = np.zeros((6, 1))
        RHS[3:6, :] = - hatOmega0 @ J0 @ Omega0

        for i, quad in enumerate(self.quads.systems):
            m = quad.mass
            link = quad.link
            omega = link.omega.state
            q = link.uvec.state
            qqT = q @ q.T
            hr = link.hatrho
            u = f_des[i] * quad.R @ e3

            LHS[:3, :3] += m * qqT
            LHS[:3, 3:6] += - m * qqT @ R0 @ hr
            LHS[3:6, :3] += m * hr @ R0.T @ qqT
            LHS[3:6, 3:6] += - m * hr @ R0.T @ qqT @ R0 * hr

            rhs1 = (qqT @ u
                    - m * link.length * omega.T @ omega * q
                    - m * qqT @ R0 @ hatOmega02 @ link.rho)
            RHS[:3, :] += rhs1
            RHS[3:6, :] += hr @ R0.T @ rhs1

        load_accs = inv(LHS) @ RHS
        load_acc, load_ang_acc = load_accs[:3] + self.g * e3, load_accs[3:6]

        self.load.set_dot(load_acc, load_ang_acc)

        for i, quad in enumerate(self.quads.systems):
            link = quad.link
            q = link.uvec.state
            hq = hat(q)
            u = f_des[i] * quad.R @ e3

            ang_acc = 1 / link.length * (
                hq @ (
                    load_accs[:3]
                    - R0 @ link.hatrho @ load_accs[3:6]
                    + R0 @ hatOmega02 @ link.rho
                )
                - 1 / quad.mass * hq @ (self.I - qqT) @ u
            )

            link.set_dot(ang_acc)

            M = self.control_attitude(quad_att_des[i], quad)
            quad.set_dot(M)

        return dict(quad_att_des=quad_att_des, quad_moment=M, f_des=f_des)

    def step(self, action, des):
        # quad_att_des = 3*[np.vstack((np.pi/12., 0., 0.))]
        # quad_att_des = 3*[np.vstack((0., 0., 0.))]
        # f_des = 3*[25]
        load_pos_des, load_att_des, psi_des = des
        quad_att_des, f_des = self.transform_action2des(action, psi_des)
        *_, time_out = self.update(quad_att_des=quad_att_des, f_des=f_des)
        done = self.terminate(time_out)
        obs = self.observe(load_pos_des, load_att_des)
        reward = self.get_reward(load_pos_des, load_att_des)
        info = {
            'time': self.clock.get(),
            'reward': reward,
            'action': action,
        }
        return obs, reward, done, info

    def logger_callback(self, t, quad_att_des, f_des):
        load_att = R2angle(self.load.R.state)
        quad_pos = [None]*self.N
        quad_vel = [None]*self.N
        quad_att = [None]*self.N
        anchor_pos = [None]*self.N
        distance_btw_quad2anchor = [None]*self.N

        x0 = self.load.pos.state
        R0 = self.load.R.state

        for i, quad in enumerate(self.quads.systems):
            link = quad.link
            q = link.uvec.state
            rho = link.rho

            anchor_pos[i] = x0 + R0 @ rho
            quad_pos[i] = anchor_pos[i] - link.length * q
            quad_vel[i] = (
                self.load.vel.state
                + R0 @ hat(self.load.omega.state) @ rho
                - link.length * hat(link.omega.state).dot(q)
            )
            quad_att[i] = R2angle(quad.R.state)
            distance_btw_quad2anchor[i] = np.sqrt(
                (quad_pos[i][0][0] - anchor_pos[i][0][0])**2
                + (quad_pos[i][1][0] - anchor_pos[i][1][0])**2
                + (quad_pos[i][2][0] - anchor_pos[i][2][0])**2
            )
            if distance_btw_quad2anchor[i] < 0.1 or distance_btw_quad2anchor[i] > 1.:
                print('problem!')
        distance_btw_quads = self.check_collision(quad_pos)
        return dict(time=t, **self.observe_dict(), load_att=load_att,
                    anchor_pos=anchor_pos, quad_vel=quad_vel,
                    quad_att=quad_att, quad_pos=quad_pos,
                    distance_btw_quads=distance_btw_quads,
                    distance_btw_quad2anchor=distance_btw_quad2anchor)

    def check_collision(self, quads_pos):
        distance = [
            np.sqrt(
                (quads_pos[i-1][0][0]-quads_pos[i][0][0])**2
                + (quads_pos[i-1][1][0]-quads_pos[i][1][0])**2
                + (quads_pos[i-1][2][0]-quads_pos[i][2][0])**2
            ) for i in range(self.N)
        ]
        if not self.iscollision and any(
            i < self.cfg.quad.iscollision for i in distance
        ):
            self.iscollision = True
        return distance

#     def terminate(self, done):
#         load_posz = self.load.pos.state[2]
#         done = 1. if (load_posz < 0 or done or self.iscollision) else 0.
#         return done

    def control_attitude(self, quad_att_des, quad):
        omega = quad.omega.state,
        J = quad.J
        quad_att = np.vstack(R2angle(quad.R.state))
        phi, theta, _ = quad_att.squeeze()
        wx, wy, wz = omega.squeeze()

        L = np.array([
            [1, sin(phi)*tan(theta), cos(phi)*tan(theta)],
            [0, cos(phi), -sin(phi)],
            [0, sin(phi)/cos(theta), cos(phi)/cos(theta)]
        ])
        L2 = np.array([
            [wy*cos(phi)*tan(theta) - wz*sin(phi)*tan(theta),
             wy*sin(phi)/cos(theta)**2 + wz*cos(phi)/cos(theta)**2,
             0],
            [-wy*sin(phi) - wz*cos(phi), 0, 0],
            [wy*cos(phi)/cos(theta) - wz*sin(phi)*cos(theta),
             wy*sin(phi)*tan(theta)/cos(theta)
             - wz*cos(phi)*tan(theta)/cos(theta),
             0]
        ])
        e2 = L.dot(omega)
        s_clip = np.clip(
            (self.Ke * (quad_att - quad_att_des) + e2)/self.cb, -1, 1)
        return J @ inv(L) @ (
            - self.Ke * e2 - np.vstack((wx, 0, 0)) - L2 @ e2
            - s_clip * (self.mu + self.Ks)
        ) + hat(omega) @ J.dot(omega)

#     def observe(self, load_pos_des, load_att_des):
#         obs = [np.array(rot.cart2sph2(link.uvec.state))[1::]
#                for link in self.links.systems]
#         load_pos = self.load.pos.state
#         load_att = np.vstack(rot.dcm2angle(self.load.dcm.state.T))[::-1]
#         e_load_pos = load_pos - load_pos_des
#         e_load_att = load_att - load_att_des
#         obs.append(e_load_pos.reshape(-1,))
#         obs.append(e_load_att.reshape(-1,))
#         return np.hstack(obs)

#     def get_reward(self, load_pos_des, load_att_des):
#         error = self.observe(load_pos_des, load_att_des)[0:6]
#         load_pos = self.load.pos.state
#         if (load_pos[2] < 0 or self.iscollision):
#             r = -np.array([self.cfg.ddpg.reward_max])
#         else:
#             r = -np.transpose(error).dot(
#                 self.cfg.ddpg.P.dot(error)
#             ).reshape(-1,)
#         r_scaled = (r/(self.cfg.ddpg.reward_max/2)+1)*10
#         return r_scaled

#     def transform_action2des(self, action, psi_des):
#         f_des = [None]*self.cfg.quad.num
#         quad_att_des = [None]*self.cfg.quad.num
#         for i in range(self.cfg.quad.num):
#             chi, gamma = action[3*i+1:3*i+3]
#             u_des = rot.sph2cart2(1, chi, gamma)
#             phi, theta = self.find_euler(u_des, psi_des[i])
#             quad_att_des[i] = np.vstack((phi, theta, psi_des[i]))
#             f_des[i] = action[3*i]
#         return quad_att_des, f_des

#     def find_euler(self, vec, psi):
#         vec_n = rot.angle2dcm(psi, 0, 0).dot(vec)
#         theta = np.arctan2(vec_n[0], vec_n[2]).item()
#         phi = np.arctan2(
#             -vec_n[1]*vec_n[2],
#             np.cos(theta)*(1-vec_n[1]**2)
#         ).item()
#         return phi, theta


if __name__ == "__main__":
    env = MultiQuadSlungLoad()
