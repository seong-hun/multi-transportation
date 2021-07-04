import numpy as np
import random
from types import SimpleNamespace as SN
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt

import torch

import fym
from fym.utils import rot

import envs
import config
from agents import DDPG
from dynamics import MultiQuadSlungLoad
from utils import draw_plot, compare_episode, OrnsteinUhlenbeckNoise

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

cfg = config.load()


def load_config():
    cfg = SN()
    cfg.epi_train = 1
    cfg.epi_eval = 1
    cfg.dt = 0.1
    cfg.max_t = 5.
    cfg.solver = 'odeint'
    cfg.ode_step_len = 10
    cfg.dir = Path('log', datetime.today().strftime('%Y%m%d-%H%M%S'))
    cfg.g = np.vstack((0., 0., -9.81))

    cfg.animation = SN()
    cfg.animation.quad_size = 0.315
    cfg.animation.rotor_size = 0.15
    cfg.animation.view_angle = [None, None]

    cfg.quad = SN()
    cfg.quad.num = 3
    cfg.quad.mass = 0.755
    cfg.quad.J = np.diag([0.0820, 0.0845, 0.1377])
    cfg.quad.iscollision = 0.5
    cfg.quad.psi_des = cfg.quad.num*[0]
    cfg.quad.omega_init = np.vstack((0., 0., 0.))

    cfg.load = SN()
    cfg.load.pos_bound = [[-5, 5], [-5, 5], [1, 10]]
    cfg.load.att_bound = [
        [-np.pi/4, np.pi/4],
        [-np.pi/4, np.pi/4],
        [-np.pi, np.pi]
    ]
    cfg.load.pos_init = np.vstack((0., 0., 3.))
    cfg.load.dcm_init = rot.angle2dcm(0., 0., 0.).T
    cfg.load.vel_init = np.vstack((0., 0., 0.))
    cfg.load.mass = 1.5
    cfg.load.J = np.diag([0.2, 0.2, 0.2])
    cfg.load.cg = np.vstack((0., 0., -0.7))
    cfg.load.size = 1.
    cfg.load.pos_des = np.vstack((0., 0., 5.))
    cfg.load.att_des = np.vstack((0., 0., 0.))

    cfg.link = SN()
    cfg.link.len = cfg.quad.num * [0.5]
    cfg.link.anchor = [
        np.vstack((
            cfg.load.size * np.cos(i*2*np.pi/cfg.quad.num),
            cfg.load.size * np.sin(i*2*np.pi/cfg.quad.num),
            0
        )) - cfg.load.cg for i in range(cfg.quad.num)
    ]
    cfg.link.uvec_bound = [[-np.pi, np.pi], [0, np.pi/4]]

    cfg.controller = SN()
    cfg.controller.Ke = 20.
    cfg.controller.Ks = 80.
    cfg.controller.chattering_bound = 0.5
    cfg.controller.unc_max = 0.1

    cfg.ddpg = SN()
    cfg.ddpg.P = np.diag([1., 1., 1., 5., 5., 2.])
    cfg.ddpg.reward_max = 120
    cfg.ddpg.state_dim = 6 + 2*cfg.quad.num
    cfg.ddpg.action_dim = 3*cfg.quad.num
    cfg.ddpg.action_scaling = torch.Tensor(
        cfg.quad.num * [2.5, np.pi, np.pi/12]
    )
    cfg.ddpg.action_bias = torch.Tensor(cfg.quad.num * [12.5, 0, np.pi/12])
    cfg.ddpg.memory_size = 20000
    cfg.ddpg.actor_lr = 0.0001
    cfg.ddpg.critic_lr = 0.001
    cfg.ddpg.discount = 0.999
    cfg.ddpg.softupdate = 0.001
    cfg.ddpg.batch_size = 64

    cfg.noise = SN()
    cfg.noise.rho = 0.15
    cfg.noise.mu = 0.
    cfg.noise.sigma = 0.2
    cfg.noise.size = 9

    return cfg


def train(agent, des, cfg, noise, env):
    x = env.reset(des)
    noise.reset()
    while True:
        action = np.clip(
            agent.get_action(x) + noise.get_noise(),
            np.array(-cfg.ddpg.action_scaling + cfg.ddpg.action_bias),
            np.array(cfg.ddpg.action_scaling + cfg.ddpg.action_bias)
        )
        xn, r, done, _ = env.step(action, des)
        agent.memorize((x, action, r, xn, done))
        x = xn
        if len(agent.memory) > 5 * cfg.ddpg.batch_size:
            agent.train()
        if done:
            break
    plt.close('all')


def evaluate(env, agent, des, cfg, dir_env_data, dir_agent_data):
    env.logger = fym.Logger(dir_env_data)
    env.logger.set_info(cfg=cfg)
    logger_agent = fym.Logger(dir_agent_data)
    x = env.reset(des, fixed_init=True)
    while True:
        action = agent.get_action(x)
        xn, _, done, info = env.step(action, des)
        logger_agent.record(**info)
        x = xn
        if done:
            break
    logger_agent.close()
    env.logger.close()
    plt.close('all')


def exp2():
    env = MultiQuadSlungLoad()
    agent = DDPG()
    noise = OrnsteinUhlenbeckNoise(
        cfg.noise.rho,
        cfg.noise.mu,
        np.array(cfg.ddpg.action_scaling),
        cfg.dt,
        cfg.ddpg.action_dim
    )
    des = [cfg.load.pos_des, cfg.load.att_des, cfg.quad.psi_des]

    for epi_num in tqdm(range(cfg.epi_train)):
        train(agent, des, cfg, noise, env)

        if (epi_num+1) % cfg.epi_eval == 0:
            dir_save = Path(cfg.dir, f"epi_after_{epi_num+1:05d}")
            dir_env_data = Path(dir_save, "env_data.h5")
            dir_agent_data = Path(dir_save, "agent_data.h5")
            dir_agent_params = Path(dir_save, "agent_params.h5")

            evaluate(env, agent, des, cfg, dir_env_data, dir_agent_data)
            draw_plot(dir_env_data, dir_agent_data, dir_save)
            agent.save_parameters(dir_agent_params)
    env.close()


def run():
    expcfg = config.load().exp

    class Env(fym.BaseEnv):
        def __init__(self):
            super().__init__(**expcfg.sim.kwargs)
            self.plant = MultiQuadSlungLoad()

            self.logger = fym.Logger(expcfg.path.abspath.envpath)
            self.logger.set_info(cfg=config.load())

        def step(self):
            *_, done = self.update()
            return done

        def set_dot(self, t):
            fs = [5 * np.clip(quad.pos[2] - (-5), 0, None) for quad in self.plant.quads.systems]
            Ms = [np.zeros((3, 1))] * self.plant.N
            self.plant.set_dot(t, fs, Ms)

        def logger_callback(self, t):
            state_dict = self.observe_dict()
            quads_dict = state_dict["plant"]["quads"]
            for k, v in quads_dict.items():
                v["pos"] = self.plant.quads.systems_dict[k].pos

            return dict(t=t, **state_dict)

    env = Env()

    env.reset()
    while True:
        env.render()
        done = env.step()

        if done:
            break

    env.close()


def exp1():
    settings = {
        "exp.path.basedir": "data/exp1",
        "exp.path.relpath.moviepath": "movie.mp4",
        "exp.sim.kwargs": dict(dt=0.01, max_t=10),
        "dynamics.MQSL.quads.num": 8,
    }
    config.set(settings)
    run()


def exp1plot():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    import dvis

    data, info = fym.load("data/exp1/env.h5", with_info=True)
    cfg = info["cfg"]

    fig = plt.figure()
    ax = plt.subplot(111, projection="3d")

    quads = {name: dvis.Quadrotor(ax) for name in data["plant"]["quads"]}
    links = {name: dvis.Link(ax) for name in data["plant"]["links"]}
    links_cfg = fym.parser.decode(cfg.dynamics.MQSL.links)
    load = dvis.Load(ax, [links_cfg[name]["anchor"].squeeze()
                          for name in data["plant"]["links"]])

    def init_func():
        # set an axis
        ax.set_xlim3d(-3, 3)
        ax.set_ylim3d(-3, 3)
        ax.set_zlim3d(-6, -2)
        ax.invert_zaxis()

        fig.subplots_adjust(
            left=0.1, right=0.9, top=0.9, bottom=0.1,
            wspace=0.2, hspace=0.2)

    def func(frame):
        x0 = data["plant"]["load"]["pos"][frame]
        R0 = data["plant"]["load"]["R"][frame]

        load.set(x0.squeeze(), R0)

        for k, v in fym.parser.decode(data["plant"]["quads"]).items():
            quads[k].set(v["pos"][frame].squeeze(), v["R"][frame])

        for k, v in fym.parser.decode(data["plant"]["links"]).items():
            start = x0 + R0 @ links_cfg[k]["anchor"]
            end = start - v["uvec"][frame] * links_cfg[k]["length"]
            links[k].set(start.squeeze(), end.squeeze())

    fps = 30
    interval = 1 / fps
    step = int(interval / cfg.exp.sim.kwargs.dt) + 1
    interval = step * cfg.exp.sim.kwargs.dt * 1000
    frames = range(0, data["t"].size, step)
    ani = dvis.FuncAnimation(fig, init_func=init_func, func=func,
                             frames=frames, interval=interval)
    ani.save(cfg.exp.path.abspath.moviepath)


def main():
    exp1()
    exp1plot()


if __name__ == "__main__":
    main()
