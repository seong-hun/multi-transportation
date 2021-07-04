import numpy as np
from matplotlib import pyplot as plt

import fym
import dvis

import envs
import config
from dynamics import MultiQuadSlungLoad

cfg = config.load()


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
            fs = [5 * np.clip(quad.pos[2] - (-5), 0, None)
                  for quad in self.plant.quads.systems]
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
    step = int(1 / fps / cfg.exp.sim.kwargs.dt) + 1
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
