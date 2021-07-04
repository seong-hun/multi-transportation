import numpy as np
from pathlib import Path

from fym import parser

pi = np.pi

default_cfg = {
    # ====== Experiments ====== #

    # ------ Paths ------ #
    "exp.path": {
        "basedir": "data/exp",
        "relpath": {
            "envpath": "env.h5",
            "imgdir": "img"
        },
        "abspath": {},
    },

    # ------ Problem setup ------ #

    "exp.prob": {
        "Controller": None,
    },

    # ------ Simulation setup ------ #

    "exp.sim.kwargs": dict(dt=0.01, max_t=10),

    # ====== envs.py ====== #

    # ------ Enviroment ------ #


    # ------ Controller ------ #

    "env.controller.Ke": 20,
    "env.controller.Ks": 80,
    "env.controller.chatteringBound": 0.5,
    "env.controller.maxUncertainty": 0.1,

    # ====== dynamcis.py ====== #

    "dynamics.gravity": 9.81,

    # ------ Load ------ #

    # Random initial states
    "dynamics.load.randomInit": True,
    "dynamics.load.initBounds": {
        "position": [
            (-5, 5),  # x
            (-5, 5),  # y
            (1, 10),  # z
        ],
        "attitude": np.deg2rad([
            (-45, 45),  # phi
            (-45, 45),  # theta
            (-180, 180),  # psi
        ]),
    },

    # Deterministic initial states
    "dynamics.load.initStates": {
        "pos": np.vstack((0., 0., -3.)),
        "vel": np.zeros((3, 1)),
        "R": np.eye(3),
        "Omega": np.zeros((3, 1))
    },

    # Physical properties of the load
    "dynamics.load.physicalProperties": {
        "mass": 1.5,
        "J": np.diag((0.2, 0.2, 0.2)),
        "cg": np.vstack((0., 0., 0.7)),
        "size": 1.,
    },

    # ------ Link ------ #

    # Random initial states
    "dynamics.link.initBounds": {
        "uvec": np.deg2rad([
            (-90, 90),
            (0, 45),
        ]),
    },

    # Deterministic initial states
    "dynamics.link.initStates": {
        "uvec": None,
        "omega": np.zeros((3, 1)),
    },

    # ------ Quadrotor ------ #

    "dynamics.quadrotor.initStates": {
        "R": np.eye(3),
        "Omega": np.zeros((3, 1)),
    },
    "dynamics.quadrotor.isCollision": 0.5,
    "dynamics.quadrotor.physicalProperties": {
        "mass": 0.755,
        "J": np.diag([0.0820, 0.0845, 0.1377]),
    },

    # ------ MultiQuadSlungLoad ------ #

    "dynamics.MQSL.quads.num": 5,

    "dynamics.MQSL.links.autoInit": True,
    "dynamics.MQSL.links.autoInitParams": {
        "length": 1,
        "azimuth": "equal",
        "elevation": np.deg2rad(-5),
    },
}

cfg = parser.copy(default_cfg)


def load():
    return cfg


def set(d=None):
    parser.update(cfg, default_cfg, prune=True)
    parser.update(cfg, d or {})

    # Setup paths
    pathcfg = load().exp.path
    for k, v in parser.decode(pathcfg.relpath).items():
        Path(pathcfg.basedir, pathcfg.relpath.envpath)
        parser.update(pathcfg.abspath, {k: Path(pathcfg.basedir, v)})

    # if pathcfg.envpath is None:
    #     pathcfg.envpath = Path(pathcfg.basedir, pathcfg.relpath.envpath)

    # if pathcfg.imgdir is None:
    #     pathcfg.imgdir = Path(pathcfg.basedir, pathcfg.relpath.imgdir)

    # Setup simulation
    expcfg = load().exp
    expcfg.sim.kwargs = parser.decode(expcfg.sim.kwargs)
