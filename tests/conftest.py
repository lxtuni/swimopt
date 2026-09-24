# -*- coding: utf-8 -*-
"""Shared fixtures. Run the suite from the project root with:  python -m pytest"""
import copy
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data")
sys.path.insert(0, ROOT)
os.chdir(ROOT)                    # model paths in config.json are relative to the root

from simulate import Swimmer, load_cfg   # noqa: E402

_BASE = load_cfg(os.path.join(ROOT, "config.json"))


@pytest.fixture
def cfg():
    """config.json with a short rollout, so tests that simulate stay quick."""
    c = copy.deepcopy(_BASE)
    c["sim_time"] = 1.5
    c["settle_time"] = 0.3
    return c


def make_swimmer(cfg, **overrides):
    c = copy.deepcopy(cfg)
    hydro = overrides.pop("hydro", None)
    c.update(overrides)
    if hydro:
        c["hydro"].update(hydro)
    return Swimmer(c["model"], c)


def write_mjcf(tmp_path, body_xml, name="probe.xml"):
    """A one-robot MJCF around `body_xml`, with a water plane and no gravity tweaks."""
    path = tmp_path / name
    path.write_text(f"""<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="vis_water" type="plane" size="1 1 .1" contype="0" conaffinity="0"/>
    {body_xml}
  </worldbody>
</mujoco>""", encoding="utf-8")
    return str(path)
