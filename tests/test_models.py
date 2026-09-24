# -*- coding: utf-8 -*-
"""The shipped model, the URDF importer, and set_model's config rewrite."""
import copy
import os
import shutil
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from conftest import DATA, ROOT
from hydro import HydroModel
from simulate import load_cfg


def test_toy_quad_joint_ranges_are_radians():
    """MJCF reads ranges as degrees unless told otherwise; toy_quad once shipped with
    +/-1.3 *degree* limits that pinned every leg."""
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "robots", "toy_quad.xml"))
    hinges = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    assert min(m.jnt_range[j, 1] for j in hinges) > 1.0


def test_toy_quad_commands_stay_inside_joint_limits():
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "robots", "toy_quad.xml"))
    for a in range(m.nu):
        j = m.actuator_trnid[a, 0]
        lo, hi = m.jnt_range[j]
        assert lo <= m.actuator_ctrlrange[a, 0] and m.actuator_ctrlrange[a, 1] <= hi


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    out = tmp_path_factory.mktemp("imp") / "mini.xml"
    subprocess.run([sys.executable, os.path.join(ROOT, "import_model.py"),
                    os.path.join(DATA, "mini.urdf"), str(out), "--drive", "2.1"],
                   check=True, capture_output=True, cwd=ROOT)
    return mujoco.MjModel.from_xml_path(str(out))


def test_import_gives_a_floating_base(imported):
    assert any(imported.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
               for j in range(imported.njnt))


def test_import_actuators_use_the_joint_range(imported):
    assert imported.nu == 1
    j = imported.actuator_trnid[0, 0]
    assert np.allclose(imported.actuator_ctrlrange[0], imported.jnt_range[j])


def test_import_counts_each_link_volume_once(imported):
    h = HydroModel(imported, load_cfg(os.path.join(ROOT, "config.json"))["hydro"])
    counted = sum(i["V"] for i in h.items)
    real = 0.2 * 0.1 * 0.05 + 0.01 * 0.01 * 0.06
    assert counted == pytest.approx(real)


def test_set_model_rewrites_only_the_two_lines(tmp_path):
    work = tmp_path / "proj"
    work.mkdir()
    shutil.copy(os.path.join(ROOT, "config.json"), work / "config.json")
    before = load_cfg(str(work / "config.json"))
    import set_model
    here = os.getcwd()
    os.chdir(work)
    try:
        set_model.set_in_config("config.json", "robots/other.xml", "base")
    finally:
        os.chdir(here)
    after = load_cfg(str(work / "config.json"))
    assert after["model"] == "robots/other.xml" and after["trunk_body"] == "base"
    for k in before:
        if k not in ("model", "trunk_body"):
            assert before[k] == after[k], k
