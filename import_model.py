# -*- coding: utf-8 -*-
"""
Layer 1, model import -- convert a URDF into an MJCF this pipeline can use.

It does four things automatically:
  1. rewrite package:// mesh paths, which MuJoCo cannot resolve
  2. inject <mujoco><compiler> settings (mesh directory, inertia fixes)
  3. add position actuators to the selected joints, since URDF has no actuator concept
  4. print a report of bodies, joints and geoms, and how hydro coefficients will map

Usage:
    python import_model.py my_robot.urdf robots/mine.xml --drive "2.1,1.1"

    --drive    comma-separated keywords; joints whose name contains one get an
               actuator. Omit it to actuate every movable joint.
    --meshdir  mesh folder; by default it is located automatically.
"""
import os
import re
import sys
import shutil
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import mujoco


def find_meshdir(urdf_path, text):
    """Pick a mesh filename from the URDF, find it on disk, and infer meshdir."""
    names = re.findall(r'filename="([^"]+)"', text)
    if not names:
        return "."
    rel = names[0].lstrip("./")                    # e.g. meshes/base_link.STL
    fname = os.path.basename(rel)
    root = os.path.dirname(os.path.abspath(urdf_path))
    for base in [root, os.path.dirname(root), os.path.dirname(os.path.dirname(root))]:
        for dirpath, _, files in os.walk(base):
            if fname in files:
                # meshdir must be such that (meshdir + rel) is the real file
                sub = os.path.dirname(rel)         # "meshes" or ""
                cand = os.path.abspath(os.path.join(dirpath, "..")) if sub else dirpath
                if os.path.isfile(os.path.join(cand, rel)):
                    return cand
                if os.path.isfile(os.path.join(dirpath, fname)) and not sub:
                    return dirpath
    return root


def fix_urdf(src, meshdir):
    """Return the URDF text with package:// stripped and compiler options injected."""
    with open(src, "r", encoding="utf-8", errors="ignore") as fh:
        s = fh.read()
    s = re.sub(r'package://[^/]+/', '', s)
    if "<mujoco>" not in s:
        inject = (f'  <mujoco>\n'
                  f'    <compiler meshdir="{meshdir}" balanceinertia="true" '
                  f'discardvisual="false" fusestatic="false" strippath="false"/>\n'
                  f'  </mujoco>\n')
        s = re.sub(r'(<robot[^>]*>)', r'\1\n' + inject, s, count=1)
    return s


def add_actuators(mjcf_path, drive_keys, kp=60.0, forcerange=12.0, ctrlrange=1.4):
    """Add a position actuator to every joint matching one of `drive_keys`."""
    with open(mjcf_path, encoding="utf-8") as fh:
        s = fh.read()
    joints = re.findall(r'<joint[^>]*name="([^"]+)"[^>]*>', s)
    sel = [j for j in joints
           if not drive_keys or any(k.strip().lower() in j.lower() for k in drive_keys)]
    if not sel:
        print("[!] no joint matched, adding no actuators")
        return s, []
    acts = "\n".join(
        f'    <position name="{j}" joint="{j}" kp="{kp}" dampratio="1" '
        f'ctrlrange="-{ctrlrange} {ctrlrange}" forcerange="-{forcerange} {forcerange}"/>'
        for j in sel)
    if "<actuator>" in s:
        s = re.sub(r'</actuator>', acts + "\n  </actuator>", s, count=1)
    else:
        s = s.replace("</mujoco>", f"  <actuator>\n{acts}\n  </actuator>\n</mujoco>")
    with open(mjcf_path, "w", encoding="utf-8") as fh:
        fh.write(s)
    return s, sel


def make_portable(mjcf_path):
    """Copy referenced meshes next to the model and switch to relative paths.

    MuJoCo saves absolute mesh paths, so without this the model stops loading on any
    other machine.
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    outdir = os.path.dirname(os.path.abspath(mjcf_path))
    mdir = os.path.join(outdir, "meshes")
    comp0 = root.find("compiler")
    old_meshdir = (comp0.get("meshdir") if comp0 is not None else "") or ""
    n_copy = 0
    for mesh in root.iter("mesh"):
        f = mesh.get("file")
        if not f:
            continue
        if os.path.isabs(f):
            src = f
        else:
            cands = [os.path.join(old_meshdir, f), os.path.join(outdir, f)]
            src = next((c for c in cands if os.path.isfile(c)), cands[0])
        base = os.path.basename(f)
        dst = os.path.join(mdir, base)
        if os.path.isfile(src):
            os.makedirs(mdir, exist_ok=True)
            if not os.path.isfile(dst) or os.path.getsize(dst) != os.path.getsize(src):
                shutil.copy2(src, dst)
                n_copy += 1
        mesh.set("file", base)                      # bare filename
    comp = root.find("compiler")
    if comp is None:
        comp = ET.Element("compiler")
        root.insert(0, comp)
    comp.set("meshdir", "meshes")                   # to go with the bare filenames
    comp.set("strippath", "false")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)
    return n_copy


def add_defaults(mjcf_path, armature, damping):
    """Set armature and damping on every joint; both matter for light link chains."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    d = root.find("default")
    if d is None:
        d = ET.Element("default")
        root.insert(0, d)
    j = d.find("joint")
    if j is None:
        j = ET.SubElement(d, "joint")
    j.set("armature", str(armature))
    j.set("damping", str(damping))
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)


def name_geoms(mjcf_path):
    """Name unnamed geoms after their body.

    URDF conversion usually leaves geoms unnamed, and the hydrodynamic rules match on
    names, so without this every geom falls back to the default coefficients.
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    n = 0

    def walk(el, body_name):
        nonlocal n
        for ch in el:
            if ch.tag == "body":
                walk(ch, ch.get("name", body_name))
            else:
                if ch.tag == "geom" and not ch.get("name"):
                    ch.set("name", f"{body_name}_g{n}")
                    n += 1
                walk(ch, body_name)

    wb = root.find("worldbody")
    if wb is not None:
        walk(wb, "world")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)
    return n


def report(model, rules_hint=("2.4", "2.3", "2.2", "2.1", "1.", "base")):
    print("\n" + "=" * 64)
    print("MODEL REPORT")
    print("=" * 64)
    print(f"bodies {model.nbody-1} | geoms {model.ngeom} | dofs {model.nq} | "
          f"actuators {model.nu}")

    print("\n[actuators] (gait parameters are generated from these)")
    for i in range(model.nu):
        print("   ", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))

    print("\n[geoms -> hydro coefficients] (matched by keyword against hydro.rules)")
    hit = {k: [] for k in rules_hint}
    miss = []
    for g in range(model.ngeom):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        for k in rules_hint:
            if k in n.lower():
                hit[k].append(n)
                break
        else:
            miss.append(n)
    for k, v in hit.items():
        if v:
            print(f"    contains '{k}' -> {len(v)}: "
                  f"{', '.join(v[:4])}{' ...' if len(v) > 4 else ''}")
    if miss:
        print(f"    [!] unmatched, will use default coefficients -> {len(miss)}: "
              f"{', '.join(miss[:6])}{' ...' if len(miss) > 6 else ''}")
        print("        fix by adding rules to hydro.rules in config.json, "
              "or by renaming the geoms")

    tot_m = float(np.sum(model.body_mass[1:]))
    print(f"\ntotal mass {tot_m*1000:.1f} g   "
          f"(if this disagrees with the real robot, correct mass in the model)")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(description="Convert a URDF into an MJCF for swimopt.")
    ap.add_argument("urdf")
    ap.add_argument("out")
    ap.add_argument("--drive", default="",
                    help="comma-separated joint-name keywords to actuate")
    ap.add_argument("--armature", type=float, default=1e-4,
                    help="joint rotor inertia; the key to stabilizing light link chains")
    ap.add_argument("--damping", type=float, default=0.01, help="joint damping")
    ap.add_argument("--kp", type=float, default=20.0, help="position actuator stiffness")
    ap.add_argument("--force", type=float, default=5.0, help="actuator force limit, N*m")
    ap.add_argument("--meshdir", default="")
    a = ap.parse_args()

    with open(a.urdf, encoding="utf-8", errors="ignore") as fh:
        raw = fh.read()
    raw_nopkg = re.sub(r'package://[^/]+/', '', raw)
    meshdir = a.meshdir or find_meshdir(a.urdf, raw_nopkg)
    print(f"[1/4] reading {a.urdf}\n      mesh directory (auto-located): {meshdir}")

    fixed = fix_urdf(a.urdf, meshdir)
    outdir = os.path.dirname(os.path.abspath(a.out)) or "."
    os.makedirs(outdir, exist_ok=True)
    tmp = os.path.join(outdir, "_mj_tmp.urdf")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(fixed)

    print("[2/4] compiling the URDF with MuJoCo ...")
    try:
        m = mujoco.MjModel.from_xml_path(tmp)
    except Exception as e:
        print("  [X] compilation failed:", str(e)[:400])
        print("  usual causes: wrong mesh path (pass --meshdir), missing STL files, "
              "or invalid inertia")
        try:
            os.remove(tmp)
        except OSError:
            pass
        sys.exit(1)

    mujoco.mj_saveLastXML(a.out, m)
    try:
        os.remove(tmp)
    except OSError:
        pass
    print(f"[3/4] saved MJCF: {a.out}")

    nc = make_portable(a.out)
    print(f"      copied {nc} meshes into robots/meshes/ "
          f"(model now uses relative paths and is portable)")
    add_defaults(a.out, a.armature, a.damping)
    nn = name_geoms(a.out)
    print(f"      named {nn} geoms after their body (needed for hydro rule matching)")

    print("[4/4] adding actuators ...")
    keys = [k for k in a.drive.split(",") if k.strip()]
    _, sel = add_actuators(a.out, keys, kp=a.kp, forcerange=a.force)
    print(f"      added {len(sel)} position actuators"
          + (f" (matching {keys})" if keys else " (all movable joints)"))

    report(mujoco.MjModel.from_xml_path(a.out))
    print(f'\nNext: set "model" to "{a.out}" in config.json, then run 1_demo_gait.bat.')


if __name__ == "__main__":
    main()
