# -*- coding: utf-8 -*-
"""
Switch the active robot -- rewrites "model" and "trunk_body" in config.json.

Usage:
    python set_model.py                     # list robots/ and pick one
    python set_model.py robots/body2.xml    # set it directly
"""
import os
import re
import sys
import glob

import mujoco

CFG_PATH = "config.json"


def detect_trunk(model):
    """Find the trunk body: prefer the one with a free joint, else body 1."""
    for b in range(1, model.nbody):
        jadr, jnum = model.body_jntadr[b], model.body_jntnum[b]
        for j in range(jadr, jadr + jnum):
            if j >= 0 and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 1)


def set_in_config(cfg_path, model_path, trunk):
    """Replace the first uncommented "model" and "trunk_body" lines in place."""
    with open(cfg_path, encoding="utf-8") as fh:
        s = fh.read()
    name = os.path.basename(model_path)
    s = re.sub(r'(?m)^(\s*)"model"\s*:\s*"[^"]*"\s*,?.*$',
               rf'\1"model": "{model_path}",     // active model: {name}', s, count=1)
    s = re.sub(r'(?m)^(\s*)"trunk_body"\s*:\s*"[^"]*"\s*,?.*$',
               rf'\1"trunk_body":   "{trunk}",  // trunk body (auto-detected)', s, count=1)
    with open(cfg_path, "w", encoding="utf-8") as fh:
        fh.write(s)


def current_model(cfg_path):
    try:
        with open(cfg_path, encoding="utf-8") as fh:
            m = re.search(r'(?m)^\s*"model"\s*:\s*"([^"]*)"', fh.read())
        return m.group(1) if m else ""
    except OSError:
        return ""


def choose_interactively():
    files = sorted(glob.glob("robots/*.xml"))
    if not files:
        print("No .xml model in robots/. Convert a URDF first with 4_import_model.bat.")
        return None
    cur = current_model(CFG_PATH).replace("\\", "/")
    print("Models available in robots/:\n")
    for i, f in enumerate(files, 1):
        mark = "  <- active" if f.replace("\\", "/") == cur else ""
        print(f"  [{i}] {f}{mark}")
    print()
    k = input("Pick one (enter a number): ").strip()
    if not k.isdigit() or not (1 <= int(k) <= len(files)):
        print("Not a valid number, aborting.")
        return None
    return files[int(k) - 1]


def check_hydro_rules(model):
    """Warn about geoms that no hydrodynamic rule matches; they fall back to default."""
    from simulate import load_cfg
    rules = [r["match"] for r in load_cfg(CFG_PATH).get("hydro", {}).get("rules", [])]
    names = [(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
             for g in range(model.ngeom)]
    unmatched = [n for n in names if n and not any(r.lower() in n.lower() for r in rules)]
    if unmatched:
        print(f"\n[!] {len(unmatched)}/{len(names)} geoms match no hydro rule "
              f"and will use the default coefficients:")
        print("    " + ", ".join(unmatched[:6]) + (" ..." if len(unmatched) > 6 else ""))
        print("    -> edit hydro.rules in config.json so the match keywords fit your model")
    else:
        print("\n[ok] every geom matches a hydrodynamic rule")


def main():
    choice = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".xml") else None
    if choice is None:
        choice = choose_interactively()
        if choice is None:
            return

    choice = choice.replace("\\", "/")
    print(f"\nloading {choice} ...")
    model = mujoco.MjModel.from_xml_path(choice)
    trunk = detect_trunk(model)
    set_in_config(CFG_PATH, choice, trunk)
    print("written to config.json:")
    print(f'   "model"      : "{choice}"')
    print(f'   "trunk_body" : "{trunk}"   (auto-detected)')
    print(f"\nmodel: bodies {model.nbody-1} | geoms {model.ngeom} | actuators {model.nu}")

    if model.nu == 0:
        print("[!] this model has no actuators. "
              "Re-import it with 4_import_model.bat, which adds them automatically.")
    else:
        shown = ", ".join(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                          for i in range(min(model.nu, 12)))
        print("actuators:", shown + (" ..." if model.nu > 12 else ""))

    check_hydro_rules(model)
    print("\nNext: run 1_demo_gait.bat or 5_optimize_live.bat.")


if __name__ == "__main__":
    main()
