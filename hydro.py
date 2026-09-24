# -*- coding: utf-8 -*-
"""
Layer 2, physics -- robot-agnostic Morison-type hydrodynamics.

For every geom in the model, per simulation step:

    F = rho*g*V*f                  buoyancy
      - 0.5*rho*Cd*A*v|v|*f        quadratic drag, anisotropic per local axis
      - cv*v*f                     linear viscous damping

where f is the immersion fraction, computed from the geom's true vertical extent
under its current orientation rather than a bounding sphere.

Added mass uses the "mass trick": at init time m_a = Ca*rho*V is folded into the
body's inertial mass, and a constant upward force m_a*g cancels the extra weight it
would otherwise introduce. That approximation is isotropic, but unlike an explicit
dv/dt term it is numerically unconditionally stable.

Coefficients are assigned by matching geom names against rules in config.json, so a
new robot needs configuration changes only, never code changes.
"""
import numpy as np
import mujoco

GT = mujoco.mjtGeom

_I1 = np.array([1, 2, 0])
_I2 = np.array([2, 0, 1])


def _cross(a, b):
    """Row-wise cross product of two (n, 3) arrays.

    np.cross validates and broadcasts on every call, which costs more than the
    arithmetic for the dozen rows this runs on each step.
    """
    return a[:, _I1] * b[:, _I2] - a[:, _I2] * b[:, _I1]


def geom_volume_areas(gtype, size):
    """Return (volume, [Ax, Ay, Az]), where A_i is the area facing the local i axis."""
    if gtype == GT.mjGEOM_BOX:
        sx, sy, sz = size[:3]
        V = 8 * sx * sy * sz
        return V, np.array([4 * sy * sz, 4 * sx * sz, 4 * sx * sy])
    if gtype == GT.mjGEOM_SPHERE:
        r = size[0]
        V = 4 / 3 * np.pi * r ** 3
        A = np.pi * r ** 2
        return V, np.array([A, A, A])
    if gtype == GT.mjGEOM_CAPSULE:
        r, hl = size[0], size[1]
        V = np.pi * r ** 2 * (2 * hl) + 4 / 3 * np.pi * r ** 3
        a_side = 2 * r * (2 * hl) + np.pi * r ** 2
        return V, np.array([a_side, a_side, np.pi * r ** 2])
    if gtype == GT.mjGEOM_CYLINDER:
        r, hl = size[0], size[1]
        V = np.pi * r ** 2 * (2 * hl)
        return V, np.array([2 * r * 2 * hl, 2 * r * 2 * hl, np.pi * r ** 2])
    if gtype == GT.mjGEOM_ELLIPSOID:
        a, b, c = size[:3]
        V = 4 / 3 * np.pi * a * b * c
        return V, np.array([np.pi * b * c, np.pi * a * c, np.pi * a * b])
    # mesh, plane and anything else: fall back to the bounding sphere
    r = max(float(size[0]), 1e-4)
    V = 4 / 3 * np.pi * r ** 3
    A = np.pi * r ** 2
    return V, np.array([A, A, A])


def vertical_half_extent(gtype, size, R):
    """Half extent along world z in the current orientation, used for partial immersion."""
    if gtype == GT.mjGEOM_BOX:
        return abs(R[2, 0]) * size[0] + abs(R[2, 1]) * size[1] + abs(R[2, 2]) * size[2]
    if gtype == GT.mjGEOM_SPHERE:
        return size[0]
    if gtype in (GT.mjGEOM_CAPSULE, GT.mjGEOM_CYLINDER):
        return abs(R[2, 2]) * size[1] + size[0]
    if gtype == GT.mjGEOM_ELLIPSOID:
        return np.sqrt((R[2, 0] * size[0]) ** 2
                       + (R[2, 1] * size[1]) ** 2
                       + (R[2, 2] * size[2]) ** 2)
    return max(float(size[0]), 1e-4)


class HydroModel:
    """Scan every geom, assign coefficients by name rule, apply forces each step."""

    def __init__(self, model, cfg):
        self.m = model
        self.rho = cfg.get("rho", 1000.0)
        self.g = abs(float(model.opt.gravity[2])) or 9.81
        self.water_z = cfg.get("water_z", 0.0)
        self.rules = cfg.get("rules", [])          # [{match, cd: [x,y,z], ca, cv}]
        self.default = cfg.get("default", {"cd": [0.3, 0.3, 0.3], "ca": 0.2, "cv": 0.02})
        self.exclude = cfg.get("exclude", [])      # geoms whose name contains these are skipped
        self.cfg_mesh_fill = cfg.get("mesh_fill", 0.55)   # solid fill ratio of a mesh bbox
        # Circulatory lift. Off by default: with lift disabled this model is purely
        # resistive, which is what every result before this option was produced with.
        self.lift_on = bool(cfg.get("lift", False))
        self.skip_visual = bool(cfg.get("skip_visual_duplicates", True))
        self.items = []                            # precomputed data per wetted geom
        self._scan()
        self._apply_added_mass()
        self._vectorize()

    # ---------- initialization ----------
    def _gname(self, gid):
        n = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, gid)
        return n if n else f"geom{gid}"

    def _match(self, name):
        for r in self.rules:
            if r["match"].lower() in name.lower():
                return r
        return self.default

    def _visual_only(self, gid):
        return self.m.geom_contype[gid] == 0 and self.m.geom_conaffinity[gid] == 0

    def _scan(self):
        # A URDF import keeps both the visual and the collision shape of every link.
        # Counting both doubles every fluid force, so a visual-only geom is skipped
        # whenever its body also has a collision geom. Bodies with nothing but visual
        # geometry keep it. Models imported by the current import_model.py already
        # have these marked 'vis_'; this catches models imported before that fix.
        has_collision = np.zeros(self.m.nbody, dtype=bool)
        for gid in range(self.m.ngeom):
            if not self._visual_only(gid):
                has_collision[self.m.geom_bodyid[gid]] = True
        self.skipped_visual = []
        for gid in range(self.m.ngeom):
            name = self._gname(gid)
            if any(x.lower() in name.lower() for x in self.exclude):
                continue
            bid = int(self.m.geom_bodyid[gid])
            if bid == 0:                        # the world body takes no fluid force
                continue
            if self.skip_visual and self._visual_only(gid) and has_collision[bid]:
                self.skipped_visual.append(name)
                continue
            gtype = self.m.geom_type[gid]
            if gtype in (GT.mjGEOM_PLANE, GT.mjGEOM_HFIELD):
                continue
            if gtype == GT.mjGEOM_MESH:
                V, A, half = self._mesh_box(gid)
                size_eff = half
            else:
                V, A = geom_volume_areas(gtype, self.m.geom_size[gid])
                size_eff = np.array(self.m.geom_size[gid], dtype=float)
            rule = self._match(name)
            ca = float(rule.get("ca", self.default["ca"]))
            cd = np.array(rule.get("cd", self.default["cd"]), dtype=float)
            # The surface behaves like a plate whose face is the axis with the largest
            # drag coefficient. That axis is the plate normal used by the lift model.
            face = int(np.argmax(cd))
            self.items.append(dict(
                gid=gid, bid=bid, name=name, V=V, A=A,
                cd=cd,
                cl=float(rule.get("cl", self.default.get("cl", 0.0))),
                face=face,
                ca=ca,
                cv=float(rule.get("cv", self.default["cv"])),
                gtype=(int(GT.mjGEOM_BOX) if gtype == GT.mjGEOM_MESH else int(gtype)),
                size=size_eff,
                ma=ca * self.rho * V,
            ))

    def _mesh_box(self, gid):
        """Treat a mesh as the box bounding its vertices, which beats a bounding sphere.

        `fill` accounts for a mesh not filling its own bounding box; it defaults to
        0.55 and is tunable through hydro.mesh_fill in the config.
        """
        mid = int(self.m.geom_dataid[gid])
        adr = int(self.m.mesh_vertadr[mid])
        num = int(self.m.mesh_vertnum[mid])
        v = self.m.mesh_vert[adr:adr + num].reshape(-1, 3)
        half = np.maximum((v.max(0) - v.min(0)) / 2.0, 1e-4)
        fill = float(self.cfg_mesh_fill)
        V = 8 * half[0] * half[1] * half[2] * fill
        A = np.array([4 * half[1] * half[2],
                      4 * half[0] * half[2],
                      4 * half[0] * half[1]]) * fill
        return V, A, half

    def _apply_added_mass(self):
        """The mass trick: fold added mass into each body, scaling inertia to match."""
        self.ma_body = np.zeros(self.m.nbody)
        for it in self.items:
            self.ma_body[it["bid"]] += it["ma"]
        for bid in range(1, self.m.nbody):
            ma = self.ma_body[bid]
            m0 = self.m.body_mass[bid]
            if ma <= 0 or m0 <= 0:
                continue
            scale = (m0 + ma) / m0
            self.m.body_mass[bid] = m0 + ma
            self.m.body_inertia[bid] *= scale
        # Writing body_mass after compilation leaves MuJoCo's derived constants stale:
        #   body_subtreemass  -> mj_subtreeVel reported velocities scaled by old/new mass
        #   dof_M0            -> a "simple" body (a lone free body) takes its mass matrix
        #                        from here, so its added mass was silently ignored
        #   *_invweight0      -> constraint softness is scaled by the old masses
        # mj_setConst recomputes all of them from the new masses.
        mujoco.mj_setConst(self.m, mujoco.MjData(self.m))

    def _vectorize(self):
        n = len(self.items)
        self.empty = (n == 0)
        if self.empty:
            # A model with no wetted geom is legal but produces no fluid force.
            self.gids = np.zeros(0, dtype=np.int32)
            self.bids = np.zeros(0, dtype=np.int32)
            self.CL = np.zeros(0)
            self.has_lift = False
            self.reset_impulse()
            return
        self.gids = np.array([i["gid"] for i in self.items], dtype=np.int32)
        self.bids = np.array([i["bid"] for i in self.items], dtype=np.int32)
        self.V = np.array([i["V"] for i in self.items])
        self.A = np.array([i["A"] for i in self.items])
        self.CD = np.array([i["cd"] for i in self.items])
        self.CL = np.array([i["cl"] for i in self.items])
        self.CV = np.array([i["cv"] for i in self.items])
        # Plate normal in the geom's local frame, and the area facing it.
        face = np.array([i["face"] for i in self.items], dtype=int)
        self.NRM = np.zeros((n, 3))
        self.NRM[np.arange(n), face] = 1.0
        self.A_face = self.A[np.arange(n), face]
        self.has_lift = bool(self.lift_on and np.any(self.CL > 0))
        self.MA = np.array([i["ma"] for i in self.items])
        self.SZ = np.array([i["size"] for i in self.items])
        self.TY = np.array([int(i["gtype"]) for i in self.items])
        self.rootid = self.m.body_rootid[self.bids]
        self.is_box = self.TY == int(GT.mjGEOM_BOX)
        self.is_sph = self.TY == int(GT.mjGEOM_SPHERE)
        self.is_cap = (self.TY == int(GT.mjGEOM_CAPSULE)) | (self.TY == int(GT.mjGEOM_CYLINDER))
        self.is_ell = self.TY == int(GT.mjGEOM_ELLIPSOID)
        self._precompute_hot_path(n)
        self.reset_impulse()

    def _precompute_hot_path(self, n):
        """Everything apply() needs that does not change between steps.

        apply() runs every simulation step on a dozen or so geoms, where numpy's
        per-call overhead dwarfs the arithmetic. Constants are folded here once, and
        each geom type is handled through a plain index array instead of a boolean
        mask evaluated every step.
        """
        box = int(GT.mjGEOM_BOX)
        cap = (int(GT.mjGEOM_CAPSULE), int(GT.mjGEOM_CYLINDER))
        self.i_box = np.flatnonzero(self.TY == box)
        self.i_cap = np.flatnonzero(np.isin(self.TY, cap))
        self.i_ell = np.flatnonzero(self.TY == int(GT.mjGEOM_ELLIPSOID))
        self.SZ_box = self.SZ[self.i_box]
        self.SZ_cap = self.SZ[self.i_cap]
        self.SZ_ell = self.SZ[self.i_ell]
        # Spheres and anything unrecognized have an orientation-independent extent.
        self.hz_const = np.where(self.TY == int(GT.mjGEOM_SPHERE), self.SZ[:, 0],
                                 np.maximum(self.SZ[:, 0], 1e-4)).astype(float)
        # Folded force coefficients.
        self.K_drag = 0.5 * self.rho * self.CD * self.A           # (n, 3)
        self.K_lift = self.rho * self.CL * self.A_face            # 0.5*rho*2*cl*A
        self.K_buoy = self.rho * self.g * self.V                  # (n,)
        # Per-body aggregation: xfrc_applied rows for the bodies that own wet geoms.
        self.bodies = np.unique(self.bids)
        slot = {b: k for k, b in enumerate(self.bodies)}
        self.to_body = np.zeros((len(self.bodies), n))
        for i, b in enumerate(self.bids):
            self.to_body[slot[b], i] = 1.0
        # Added-mass weight cancellation, per body. It has to act at the body's centre
        # of mass, because that is where gravity pulls on the mass it cancels; applied
        # at a geom centre it would add a spurious torque on any body whose centre of
        # mass is not at its geom's centre.
        self.ma_weight = self.ma_body * self.g                    # (nbody,)

    # ---------- thrust bookkeeping ----------
    def reset_impulse(self):
        """Zero the running impulse totals. Call once scoring starts."""
        self.imp_drag = np.zeros(3)    # from the quadratic + viscous resistive terms
        self.imp_lift = np.zeros(3)    # from the circulatory lift term

    def _lift(self, R, v_world, f):
        """Flat-plate circulatory lift, perpendicular to the local relative flow.

        Post-stall flat-plate model: Cl(alpha) = cl * sin(2*alpha), where alpha is the
        angle between the flow and the plate's plane. At small alpha this grows like
        alpha while the resistive term grows like alpha^2, which is exactly the regime
        a foil-like stroke works in and the resistive model alone cannot reward.

        R is (n,3,3), v_world (n,3), f (n,) immersion fractions.
        """
        speed2 = (v_world * v_world).sum(1)
        speed = np.sqrt(speed2)
        live = speed > 1e-9
        v_hat = v_world * (live / np.maximum(speed, 1e-12))[:, None]
        n_world = (R * self.NRM[:, None, :]).sum(2)          # plate normal, world frame
        sin_a = (n_world * v_hat).sum(1)                     # flow angle to the plate
        cos_a = np.sqrt(np.clip(1.0 - sin_a * sin_a, 0.0, 1.0))
        # Direction: perpendicular to the flow, in the plane spanned by flow and normal.
        n_perp = n_world - sin_a[:, None] * v_hat
        n_len = np.sqrt((n_perp * n_perp).sum(1))
        l_hat = n_perp * ((n_len > 1e-9) / np.maximum(n_len, 1e-12))[:, None]
        # The minus sign makes lift oppose the plate's own normal motion.
        mag = -self.K_lift * sin_a * cos_a * speed2 * f
        return mag[:, None] * l_hat

    # ---------- called every simulation step ----------
    def apply(self, data):
        """All wet geoms in one pass. Checked against _apply_slow by the test suite."""
        xfrc = data.xfrc_applied
        xfrc[:] = 0.0
        if self.empty:
            return
        P = data.geom_xpos[self.gids]                        # (n, 3)
        R = data.geom_xmat[self.gids].reshape(-1, 3, 3)      # (n, 3, 3)

        # vertical half extent in the current orientation, per geom type
        hz = self.hz_const.copy()
        if self.i_box.size:
            hz[self.i_box] = (np.abs(R[self.i_box, 2]) * self.SZ_box).sum(1)
        if self.i_cap.size:
            hz[self.i_cap] = (np.abs(R[self.i_cap, 2, 2]) * self.SZ_cap[:, 1]
                              + self.SZ_cap[:, 0])
        if self.i_ell.size:
            hz[self.i_ell] = np.sqrt(((R[self.i_ell, 2] * self.SZ_ell) ** 2).sum(1))
        np.maximum(hz, 1e-6, out=hz)
        f = np.clip((self.water_z - P[:, 2] + hz) / (2.0 * hz), 0.0, 1.0)   # (n,)

        # velocity of each geom's own point: v = cvel_lin + omega x (p - subtree_com)
        cv = data.cvel[self.bids]
        v = cv[:, 3:6] + _cross(cv[:, 0:3], P - data.subtree_com[self.rootid])

        # anisotropic quadratic drag in the geom's local frame, then back to world
        vl = (R * v[:, :, None]).sum(1)                      # R^T v
        fl = -self.K_drag * vl * np.abs(vl) * f[:, None]
        F = (R * fl[:, None, :]).sum(2)                      # R f_local
        F -= (self.CV * f)[:, None] * v                      # linear viscous

        dt = self.m.opt.timestep
        self.imp_drag += F.sum(0) * dt
        if self.has_lift:
            F_lift = self._lift(R, v, f)
            self.imp_lift += F_lift.sum(0) * dt
            F = F + F_lift

        F[:, 2] += self.K_buoy * f                           # buoyancy
        T = _cross(P - data.xipos[self.bids], F)             # torque about the COM
        xfrc[self.bodies] = self.to_body @ np.concatenate((F, T), axis=1)
        xfrc[:, 2] += self.ma_weight                         # at the COM, no torque

    def _apply_slow(self, data):
        """Scalar reference implementation of `apply`, used by the tests.

        Deliberately written differently: one geom at a time, with velocities from
        mj_objectVelocity rather than from cvel. It is the readable statement of the
        physics, and the fast path must agree with it.
        """
        data.xfrc_applied[:] = 0.0
        res = np.zeros(6)
        for it in self.items:
            gid, bid = it["gid"], it["bid"]
            p = data.geom_xpos[gid]
            R = data.geom_xmat[gid].reshape(3, 3)
            hz = vertical_half_extent(it["gtype"], it["size"], R)
            f = np.clip((self.water_z - (p[2] - hz)) / (2 * hz), 0.0, 1.0) if hz > 1e-9 else 0.0
            F = np.zeros(3)
            if f > 0.0:
                mujoco.mj_objectVelocity(self.m, data, mujoco.mjtObj.mjOBJ_GEOM, gid, res, 0)
                v_world = res[3:6]
                v_local = R.T @ v_world
                f_local = -0.5 * it["cd"] * self.rho * it["A"] * v_local * np.abs(v_local) * f
                F += R @ f_local
                F += -it["cv"] * v_world * f
                F[2] += self.rho * self.g * it["V"] * f
                if self.lift_on and it["cl"] > 0:
                    s = float(np.linalg.norm(v_world))
                    if s > 1e-9:
                        n = R[:, it["face"]]
                        sa = float(n @ v_world) / s
                        ca = np.sqrt(max(0.0, 1.0 - sa * sa))
                        n_perp = n - sa * v_world / s
                        nl = float(np.linalg.norm(n_perp))
                        if nl > 1e-9:
                            F += (-0.5 * self.rho * 2.0 * it["cl"] * sa * ca
                                  * it["A"][it["face"]] * s * s * f) * (n_perp / nl)
            data.xfrc_applied[bid, :3] += F
            data.xfrc_applied[bid, 3:] += np.cross(p - data.xipos[bid], F)
        # added-mass weight cancellation, at each body's centre of mass
        data.xfrc_applied[:, 2] += self.ma_body * self.g

    # ---------- diagnostics ----------
    def summary(self):
        tot_v = sum(i["V"] for i in self.items)
        tot_ma = sum(i["ma"] for i in self.items)
        tot_m = float(np.sum(self.m.body_mass[1:]))
        if self.has_lift:
            lifting = [i["name"] for i in self.items if i["cl"] > 0]
            mode = (f"resistive + lift on {len(lifting)} geoms "
                    f"(cl up to {self.CL.max():.2f})")
        elif self.lift_on:
            mode = "resistive only (lift enabled but every cl is 0)"
        else:
            mode = "resistive only (no lift term)"
        lines = [f"[hydro] geoms={len(self.items)} volume={tot_v*1e6:.1f}cm3 "
                 f"fully-submerged buoyancy={self.rho*tot_v*1000:.1f}g of water | "
                 f"added mass={tot_ma*1000:.1f}g | "
                 f"total mass incl. added={tot_m*1000:.1f}g",
                 f"[hydro] propulsion model: {mode}"]
        if self.skipped_visual:
            lines.append(f"[hydro] skipped {len(self.skipped_visual)} visual-only geoms "
                         f"that duplicate a collision geom on the same body "
                         f"(they would double every fluid force)")
        return "\n".join(lines)
