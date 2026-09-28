"""A falsifiable, non-novel sticking-contact reference in image coordinates.

This is not a world-model revision. RGB supplies a geometric reference point,
not a measured centre of mass. The scalar isotropic completion is an extra
assumption, not a consequence of non-penetration or of rigid-body kinematics.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def contact_geometry(object_mask, pusher_center, pusher_radius, reference):
    """Nearest *visible* object pixel; occlusion/raster error remains explicit."""
    points = np.argwhere(object_mask)[:, ::-1].astype(float)
    if not len(points):
        raise ValueError("no visible object")
    pusher_center = np.asarray(pusher_center, dtype=float)
    distances = np.linalg.norm(points - pusher_center, axis=1)
    nearest = points[np.argmin(distances)]
    return {"point": nearest, "r": nearest - np.asarray(reference),
            "gap": float(distances.min() - pusher_radius)}


def sticking_completion(r, pusher_displacement, ell_squared):
    """min ||v||² + ell² omega² subject to v + omega J r = delta_p.

    Uses displacement over one observed interval; finite-rotation and changing
    contact errors are NOT removed by the analytic minimizer.
    """
    r, dp = np.asarray(r, dtype=float), np.asarray(pusher_displacement, dtype=float)
    if r.shape != (2,) or dp.shape != (2,) or not np.isfinite([*r, *dp, ell_squared]).all():
        raise ValueError("expected finite 2D inputs")
    if ell_squared < 0 or ell_squared + r @ r <= 0:
        raise ValueError("completion is not uniquely defined")
    omega = (r[0] * dp[1] - r[1] * dp[0]) / (ell_squared + r @ r)
    velocity = dp - omega * np.array([-r[1], r[0]])
    return velocity, float(omega)


def fit_sticking_scale(rows):
    """Rows must already be restricted by past image geometry, never residuals.

    Least-squares scalar is a classical mechanics reference, not a claimed new
    fitting principle. Return unavailable when completed data has no rotation.
    """
    if not rows:
        return {"status": "no_prefix_contact", "count": 0}
    omega = np.asarray([row["theta"] for row in rows])
    r = np.asarray([[row["rx"], row["ry"]] for row in rows])
    dp = np.asarray([[row["dpx"], row["dpy"]] for row in rows])
    rhs = r[:, 0] * dp[:, 1] - r[:, 1] * dp[:, 0] - (r*r).sum(1)*omega
    if not np.isfinite(np.concatenate((omega, rhs))).all():
        raise ValueError("nonfinite observed geometry")
    denominator = float(omega @ omega)
    if denominator <= 1e-12:
        return {"status": "no_resolved_prefix_rotation", "count": len(rows),
                "sum_theta_squared": denominator}
    unconstrained = float(omega @ rhs / denominator)
    ell_squared = max(0., unconstrained)
    return {"status": "fit", "count": len(rows), "ell_squared": ell_squared,
            "unconstrained_ell_squared": unconstrained,
            "nonnegative_boundary_active": unconstrained < 0,
            "sum_theta_squared": denominator,
            "prefix_relation_rmse": float(np.sqrt(np.mean((rhs-ell_squared*omega)**2)))}


def _finite_vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return result


def _roundoff(*values):
    """Floating arithmetic check only; not a contact/friction threshold."""
    return 64*np.finfo(np.float64).eps*max(1., *(float(np.abs(v).max()) for v in values))


def _kinematic_roundoff(J, M, force, dp, normal, tangent):
    """Displacement error scales, distinct from force/cone tolerances.

    The represented J M J^T solve and J (M J^T f) evaluation can cancel
    large intermediate products. Final displacement alone is not their
    roundoff scale. Keep the existing 64-epsilon factor, not a physical slack.
    Projection bounds include the absolute normal/tangent dot-product inputs.
    """
    scale = np.abs(J)@(np.abs(M)@(np.abs(J.T)@np.abs(force)))+np.abs(dp)
    if not np.isfinite(scale).all():
        raise FloatingPointError("nonfinite kinematic roundoff scale")
    factor = 64*np.finfo(np.float64).eps
    return (factor*np.maximum(1., scale),
            factor*max(1., float(np.abs(normal)@scale)),
            factor*max(1., float(np.abs(tangent)@scale)))


class UnsupportedContactGeometry(ValueError):
    """A conservative sufficient-support check failed, not non-identifiability."""


class SingleContactScene:
    """Source-mask raster signed distance, transformed by a rigid pose.

    Positive distance is outside the visible source object. Pixel centres and
    bilinear interpolation define the geometry; this is not an exact shape
    reconstruction. Pose rotates about ``reference`` then translates it.
    By default the diagonal mobility assumes this reference is its response
    centre; an explicit body-fixed response offset changes that assumption.
    No padding, clipped queries, or zero-gradient normal fallback is used.

    Unknown pixels remain unknown. Known occupied versus occupied-or-unknown
    SDF agreement is only a conservative sufficient support check. The filled
    extreme can violate pusher non-penetration and need not be a physically
    feasible completion; disagreement is not an impossibility theorem about
    the true shape, contact, or action prediction.
    """

    def __init__(self, mask, reference, pusher_radius, unknown_mask=None):
        array = np.asarray(mask)
        if (array.ndim != 2 or min(array.shape) < 3 or not np.isfinite(array).all()
                or not np.isin(array, (0, 1)).all()):
            raise ValueError("mask must be a finite binary image of at least 3 by 3")
        array = array.astype(bool)
        if not array.any() or array.all():
            raise ValueError("signed distance requires both object and background pixels")
        if not np.isfinite(pusher_radius) or pusher_radius < 0:
            raise ValueError("pusher radius must be finite and nonnegative")
        self.reference = _finite_vector(reference, 2, "reference").copy()
        self.pusher_radius = float(pusher_radius)
        self.sdf = distance_transform_edt(~array)-distance_transform_edt(array)
        grad_y, grad_x = np.gradient(self.sdf)
        self.gradient = np.stack((grad_x, grad_y), axis=-1)
        unknown = np.zeros_like(array) if unknown_mask is None else np.asarray(unknown_mask)
        if (unknown.shape != array.shape or not np.isfinite(unknown).all()
                or not np.isin(unknown, (0, 1)).all()):
            raise ValueError("unknown_mask must be a finite binary image matching mask")
        unknown = unknown.astype(bool)
        if np.any(unknown & array):
            raise ValueError("known occupied and unknown masks must be disjoint")
        self.has_unknown = bool(unknown.any())
        possible = array | unknown
        if possible.all():
            raise ValueError("possible object must leave known background in image support")
        self.possible_sdf = distance_transform_edt(~possible)-distance_transform_edt(possible)
        grad_y, grad_x = np.gradient(self.possible_sdf)
        self.possible_gradient = np.stack((grad_x, grad_y), axis=-1)
        self.unknown_distance = distance_transform_edt(~unknown) if self.has_unknown else None

    def _interpolate(self, field, point):
        x, y = point
        height, width = self.sdf.shape
        if not (0 <= x <= width-1 and 0 <= y <= height-1):
            raise ValueError("signed-distance query is outside source image support")
        ix, iy = min(int(np.floor(x)), width-2), min(int(np.floor(y)), height-2)
        fx, fy = x-ix, y-iy
        return ((1-fy)*((1-fx)*field[iy, ix]+fx*field[iy, ix+1])
                + fy*((1-fx)*field[iy+1, ix]+fx*field[iy+1, ix+1]))

    def geometry(self, pose, pusher_xy, *, with_normal=True, contact_reach=0.):
        """Return signed gap; optionally world contact point/r/inward normal.

        A distant or stationary integrator can request only the gap: a normal
        that will not be used need not be defined. Contact queries fail if the
        interpolated SDF gradient is zero. The point is a local SDF footpoint,
        not a privileged contact observation.

        ``possible_gap`` uses occupied-or-unknown for a conservative far-space
        check; ``unknown_proximity`` is raster distance to unknown minus pusher
        radius (None without unknown). A full normal query within contact_reach
        requires both extremes to agree to floating arithmetic precision.
        """
        pose = _finite_vector(pose, 3, "pose")
        pusher = _finite_vector(pusher_xy, 2, "pusher")
        if not np.isfinite(contact_reach) or contact_reach < 0:
            raise ValueError("contact reach must be finite and nonnegative")
        cosine, sine = np.cos(pose[2]), np.sin(pose[2])
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        current_reference = self.reference+pose[:2]
        source_pusher = self.reference+rotation.T@(pusher-current_reference)
        distance = float(self._interpolate(self.sdf, source_pusher))
        possible_distance = float(self._interpolate(self.possible_sdf, source_pusher))
        possible_gap = possible_distance-self.pusher_radius
        proximity = (float(self._interpolate(self.unknown_distance, source_pusher))-self.pusher_radius
                     if self.has_unknown else None)
        result = {"gap": distance-self.pusher_radius, "possible_gap": possible_gap,
                  "unknown_proximity": proximity,
                  "geometry_support": "known_only" if not self.has_unknown else
                      ("possible_material_out_of_reach" if possible_gap > contact_reach else "normal_not_checked")}
        if not with_normal:
            return result
        gradient = self._interpolate(self.gradient, source_pusher)
        length = float(np.linalg.norm(gradient))
        if not np.isfinite(length) or length == 0.:
            raise ValueError("contact normal is undefined: zero/nonfinite SDF gradient")
        inward_source = -gradient/length
        point_source = source_pusher+distance*inward_source
        r = rotation@(point_source-self.reference)
        result.update(point=current_reference+r, r=r, normal=rotation@inward_source)
        if self.has_unknown and possible_gap <= contact_reach:
            possible_gradient = self._interpolate(self.possible_gradient, source_pusher)
            possible_length = float(np.linalg.norm(possible_gradient))
            if not np.isfinite(possible_length) or possible_length == 0.:
                raise UnsupportedContactGeometry("unknown-filled contour has undefined contact normal")
            possible_inward = -possible_gradient/possible_length
            possible_point = source_pusher+possible_distance*possible_inward
            if (abs(distance-possible_distance) > _roundoff(distance, possible_distance)
                    or np.max(np.abs(point_source-possible_point)) > _roundoff(point_source, possible_point)
                    or np.max(np.abs(inward_source-possible_inward)) > _roundoff(inward_source, possible_inward)):
                raise UnsupportedContactGeometry("unknown pixels alter reachable signed gap, contact point or normal")
            result["geometry_support"] = "known_and_unknown_filled_agree"
        return result


def single_contact_mode(r, n, dp, ell_squared, mu):
    """Single quasi-static Coulomb-contact displacement reference, world frame.

    M=diag(1,1,1/ell_squared), J=[I, ninety_degree_rotation(r)].
    f=lambda*n+tau*t is a model-derived normalized coefficient, not measured
    privileged force. A positive t.(dp-J*twist) requires positive tau at slide.
    This local, isotropic-mobility assumption is not a complete simulator.
    """
    r, n, dp = (_finite_vector(value, 2, name) for value, name in ((r, "r"), (n, "normal"), (dp, "dp")))
    if not np.isfinite([ell_squared, mu]).all() or ell_squared <= 0 or mu < 0:
        raise ValueError("ell_squared must be positive and mu nonnegative, both finite")
    if abs(float(n@n)-1.) > _roundoff(n):
        raise ValueError("inward normal must have unit length")
    tangent = np.array([-n[1], n[0]])
    J = np.column_stack((np.eye(2), [-r[1], r[0]]))
    M = np.diag([1., 1., 1./ell_squared])
    W = J@M@J.T
    closing = float(n@dp)

    def result(mode, force):
        twist = M@J.T@force
        contact = J@twist
        normal_force, tangential_force = float(n@force), float(tangent@force)
        slip = float(tangent@(dp-contact))
        cone_violation = max(0., -normal_force, abs(tangential_force)-mu*normal_force)
        normal_residual = float(n@(contact-dp))
        if not np.isfinite(np.r_[twist, force, contact, slip, cone_violation, normal_residual]).all():
            raise FloatingPointError("nonfinite single-contact solution")
        tolerance = _roundoff(dp, contact, force)
        if mode != "separate":
            component_tolerance, normal_tolerance, tangent_tolerance = _kinematic_roundoff(
                J, M, force, dp, n, tangent)
            if (normal_force < -tolerance or cone_violation > tolerance
                    or abs(normal_residual) > normal_tolerance):
                raise ArithmeticError("single-contact force cone or normal kinematics failed")
            if mode == "stick" and np.any(np.abs(contact-dp) > component_tolerance):
                raise ArithmeticError("sticking kinematics failed")
            if mode.startswith("slide"):
                sign = 1. if mode == "slide+" else -1.
                if (abs(tangential_force-sign*mu*normal_force) > tolerance
                        or sign*slip < -tangent_tolerance):
                    raise ArithmeticError("sliding cone edge or slip direction failed")
        return {"mode": mode, "twist": twist, "force": force,
                "contact_displacement": contact, "normal_force": normal_force,
                "tangential_force": tangential_force, "tangential_slip": slip,
                "normal_kinematic_residual": normal_residual,
                "force_cone_violation": cone_violation}

    if closing <= 0:
        return result("separate", np.zeros(2))
    stick_force = np.linalg.solve(W, dp)
    # Independently validate the linear solve; an inaccurate solve must not be
    # hidden by the subsequent composed-product displacement error bound.
    solve_residual = W@stick_force-dp
    solve_scale = np.abs(W)@np.abs(stick_force)+np.abs(dp)
    if not np.isfinite(np.r_[stick_force, solve_residual, solve_scale]).all():
        raise FloatingPointError("nonfinite single-contact linear solve")
    if np.any(np.abs(solve_residual) > 64*np.finfo(np.float64).eps*np.maximum(1., solve_scale)):
        raise ArithmeticError("single-contact linear solve backward error failed")
    normal_force, tangential_force = float(n@stick_force), float(tangent@stick_force)
    if normal_force >= 0 and abs(tangential_force) <= mu*normal_force+_roundoff(stick_force):
        return result("stick", stick_force)
    admissible = []
    for sign, mode in ((1., "slide+"), (-1., "slide-")):
        ray = n+sign*mu*tangent
        denominator = float(n@W@ray)
        if denominator <= 0:
            continue  # This cone edge cannot supply positive normal closing.
        force = closing/denominator*ray
        contact = W@force
        slip = float(tangent@(dp-contact))
        if sign*slip >= -_roundoff(dp, contact, force):
            admissible.append(result(mode, force))
    if not admissible:
        raise ArithmeticError("no admissible Coulomb single-contact mode")
    if len(admissible) != 1:
        raise ArithmeticError("ambiguous Coulomb single-contact mode; no mode fallback")
    return admissible[0]


def integrate_contact_path(scene, initial_pose, pusher_path, ell_squared, mu, max_spatial_step=.5,
                           *, response_offset_body=None):
    """Return (poses_at_input_path_points, diagnostics), including the start.

    Each command segment is split into <=max_spatial_step pixel increments.
    Outside contact, only the fraction after a local normal gap crossing acts.
    This is a first-order small-rotation/contact approximation, not exact event
    detection or a full multi-contact/inertial/penetration-repair simulator.
    The default .5 px discretization is to be stress checked at .25 px; it is
    not a fitted material coefficient. Penetration is measured, never repaired.

    Optional response_offset_body is d=o-c in the source object's body frame:
    the fixed reference o minus an effective response centre c. At each step
    r_c=r_o+R(theta)d enters the SAME mode solver. Centre translation v and
    rotation omega give reference translation v+(R(omega)-I)R(theta)d.
    None preserves the original arithmetic and diagnostic fields exactly;
    explicit zero preserves the arithmetic but records the supplied parameter.
    Reference-change covariance concerns propagation with given parameters,
    not the completed-data estimator or its fixed-reference fitting metric.
    """
    if not isinstance(scene, SingleContactScene):
        raise TypeError("scene must be a SingleContactScene")
    pose = _finite_vector(initial_pose, 3, "initial pose").copy()
    offset = None if response_offset_body is None else _finite_vector(
        response_offset_body, 2, "response_offset_body").copy()
    shifted_center = offset is not None and bool(np.any(offset != 0.))
    path = np.asarray(pusher_path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 1 or not np.isfinite(path).all():
        raise ValueError("pusher path must be a nonempty finite N-by-2 array")
    if (not np.isfinite([ell_squared, mu, max_spatial_step]).all()
            or ell_squared <= 0 or mu < 0 or max_spatial_step <= 0):
        raise ValueError("invalid contact coefficient or spatial discretization")
    initial_geometry = scene.geometry(pose, path[0], with_normal=False)
    gap = initial_geometry["gap"]
    initial_penetration = max(0., -gap)
    diagnostics = {"initial_penetration": initial_penetration,
        "ell_squared": float(ell_squared), "mu": float(mu),
        "max_current_penetration": initial_penetration, "final_penetration": initial_penetration,
        "mode_counts": {name: 0 for name in ("free_space", "stationary", "separate", "stick", "slide+", "slide-")},
        "substeps": 0, "normal_gap_crossings": 0, "max_spatial_step": float(max_spatial_step),
        "max_realized_spatial_step": 0., "endpoint_signed_gaps": [gap],
        "endpoint_possible_gaps": [initial_geometry["possible_gap"]],
        "unknown_mask_present": scene.has_unknown, "unknown_supported_contact_steps": 0,
        "minimum_unknown_proximity": initial_geometry["unknown_proximity"],
        "penetration_repair": False,
        "interpretation": "raster single-contact quasi-static reference; not a complete physical simulator"}
    poses = [pose.copy()]
    for start, end in zip(path[:-1], path[1:]):
        displacement = end-start
        distance = float(np.linalg.norm(displacement))
        count = max(1, int(np.ceil(distance/max_spatial_step)))
        dp = displacement/count
        step_length = float(np.linalg.norm(dp))
        diagnostics["max_realized_spatial_step"] = max(diagnostics["max_realized_spatial_step"], step_length)
        for index in range(count):
            pusher = start+(index/count)*displacement
            next_pusher = start+((index+1)/count)*displacement
            gap_geometry = scene.geometry(pose, pusher, with_normal=False, contact_reach=step_length)
            gap = gap_geometry["gap"]
            diagnostics["max_current_penetration"] = max(diagnostics["max_current_penetration"], -gap)
            if distance == 0:
                mode = "stationary"
            elif gap_geometry["possible_gap"] > step_length:
                mode = "free_space"  # No normal is needed for this distant step.
            else:
                geometry = scene.geometry(pose, pusher, contact_reach=step_length)
                if geometry["geometry_support"] == "known_and_unknown_filled_agree":
                    diagnostics["unknown_supported_contact_steps"] += 1
                closing = float(geometry["normal"]@dp)
                if gap > 0 and closing <= gap:
                    mode = "free_space"
                else:
                    active_dp = dp
                    if gap > 0:
                        fraction = gap/closing
                        active_dp = (1-fraction)*dp
                        geometry = scene.geometry(pose, pusher+fraction*dp, contact_reach=step_length)
                        diagnostics["normal_gap_crossings"] += 1
                    if shifted_center:
                        cosine, sine = np.cos(pose[2]), np.sin(pose[2])
                        offset_world = np.array([[cosine, -sine], [sine, cosine]])@offset
                        response_lever = geometry["r"]+offset_world
                    else:
                        response_lever = geometry["r"]
                    contact = single_contact_mode(response_lever, geometry["normal"], active_dp, ell_squared, mu)
                    mode = contact["mode"]
                    if shifted_center:
                        reference_increment = contact["twist"].copy()
                        omega = reference_increment[2]
                        cosine, sine = np.cos(omega), np.sin(omega)
                        reference_increment[:2] += np.array(
                            [[cosine-1., -sine], [sine, cosine-1.]])@offset_world
                        pose += reference_increment
                    else:
                        pose += contact["twist"]
                    if not np.isfinite(pose).all():
                        raise FloatingPointError("nonfinite integrated contact pose")
            diagnostics["mode_counts"][mode] += 1
            diagnostics["substeps"] += 1
            endpoint_geometry = scene.geometry(pose, next_pusher, with_normal=False)
            gap = endpoint_geometry["gap"]
            if scene.has_unknown:
                diagnostics["minimum_unknown_proximity"] = min(
                    diagnostics["minimum_unknown_proximity"], endpoint_geometry["unknown_proximity"])
            diagnostics["max_current_penetration"] = max(diagnostics["max_current_penetration"], -gap)
        diagnostics["endpoint_signed_gaps"].append(gap)
        diagnostics["endpoint_possible_gaps"].append(endpoint_geometry["possible_gap"])
        poses.append(pose.copy())
    diagnostics["final_penetration"] = max(0., -gap)
    if offset is not None:
        world_offsets = []
        for value in poses:
            cosine, sine = np.cos(value[2]), np.sin(value[2])
            world_offsets.append(np.array([[cosine, -sine], [sine, cosine]])@offset)
        centers = scene.reference+np.asarray(poses)[:, :2]-np.asarray(world_offsets)
        diagnostics.update(response_offset_body=offset.tolist(),
                           response_offset_world_endpoints=np.asarray(world_offsets).tolist(),
                           response_center_endpoints=centers.tolist())
    return np.asarray(poses), diagnostics
