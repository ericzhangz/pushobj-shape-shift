"""RGB-only rigid-scene representation probes and a source-only image orbit.

The palette rule is specific to the observed PushObj RGB renderer, not a shape
template.  It separates the gray-blue object from the blue pusher and green
goal.  No simulator state, known object dimensions, or goal pose is consulted.
Registration sees BOTH images and therefore is not a forward predictor.
The separately named ``RgbSceneRenderer`` instead consumes source-only assets,
an already fitted pixel calibration and predicted poses; it never reads target
images.  This rendering prior supplies no object-motion or contact dynamics.

An occluded silhouette is not a complete object template.  Registration exposes
its geometric residual, coverage and overlap; repainting cannot reconstruct
object/background pixels hidden in its source.  In particular, a fitted SE(2)
transform is a visible-geometry reference, NOT a center-of-mass estimate.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as torch_functional


def _rgb(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("expected uint8 RGB image [height,width,3]")
    return image


def _largest(mask, minimum, name):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        raise ValueError(f"{name} not visible in RGB")
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[index, cv2.CC_STAT_AREA] < minimum:
        raise ValueError(f"insufficient visible {name} pixels")
    return labels == index


def observe_rgb(image):
    """Return visible masks, image-coordinate geometry and an explicit backdrop.

    Thresholds separate the observed palette by channel ordering, rather than
    matching a shape or exact RGB triplet.  They are fixed for this probe; other
    renderers/lighting require a new observation contract.  A missing object or
    pusher raises, rather than manufacturing a pose.  ``background_rgb`` replaces
    the occupied pixels by the image's modal color; these are unknown pixels,
    not claimed recovered background. ``background_unknown_mask`` marks them.
    """
    image = _rgb(image)
    red, green, blue = image.astype(np.int16).transpose(2, 0, 1)
    object_mask = _largest(
        (green - red >= 6) & (blue - green >= 6)
        & (blue - red <= 90) & (red >= 50) & (red <= 220),
        40, "object")
    pusher_mask = _largest(
        (blue - red >= 65) & (blue - green >= 65), 8, "pusher")
    py, px = np.nonzero(pusher_mask)
    oy, ox = np.nonzero(object_mask)
    colors, counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
    backdrop_color = colors[int(np.argmax(counts))]
    unknown = object_mask | pusher_mask
    background = image.copy()
    background[unknown] = backdrop_color
    return {
        "object_mask": object_mask,
        "pusher_mask": pusher_mask,
        "object_centroid": np.array([ox.mean(), oy.mean()]),
        "pusher_center": np.array([px.mean(), py.mean()]),
        "pusher_radius": float(np.sqrt(pusher_mask.sum() / np.pi)),
        "object_visible_area": int(object_mask.sum()),
        "background_rgb": background,
        "background_unknown_mask": unknown,
        "backdrop_color": backdrop_color,
    }


def _boundary(observation):
    mask = observation["object_mask"].astype(np.uint8)
    edge = mask & (1 - cv2.erode(mask, np.ones((3, 3), np.uint8)))
    # Exclude the artificial boundary carved by the foreground pusher.  This
    # does not infer the hidden object contour, and its extent is reported.
    occluded_edge = cv2.dilate(observation["pusher_mask"].astype(np.uint8),
                               np.ones((7, 7), np.uint8))
    valid = edge.astype(bool) & ~occluded_edge.astype(bool)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = np.concatenate([contour[:, 0, :] for contour in contours])
    points = points[valid[points[:, 1], points[:, 0]]].astype(np.float64)
    if len(points) < 12:
        raise ValueError("insufficient unoccluded object boundary for registration")
    # Deterministic subsampling only limits this diagnostic's CPU cost.
    # Sample along contour order, not raster order: raster-striding a rectangle
    # can accidentally retain only its left side and discard its entire right.
    points = points[np.linspace(0, len(points) - 1, min(240, len(points))).astype(int)]
    return points, float(valid.sum() / max(1, edge.sum()))


def _rotation(theta):
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array([[cosine, -sine], [sine, cosine]])


def register_pair(source, target):
    """Fit a rigid source->target transform from observed visible silhouettes.

    Four rotation starts prevent centroid/PCA ambiguity from silently selecting
    one orientation.  The symmetric boundary objective trims its worst 10% to
    tolerate visibility changes, but UNTRIMMED residuals and coverage are also
    returned.  Finite residuals do not certify unique pose or forward validity.
    Pixels x right/y down imply positive theta is clockwise on the image.
    """
    source, target = _rgb(source), _rgb(target)
    if source.shape != target.shape:
        raise ValueError("registration image shapes differ")
    src, dst = observe_rgb(source), observe_rgb(target)
    points, source_visible = _boundary(src)
    targets, target_visible = _boundary(dst)
    center, target_center = src["object_centroid"], dst["object_centroid"]
    source_tree, target_tree = cKDTree(points), cKDTree(targets)

    def distances(parameters):
        rotation = _rotation(parameters[0])
        destination = target_center + parameters[1:]
        transformed = (points - center) @ rotation.T + destination
        inverse = (targets - destination) @ rotation + center
        return (target_tree.query(transformed)[0], source_tree.query(inverse)[0])

    def objective(parameters):
        residual = np.concatenate(distances(parameters))
        keep = max(1, int(np.ceil(0.9 * len(residual))))
        return float(np.mean(np.partition(residual * residual, keep - 1)[:keep]))

    if np.array_equal(src["object_mask"], dst["object_mask"]):
        parameters = np.array([0., *(center - target_center)])
        alternative_gap = 0.0
        optimizer_converged = True
    else:
        fits = [minimize(objective, np.array([angle, 0., 0.]), method="Powell",
                         options={"maxiter": 100, "xtol": 1e-5, "ftol": 1e-6})
                for angle in (0., np.pi / 2, -np.pi / 2, np.pi)]
        fits.sort(key=lambda fit: (fit.fun, abs(np.arctan2(
            np.sin(fit.x[0]), np.cos(fit.x[0])))))
        if not np.isfinite(fits[0].fun) or not np.isfinite(fits[0].x).all():
            raise FloatingPointError("nonfinite rigid registration solution")
        parameters = fits[0].x
        alternative_gap = float(fits[1].fun - fits[0].fun)
        optimizer_converged = bool(fits[0].success)
    rotation = _rotation(parameters[0])
    translation = target_center + parameters[1:] - rotation @ center
    affine = np.column_stack([rotation, translation])
    moved_mask = cv2.warpAffine(src["object_mask"].astype(np.uint8), affine,
                               (source.shape[1], source.shape[0]),
                               flags=cv2.INTER_NEAREST).astype(bool)
    moved_pusher = cv2.warpAffine(src["pusher_mask"].astype(np.uint8), affine,
                                 (source.shape[1], source.shape[0]),
                                 flags=cv2.INTER_NEAREST).astype(bool)
    evaluated = ~(moved_pusher | dst["pusher_mask"])
    intersection = (moved_mask & dst["object_mask"] & evaluated).sum()
    union = ((moved_mask | dst["object_mask"]) & evaluated).sum()
    residuals = np.concatenate(distances(parameters))
    return {
        "affine": affine,
        "theta_radians": float(np.arctan2(rotation[1, 0], rotation[0, 0])),
        "chamfer_rmse_px": float(np.sqrt(np.mean(residuals ** 2))),
        "trimmed_chamfer_rmse_px": float(np.sqrt(objective(parameters))),
        "silhouette_iou": float(intersection / max(1, union)),
        "visible_point_fraction": float(np.mean(residuals <= 2.0)),
        "source_boundary_fraction": source_visible,
        "target_boundary_fraction": target_visible,
        "alternative_start_objective_gap": alternative_gap,
        "optimizer_converged": optimizer_converged,
        "uses_observed_target": True,
    }


def repaint_pair(source, target, registration, background="source"):
    """Retrospective closure: transform source pixels and paste target pusher.

    ``source`` is the stricter source-background closure.  The optional
    ``retrospective_target`` mode uses the observed target outside its moving
    entities, isolating object-template limitations from static-background
    reveal. Neither mode predicts a new action: the transform AND pusher come
    from the observed target. Hidden source-object pixels remain missing.
    """
    source, target = _rgb(source), _rgb(target)
    if source.shape != target.shape:
        raise ValueError("repaint image shapes differ")
    src, dst = observe_rgb(source), observe_rgb(target)
    if background == "source":
        canvas = src["background_rgb"].astype(np.float64)
    elif background == "retrospective_target":
        canvas = dst["background_rgb"].astype(np.float64)
    else:
        raise ValueError("background must be source or retrospective_target")
    affine = np.asarray(registration["affine"], dtype=np.float64)
    if (affine.shape != (2, 3) or not np.isfinite(affine).all()
            or not np.allclose(affine[:, :2].T @ affine[:, :2], np.eye(2), atol=1e-6)
            or not np.isclose(np.linalg.det(affine[:, :2]), 1., atol=1e-6)):
        raise ValueError("expected finite proper SE(2) affine transform")
    size = (source.shape[1], source.shape[0])
    alpha = src["object_mask"].astype(np.float64)
    moved_alpha = cv2.warpAffine(alpha, affine, size, flags=cv2.INTER_LINEAR)
    foreground = cv2.warpAffine(source.astype(np.float64) * alpha[..., None],
                                affine, size, flags=cv2.INTER_LINEAR)
    canvas = canvas * (1. - moved_alpha[..., None]) + foreground
    canvas[dst["pusher_mask"]] = target[dst["pusher_mask"]]
    return np.clip(np.rint(canvas), 0, 255).astype(np.uint8)


def fit_pusher_pixel_map(completed_rgb, completed_raw_xy):
    """Fit raw self-position -> RGB pusher center using completed observations.

    The caller is responsible for the chronological completed-evidence cutoff.
    Only the supplied RGB and raw pusher xy enter the affine calibration.  Three
    independent design columns are required; collinear evidence fails rather
    than substituting known renderer scale/coordinates. Pixel quantization and
    occlusion can make the calibration imperfect, so residuals are returned.
    ``matrix`` maps row vectors ``[raw_x, raw_y, 1]`` to image ``[x, y]``.
    """
    images = np.asarray(completed_rgb)
    xy = np.asarray(completed_raw_xy, dtype=np.float64)
    if (images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8
            or xy.shape != (len(images), 2) or len(images) < 3
            or not np.isfinite(xy).all()):
        raise ValueError("expected >=3 completed RGB frames and finite raw xy [N,2]")
    centers = np.stack([observe_rgb(image)["pusher_center"] for image in images])
    design = np.column_stack([xy, np.ones(len(xy))])
    matrix, _, rank, singular = np.linalg.lstsq(design, centers, rcond=None)
    if rank != 3:
        raise ValueError("pusher pixel calibration requires full rank 3 evidence")
    residual = design @ matrix - centers
    return {
        "matrix": matrix,
        "rank": int(rank),
        "rmse_px": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
        "max_error_px": float(np.linalg.norm(residual, axis=1).max()),
        "condition_number": float(singular[0] / singular[-1]),
        "n_completed_observations": len(images),
    }


def fit_pusher_similarity_map(completed_rgb, completed_raw_xy):
    """D-only proper similarity calibration, not a known camera convention.

    Fits x_image = s R x_self + b with s>0, det(R)=1 by centered least
    squares. Equal image-axis scale and no reflection/shear are explicit extra
    assumptions. No simulator scale, query image or object response is used.
    Unlike the affine model, two distinct source positions identify this
    restricted map even on a line; that is conditional on the similarity prior.
    """
    images = np.asarray(completed_rgb)
    xy = np.asarray(completed_raw_xy, dtype=np.float64)
    if (images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8
            or xy.shape != (len(images), 2) or len(images) < 3
            or not np.isfinite(xy).all()):
        raise ValueError("expected >=3 completed RGB frames and finite raw xy [N,2]")
    centers = np.stack([observe_rgb(image)["pusher_center"] for image in images])
    source, target = xy-xy.mean(0), centers-centers.mean(0)
    denominator = float(np.square(source).sum())
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("similarity calibration requires completed position variation")
    a = float((source*target).sum()/denominator)
    b = float((source[:, 0]*target[:, 1]-source[:, 1]*target[:, 0]).sum()/denominator)
    scale = float(np.hypot(a, b))
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("no positive-scale proper similarity solution")
    linear = np.array([[a, b], [-b, a]])  # row-vector convention
    matrix = np.vstack((linear, centers.mean(0)-xy.mean(0)@linear))
    residual = np.column_stack((xy, np.ones(len(xy))))@matrix-centers
    singular = np.linalg.svd(source, compute_uv=False)
    return {"matrix": matrix, "family": "proper_similarity", "scale": scale,
        "angle_radians": float(np.arctan2(b, a)), "parameter_count": 4,
        "rmse_px": float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        "max_error_px": float(np.linalg.norm(residual, axis=1).max()),
        "source_centered_singular_values": singular,
        "source_squared_spread": denominator, "n_completed_observations": len(images),
        "assumption": "equal pixel-axis scale, proper rotation, translation; no shear/reflection"}


def fuse_completed_object_assets(completed_rgb):
    """Transport actually visible history pixels into the last completed frame.

    ``completed_rgb[N,H,W,3]`` must end at the CURRENT image. Its chronological
    cutoff is the caller's responsibility: RGB alone does not encode timestamps.
    Each supplied image is aligned directly to that last image by the existing
    SE(2) ``register_pair``; no goal, future image, known shape, convex hull or
    hole-filling template is accepted. Nearest-neighbor transport preserves an
    observed texture sample instead of extrapolating an unseen object surface.

    The current visible silhouette and its colors are authoritative. History
    may add object pixels ONLY under the current observed pusher mask. Warped
    pixels contradicting currently visible non-object space are reported but
    not added; this prevents registration error from silently expanding the
    outer silhouette. Under the pusher, disagreement between historical visible
    object/non-object observations is also reported, marked unknown and omitted
    from the fused mask. A filled pixel uses the latest positive observation.

    ``unknown_occlusion_mask`` marks current pusher pixels never observed or
    historically contradictory after registration. It is not an inferred object
    mask, and absence from ``object_mask`` there must NOT be interpreted as a
    physical notch/free space. Uncertainty outside the pusher, subpixel palette
    boundaries, registration bias and parts outside the image are not resolved.
    Per-frame residuals/conflicts are exposed; no residual threshold is hidden.
    The pivot remains the visible current centroid, NOT a recovered COM.
    """
    images = np.asarray(completed_rgb)
    if (images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8
            or len(images) < 1):
        raise ValueError("expected nonempty completed uint8 RGB history [N,H,W,3]")
    current = images[-1]
    observed = observe_rgb(current)
    shape = current.shape[:2]
    size = (shape[1], shape[0])
    positive_count = np.zeros(shape, dtype=np.int32)
    negative_count = np.zeros(shape, dtype=np.int32)
    latest_texture = np.zeros_like(current)
    source_frame = np.full(shape, -1, dtype=np.int32)
    conflicts = np.zeros(shape, dtype=bool)
    rows = []
    for index, image in enumerate(images):
        observation = observe_rgb(image)
        registration = register_pair(image, current)
        affine = registration["affine"]

        def warp(mask):
            return cv2.warpAffine(mask.astype(np.uint8), affine, size,
                                  flags=cv2.INTER_NEAREST).astype(bool)

        positive = warp(observation["object_mask"])
        valid = warp(np.ones(shape, dtype=np.uint8))
        hidden = warp(observation["pusher_mask"])
        negative = valid & ~positive & ~hidden
        positive_count += positive
        negative_count += negative
        texture = cv2.warpAffine(image, affine, size, flags=cv2.INTER_NEAREST)
        latest_texture[positive] = texture[positive]
        source_frame[positive] = index
        current_conflict = positive & ~observed["object_mask"] & ~observed["pusher_mask"]
        conflicts |= current_conflict
        rows.append({"frame_index": index,
                     **{key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in registration.items()},
                     "transported_visible_object_pixels": int(positive.sum()),
                     "current_visible_nonobject_conflict_pixels": int(current_conflict.sum()),
                     "object_observations_under_current_pusher": int((positive & observed["pusher_mask"]).sum())})
    historical_conflict = observed["pusher_mask"] & (positive_count > 0) & (negative_count > 0)
    conflicts |= historical_conflict
    added = observed["pusher_mask"] & (positive_count > 0) & (negative_count == 0)
    fused_mask = observed["object_mask"] | added
    texture = np.zeros_like(current)
    texture[added] = latest_texture[added]
    texture[observed["object_mask"]] = current[observed["object_mask"]]
    source_frame[observed["object_mask"]] = len(images) - 1
    source_frame[~fused_mask] = -1
    unknown = observed["pusher_mask"] & (((positive_count == 0) & (negative_count == 0))
                                         | historical_conflict)
    return {
        "coordinate_frame": "last_completed_rgb",
        "reference_rgb": current.copy(),
        "reference_object_centroid": observed["object_centroid"].copy(),
        "object_mask": fused_mask,
        "object_rgb": texture,
        "source_frame_index": source_frame,
        "observation_count": positive_count,
        "nonobject_observation_count": negative_count,
        "unknown_occlusion_mask": unknown,
        "alignment_conflict_mask": conflicts,
        "registration_rows": rows,
        "completed_frame_count": len(images),
        "added_observed_object_pixels": int(added.sum()),
        "unknown_occlusion_pixels": int(unknown.sum()),
        "historical_occlusion_conflict_pixels": int(historical_conflict.sum()),
    }


class RgbSceneRenderer(torch.nn.Module):
    """Source-only differentiable rigid-image orbit, not a dynamics model.

    ``object_pose[B,3]`` is [dx_pixels, dy_pixels, theta_radians], relative to
    the visible source-object centroid.  Positive theta rotates clockwise in
    image coordinates. ``raw_pusher_xy[B,2]`` is converted by the fitted affine
    pixel map, not by a privileged renderer convention. No future RGB is taken.

    The observed source object and pusher textures are inverse-warped with
    proper SE(2)/translation using premultiplied alpha. Pusher is drawn on top.
    Source-occluded object parts remain absent; unknown static-background holes
    use the observed modal RGB color. Moving partial texture is an explicit
    hypothesis, not recovered shape, background or physical contact geometry.
    Outputs are unnormalized RGB [B,3,H,W] in [0,1], ready for the caller's SAME
    frozen encoder preprocessing. No goal image enters construction or render.

    Optional ``object_assets`` is the output of
    ``fuse_completed_object_assets`` and must refer to this exact source image.
    It changes ONLY object alpha/texture, retaining the current visible-centroid
    pose gauge. The pixel map, pusher texture and source-background construction
    are unchanged. Unknown occluded pixels remain absent in rendered texture;
    ``object_unknown_occlusion_mask`` exposes them to downstream geometry users.
    The default ``None`` path retains the original source-only values exactly.
    """

    def __init__(self, source_rgb, pixel_map, device="cpu", *, object_assets=None):
        super().__init__()
        source = _rgb(source_rgb)
        observation = observe_rgb(source)
        if not isinstance(pixel_map, dict) or "matrix" not in pixel_map:
            raise ValueError("pixel_map must contain the fitted affine matrix")
        matrix = np.asarray(pixel_map["matrix"], dtype=np.float64)
        if matrix.shape != (3, 2) or not np.isfinite(matrix).all():
            raise ValueError("expected finite pusher pixel map matrix [3,2]")
        self.height, self.width = source.shape[:2]
        if self.height < 2 or self.width < 2:
            raise ValueError("source image dimensions must be at least two")

        def buffer(name, value):
            self.register_buffer(name, torch.as_tensor(
                np.array(value, copy=True), dtype=torch.float32, device=device))

        source_chw = source.transpose(2, 0, 1).astype(np.float64) / 255.
        object_alpha = observation["object_mask"].astype(np.float64)[None]
        object_chw = source_chw
        self.object_unknown_occlusion_mask = observation["pusher_mask"].copy()
        if object_assets is not None:
            if (not isinstance(object_assets, dict)
                    or object_assets.get("coordinate_frame") != "last_completed_rgb"
                    or not np.array_equal(object_assets.get("reference_rgb"), source)
                    or not np.array_equal(object_assets.get("reference_object_centroid"),
                                          observation["object_centroid"])):
                raise ValueError("object assets must use this exact source RGB coordinate frame")
            mask = np.asarray(object_assets["object_mask"])
            texture = np.asarray(object_assets["object_rgb"])
            unknown = np.asarray(object_assets["unknown_occlusion_mask"])
            if (mask.shape != source.shape[:2] or mask.dtype != bool
                    or texture.shape != source.shape or texture.dtype != np.uint8
                    or unknown.shape != mask.shape or unknown.dtype != bool
                    or np.any(mask & unknown)
                    or np.any(observation["object_mask"] & ~mask)
                    or np.any(mask & ~observation["object_mask"] & ~observation["pusher_mask"])
                    or not np.array_equal(texture[observation["object_mask"]],
                                          source[observation["object_mask"]])):
                raise ValueError("invalid or nonconservative completed object assets")
            object_alpha = mask.astype(np.float64)[None]
            object_chw = texture.transpose(2, 0, 1).astype(np.float64) / 255.
            self.object_unknown_occlusion_mask = unknown.copy()
        pusher_alpha = observation["pusher_mask"].astype(np.float64)[None]
        buffer("background", observation["background_rgb"].transpose(2, 0, 1)[None] / 255.)
        buffer("object_texture", np.concatenate([object_chw * object_alpha,
                                                  object_alpha], axis=0)[None])
        buffer("pusher_texture", np.concatenate([source_chw * pusher_alpha,
                                                  pusher_alpha], axis=0)[None])
        buffer("object_centroid", observation["object_centroid"])
        buffer("pusher_center", observation["pusher_center"])
        buffer("pixel_matrix", matrix)
        y, x = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing="ij")
        buffer("pixel_grid", np.stack([x, y], axis=-1)[None])
        self.background_unknown_mask = observation["background_unknown_mask"].copy()

    def _sample(self, texture, source_coordinates):
        denominator = source_coordinates.new_tensor([self.width - 1, self.height - 1])
        normalized = 2. * source_coordinates / denominator - 1.
        return torch_functional.grid_sample(
            texture.expand(len(source_coordinates), -1, -1, -1), normalized,
            mode="bilinear", padding_mode="zeros", align_corners=True)

    def render(self, object_pose, raw_pusher_xy):
        """Render a batch using predicted pose/xy only; preserve their gradients."""
        if (not torch.is_tensor(object_pose) or not torch.is_tensor(raw_pusher_xy)
                or object_pose.ndim != 2 or object_pose.shape[1] != 3
                or raw_pusher_xy.shape != (len(object_pose), 2)
                or not object_pose.is_floating_point() or not raw_pusher_xy.is_floating_point()
                or len(object_pose) < 1):
            raise ValueError("expected floating object_pose [B,3] and raw_pusher_xy [B,2]")
        if (not torch.isfinite(object_pose).all() or not torch.isfinite(raw_pusher_xy).all()):
            raise FloatingPointError("nonfinite scene-rendering coordinates")
        pose = object_pose.to(device=self.pixel_grid.device, dtype=self.pixel_grid.dtype)
        xy = raw_pusher_xy.to(device=self.pixel_grid.device, dtype=self.pixel_grid.dtype)
        centered = self.pixel_grid - self.object_centroid - pose[:, None, None, :2]
        cosine = torch.cos(pose[:, 2])[:, None, None]
        sine = torch.sin(pose[:, 2])[:, None, None]
        inverse_object = torch.stack([
            cosine * centered[..., 0] + sine * centered[..., 1],
            -sine * centered[..., 0] + cosine * centered[..., 1],
        ], dim=-1) + self.object_centroid
        moved_object = self._sample(self.object_texture, inverse_object)
        homogeneous_xy = torch.cat([xy, torch.ones_like(xy[:, :1])], dim=1)
        pusher_pixels = homogeneous_xy @ self.pixel_matrix
        inverse_pusher = self.pixel_grid - (pusher_pixels - self.pusher_center)[:, None, None]
        moved_pusher = self._sample(self.pusher_texture, inverse_pusher)
        canvas = self.background * (1. - moved_object[:, 3:4]) + moved_object[:, :3]
        return canvas * (1. - moved_pusher[:, 3:4]) + moved_pusher[:, :3]

    def forward(self, object_pose, raw_pusher_xy):
        return self.render(object_pose, raw_pusher_xy)


def build_scene_renderer(source_rgb, pixel_map, device="cpu", *, object_assets=None):
    """Build an image orbit; optional history assets alter only object texture."""
    return RgbSceneRenderer(source_rgb, pixel_map, device=device, object_assets=object_assets)
