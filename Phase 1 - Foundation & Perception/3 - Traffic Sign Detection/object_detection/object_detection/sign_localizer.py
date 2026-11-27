#!/usr/bin/env python3

"""
sign_localizer.py.

Per-detection sign localisation for the inference node: turns one detection
(class, box, and the outline polygon of a segmentation model) plus the depth
image into distance estimates from every available method, and fuses them.

Methods, all reported so that they can be compared on the car:
  stereo_box   stereo depth in a small patch at the box centre (old method,
               computed by the node and passed in)
  stereo_mask  median stereo depth inside the sign's outline
  box_size     known sign height / box height (needs only the class)
  geometry     shape fit to the outline with the known sign size
               (sign_geometry.estimate_pose; needs a segmentation model)
  fused        geometry and stereo_mask combined by their uncertainties

No ROS code should exist here.
"""

import cv2
import numpy as np
import yaml

from .sign_geometry import (box_size_distance, Camera, estimate_pose, fuse_distance,
                            load_config, stereo_depth_in_mask, stereo_sigma)


class SignLocalizer:
    """Distance and angle of detected signs from outline, size and stereo."""

    def __init__(self, config_path, rotated_180=True, gate=3.0, edge_noise_px=2.0,
                 stereo_rel_floor=0.02):
        """
        Load sign sizes / shapes and the fallback calibration.

        rotated_180 : whether the node rotates camera images by 180 deg
                      before detection (the calibration is rotated to match)
        edge_noise_px : uncertainty of a segmentation outline's edge (px)
        stereo_rel_floor : smallest relative stereo error assumed (the OAK-D
                      is specified at < 2 % depth error; the disparity model
                      alone is far more optimistic close up)
        """
        camera, self.specs, stereo = load_config(config_path)
        with open(config_path, encoding='utf-8') as f:
            config_rotated = bool(yaml.safe_load(f)['camera'].get('image_rotated_180', False))
        if config_rotated != rotated_180:
            camera = camera.rotated_180()          # a 180 deg rotation undoes itself
        self.config_camera = camera
        self.info_camera = None
        self.rotated_180 = rotated_180
        self.gate = gate
        self.edge_noise_px = edge_noise_px
        self.stereo_rel_floor = stereo_rel_floor
        self.stereo_focal_px = float(stereo.get('focal_px', 800.0))
        self.stereo_baseline_m = float(stereo.get('baseline_m', 0.075))
        self.disparity_sigma_px = float(stereo.get('disparity_sigma_px', 0.25))
        self._scaled = {}

    def set_camera_info(self, k, d, width, height):
        """Use the camera's own calibration (sensor_msgs/CameraInfo fields)."""
        if not width or not height or not any(k):
            return
        cam = Camera(np.asarray(k, dtype=np.float64).reshape(3, 3),
                     np.asarray(d, dtype=np.float64), int(width), int(height))
        self.info_camera = cam.rotated_180() if self.rotated_180 else cam
        self._scaled.clear()

    @property
    def calibration_source(self):
        return 'camera_info' if self.info_camera is not None else 'config'

    def camera_for(self, width, height):
        """Calibration for images of this size (scaled if the size differs)."""
        base = self.info_camera or self.config_camera
        if (base.width, base.height) == (width, height):
            return base
        key = (id(base), width, height)
        if key not in self._scaled:
            sx, sy = width / base.width, height / base.height
            K = base.K.copy()
            K[0, :] *= sx
            K[1, :] *= sy
            self._scaled[key] = Camera(K, base.dist.copy(), width, height)
        return self._scaled[key]

    def localize(self, det, frame_shape, depth_mm=None, stereo_box=None):
        """
        All distance estimates for one detection.

        det        : dict with class_name, bbox [x1, y1, x2, y2] and, from a
                     segmentation model, polygon (Nx2 pixel coordinates)
        frame_shape: shape of the image the detection was made on
        depth_mm   : depth image (uint16, mm) rotated like the colour image
        stereo_box : old box-centre stereo distance (m) or None

        Returns a dict; 'distance' is the best estimate (fused, else whatever
        is available, else None).
        """
        h, w = frame_shape[:2]
        camera = self.camera_for(w, h)
        spec = self.specs.get(det['class_name'])
        out = {'stereo_box': stereo_box, 'stereo_mask': None, 'box_size': None,
               'geometry': None, 'geometry_sigma': None, 'view_angle_deg': None,
               'method': None, 'fit_error_px': None, 'fused': None, 'fused_sigma': None,
               'source': None, 'chi': None, 'consistent': None, 'distance': stereo_box}

        mask = None
        polygon = det.get('polygon')
        if polygon is not None and len(polygon) >= 3:
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [np.round(np.asarray(polygon)).astype(np.int32)], 1)

        if spec is not None:
            out['box_size'] = box_size_distance(det['bbox'], spec, camera)

        if mask is not None and depth_mm is not None:
            out['stereo_mask'] = stereo_depth_in_mask(depth_mm, mask)

        pose = None
        if mask is not None and spec is not None:
            try:
                pose = estimate_pose(mask, spec, camera, edge_noise_px=self.edge_noise_px)
            except cv2.error:
                pose = None
        if pose is not None:
            out.update(geometry=pose.z, geometry_sigma=pose.sigma_z,
                       view_angle_deg=pose.view_angle_deg, method=pose.method,
                       fit_error_px=pose.fit_error_px)

        stereo_z = out['stereo_mask'] if out['stereo_mask'] is not None else stereo_box
        stereo = None
        if stereo_z is not None and stereo_z > 0:
            stereo = (stereo_z, stereo_sigma(stereo_z, self.stereo_focal_px,
                                             self.stereo_baseline_m, self.disparity_sigma_px,
                                             floor_m=max(0.005, self.stereo_rel_floor * stereo_z)))
        geometry = (pose.z, pose.sigma_z) if pose is not None else None
        fused = fuse_distance(geometry, stereo, self.gate)
        if fused is not None:
            out.update(fused=fused.z, fused_sigma=fused.sigma, source=fused.source,
                       chi=fused.chi, consistent=fused.consistent, distance=fused.z)
        return out
