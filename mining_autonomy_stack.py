import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import carla
import cv2
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table

# ==========================================================
# CONFIGURATION
# ==========================================================
MAP_NAME = "Mine_01"

GNSS_FAIL_TIME = 20
IMU_SPIKE_TIME = 35
CAMERA_FAIL_TIME = 50

ENDURANCE_MODE = False
DT = 0.05
MAX_RUNTIME_S = 300 if ENDURANCE_MODE else 120

# Stereo Camera Config
STEREO_BASELINE = 0.6
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
CAMERA_FOV = 90

# Depth + Obstacle logic
OBSTACLE_BRAKE_DISTANCE_M = 8.0
OBSTACLE_CRITICAL_DISTANCE_M = 5.0
DEPTH_ROI_TOP = 220
DEPTH_ROI_BOTTOM = 560
DEPTH_ROI_LEFT = 250
DEPTH_ROI_RIGHT = 550

DRIFT_LOG_FILE = "drift_history.csv"
VALIDATION_CSV_FILE = "industrial_validation_log.csv"
PDF_REPORT_FILE = "Autonomy_Report.pdf"


@dataclass
class ValidationSnapshot:
    t_s: float
    gt_x: float
    gt_y: float
    est_x: float
    est_y: float
    speed_mps: float
    drift_m: float
    gnss_active: int
    imu_spike: int
    camera_active: int
    min_obstacle_m: float
    obstacle_brake: int
    vo_valid: int


# ==========================================================
# EKF (State: x, y, vx, vy)
# ==========================================================
H_POS = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)


def ekf_predict(
    x_state: np.ndarray,
    cov_p: np.ndarray,
    dt: float,
    process_q: np.ndarray,
    ax: float,
    ay: float,
) -> Tuple[np.ndarray, np.ndarray]:
    f_mat = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    )
    b_mat = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]], dtype=float)

    u_vec = np.array([ax, ay], dtype=float)
    x_pred = f_mat @ x_state + b_mat @ u_vec
    p_pred = f_mat @ cov_p @ f_mat.T + process_q
    return x_pred, p_pred


def ekf_update(
    x_state: np.ndarray,
    cov_p: np.ndarray,
    z_meas: np.ndarray,
    h_mat: np.ndarray,
    meas_r: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    y_residual = z_meas - h_mat @ x_state
    s_mat = h_mat @ cov_p @ h_mat.T + meas_r
    k_gain = cov_p @ h_mat.T @ np.linalg.inv(s_mat)
    x_upd = x_state + k_gain @ y_residual
    p_upd = (np.eye(len(x_state)) - k_gain @ h_mat) @ cov_p
    return x_upd, p_upd


# ==========================================================
# Visual Odometry helpers
# ==========================================================
def compute_vo_delta(prev_gray: np.ndarray, curr_gray: np.ndarray) -> Tuple[float, float, bool]:
    orb = cv2.ORB_create(2500)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    if des1 is None or des2 is None or kp1 is None or kp2 is None:
        return 0.0, 0.0, False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 15:
        return 0.0, 0.0, False

    matches = sorted(matches, key=lambda m: m.distance)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:150]])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:150]])

    transform, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)
    if transform is None or inliers is None or int(inliers.sum()) < 10:
        return 0.0, 0.0, False

    dx_px = float(transform[0, 2])
    dy_px = float(transform[1, 2])
    return dx_px, dy_px, True


def disparity_to_depth(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    baseline_m: float,
    fov_deg: float,
    width_px: int,
) -> Tuple[np.ndarray, np.ndarray]:
    gray_l = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 8,
        blockSize=7,
        P1=8 * 3 * 7 * 7,
        P2=32 * 3 * 7 * 7,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=1,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disparity = sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0
    disparity[disparity <= 0.0] = np.nan

    focal_length_px = width_px / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    depth_m = (focal_length_px * baseline_m) / disparity
    depth_m = np.clip(depth_m, 0.0, 120.0)

    return gray_l, depth_m


def main() -> None:
    # ==========================================================
    # CONNECT
    # ==========================================================
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)

    world = client.get_world()
    if MAP_NAME not in world.get_map().name:
        world = client.load_world(MAP_NAME)
        time.sleep(5)
        world = client.get_world()

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT
    world.apply_settings(settings)

    carla_map = world.get_map()
    blueprints = world.get_blueprint_library()
    print("Running on:", world.get_map().name)

    # ==========================================================
    # SPAWN TRUCK
    # ==========================================================
    truck_bp = blueprints.find("vehicle.miningtruck.miningtruck")

    spawn = carla.Transform(carla.Location(x=20, y=-75, z=-15.1), carla.Rotation(yaw=10))
    vehicle = world.try_spawn_actor(truck_bp, spawn)
    if vehicle is None:
        raise RuntimeError("Spawn failed for vehicle.miningtruck.miningtruck")

    print("Truck spawned.")

    # ==========================================================
    # SENSORS
    # ==========================================================
    gnss = world.spawn_actor(
        blueprints.find("sensor.other.gnss"),
        carla.Transform(carla.Location(z=2)),
        attach_to=vehicle,
    )

    imu = world.spawn_actor(
        blueprints.find("sensor.other.imu"),
        carla.Transform(carla.Location(z=2)),
        attach_to=vehicle,
    )

    camera_bp = blueprints.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
    camera_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
    camera_bp.set_attribute("fov", str(CAMERA_FOV))

    left_camera = world.spawn_actor(
        camera_bp,
        carla.Transform(carla.Location(x=1.5, y=-STEREO_BASELINE / 2, z=2.5)),
        attach_to=vehicle,
    )

    right_camera = world.spawn_actor(
        camera_bp,
        carla.Transform(carla.Location(x=1.5, y=STEREO_BASELINE / 2, z=2.5)),
        attach_to=vehicle,
    )

    # ==========================================================
    # GLOBAL BUFFERS
    # ==========================================================
    buffers: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
    drift_history: List[float] = []
    validation_log: List[ValidationSnapshot] = []

    gnss_active = True
    imu_spike = False
    camera_active = True

    # ==========================================================
    # CALLBACKS
    # ==========================================================
    def left_callback(image: carla.Image) -> None:
        bgr = np.reshape(np.copy(image.raw_data), (CAMERA_HEIGHT, CAMERA_WIDTH, 4))[:, :, :3]
        buffers["left"] = bgr

    def right_callback(image: carla.Image) -> None:
        bgr = np.reshape(np.copy(image.raw_data), (CAMERA_HEIGHT, CAMERA_WIDTH, 4))[:, :, :3]
        buffers["right"] = bgr

    left_camera.listen(left_callback)
    right_camera.listen(right_callback)

    # ==========================================================
    # EKF initialization
    # ==========================================================
    x_state = np.array([spawn.location.x, spawn.location.y, 0.0, 0.0], dtype=float)
    cov_p = np.eye(4, dtype=float)
    process_q = np.eye(4, dtype=float) * 0.08

    r_gnss = np.eye(2, dtype=float) * 1.5
    h_vo = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    r_vo = np.eye(2, dtype=float) * 3.5

    last_imu_ax = 0.0
    last_imu_ay = 0.0
    prev_gray: Optional[np.ndarray] = None

    def imu_callback(imu_data: carla.IMUMeasurement) -> None:
        nonlocal last_imu_ax, last_imu_ay
        last_imu_ax = float(imu_data.accelerometer.x)
        last_imu_ay = float(imu_data.accelerometer.y)

    imu.listen(imu_callback)

    # ==========================================================
    # MAIN LOOP
    # ==========================================================
    start_time = time.time()
    plt.ion()
    fig, (ax_drift, ax_depth) = plt.subplots(1, 2, figsize=(12, 5))

    try:
        while True:
            world.tick()
            t = time.time() - start_time
            if t >= MAX_RUNTIME_S:
                print("Runtime limit reached, stopping run.")
                break

            if t > GNSS_FAIL_TIME:
                gnss_active = False
            if t > IMU_SPIKE_TIME:
                imu_spike = True
            if t > CAMERA_FAIL_TIME:
                camera_active = False

            transform = vehicle.get_transform()
            gt_x = float(transform.location.x)
            gt_y = float(transform.location.y)

            ax_ekf = last_imu_ax
            ay_ekf = last_imu_ay
            if imu_spike:
                ax_ekf += np.random.normal(0.0, 3.0)
                ay_ekf += np.random.normal(0.0, 3.0)

            x_state, cov_p = ekf_predict(x_state, cov_p, DT, process_q, ax_ekf, ay_ekf)

            if gnss_active:
                z_gnss = np.array([gt_x, gt_y], dtype=float)
                x_state, cov_p = ekf_update(x_state, cov_p, z_gnss, H_POS, r_gnss)

            min_obstacle_m = float("nan")
            obstacle_brake = False
            vo_valid = False

            # ======================================================
            # Stereo depth, VO, and EKF fusion
            # ======================================================
            if buffers["left"] is not None and buffers["right"] is not None and camera_active:
                left_image = buffers["left"]
                right_image = buffers["right"]

                gray_left, depth_map = disparity_to_depth(
                    left_image,
                    right_image,
                    STEREO_BASELINE,
                    CAMERA_FOV,
                    CAMERA_WIDTH,
                )

                # Visual Odometry update from image frame-to-frame motion.
                if prev_gray is not None:
                    dx_px, dy_px, vo_valid = compute_vo_delta(prev_gray, gray_left)
                    if vo_valid:
                        # Approximate metric conversion with image scale heuristic.
                        # Scale reduces as average depth increases.
                        center_depth = np.nanmedian(
                            depth_map[
                                DEPTH_ROI_TOP:DEPTH_ROI_BOTTOM,
                                DEPTH_ROI_LEFT:DEPTH_ROI_RIGHT,
                            ]
                        )
                        depth_scale = 0.004 * max(3.0, min(30.0, float(center_depth)))
                        yaw = math.radians(transform.rotation.yaw)

                        # Camera-frame translation to world-frame approximate update.
                        local_dx = -dx_px * depth_scale
                        local_dy = dy_px * depth_scale
                        world_dx = math.cos(yaw) * local_dx - math.sin(yaw) * local_dy
                        world_dy = math.sin(yaw) * local_dx + math.cos(yaw) * local_dy

                        vo_position = np.array([x_state[0] + world_dx, x_state[1] + world_dy])
                        x_state, cov_p = ekf_update(x_state, cov_p, vo_position, h_vo, r_vo)

                prev_gray = gray_left

                # Obstacle detection ROI for safety braking.
                roi = depth_map[DEPTH_ROI_TOP:DEPTH_ROI_BOTTOM, DEPTH_ROI_LEFT:DEPTH_ROI_RIGHT]
                roi_valid = roi[np.isfinite(roi)]
                if roi_valid.size > 0:
                    min_obstacle_m = float(np.percentile(roi_valid, 5))
                    obstacle_brake = min_obstacle_m < OBSTACLE_BRAKE_DISTANCE_M

                ax_depth.clear()
                ax_depth.imshow(depth_map, cmap="plasma", vmin=0, vmax=60)
                ax_depth.set_title("Stereo SGBM Depth (m)")

            # ======================================================
            # WAYPOINT CONTROL + obstacle override
            # ======================================================
            current_wp = carla_map.get_waypoint(
                vehicle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

            next_wps = current_wp.next(18.0)
            target_wp = next_wps[0] if len(next_wps) > 0 else current_wp

            vehicle_loc = transform.location
            yaw = math.radians(transform.rotation.yaw)

            dx = float(target_wp.transform.location.x - vehicle_loc.x)
            dy = float(target_wp.transform.location.y - vehicle_loc.y)

            local_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy
            local_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy

            steering_angle = math.atan2(2 * 4.5 * local_y, max(1.0, local_x * local_x + local_y * local_y))
            steer = max(-1.0, min(1.0, 2.0 * steering_angle))

            velocity = vehicle.get_velocity()
            speed = math.sqrt(float(velocity.x * velocity.x + velocity.y * velocity.y))

            target_speed = 8.0 * (1.0 - min(abs(steer), 0.8))
            error = target_speed - speed
            throttle = min(0.5, 0.12 * error) if error > 0 else 0.0
            brake = min(0.5, -0.12 * error) if error < 0 else 0.0

            # Obstacle emergency response.
            if obstacle_brake:
                throttle = 0.0
                brake = 0.6 if (not math.isnan(min_obstacle_m) and min_obstacle_m >= OBSTACLE_CRITICAL_DISTANCE_M) else 1.0

            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

            drift = math.sqrt((x_state[0] - gt_x) ** 2 + (x_state[1] - gt_y) ** 2)
            drift_history.append(float(drift))

            validation_log.append(
                ValidationSnapshot(
                    t_s=float(t),
                    gt_x=gt_x,
                    gt_y=gt_y,
                    est_x=float(x_state[0]),
                    est_y=float(x_state[1]),
                    speed_mps=float(speed),
                    drift_m=float(drift),
                    gnss_active=int(gnss_active),
                    imu_spike=int(imu_spike),
                    camera_active=int(camera_active),
                    min_obstacle_m=float(min_obstacle_m) if not math.isnan(min_obstacle_m) else -1.0,
                    obstacle_brake=int(obstacle_brake),
                    vo_valid=int(vo_valid),
                )
            )

            ax_drift.clear()
            ax_drift.plot(drift_history)
            ax_drift.set_title("Real-Time Drift (m)")
            ax_drift.set_xlabel("Frame")
            ax_drift.set_ylabel("Meters")

            plt.pause(0.001)
            time.sleep(DT)

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Saving logs...")

    finally:
        print("Applying emergency brake...")
        vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))
        time.sleep(2)

        if drift_history:
            np.savetxt(DRIFT_LOG_FILE, np.array(drift_history), delimiter=",")

        if validation_log:
            csv_header = (
                "t_s,gt_x,gt_y,est_x,est_y,speed_mps,drift_m,gnss_active,"
                "imu_spike,camera_active,min_obstacle_m,obstacle_brake,vo_valid"
            )
            csv_rows = np.array(
                [
                    [
                        item.t_s,
                        item.gt_x,
                        item.gt_y,
                        item.est_x,
                        item.est_y,
                        item.speed_mps,
                        item.drift_m,
                        item.gnss_active,
                        item.imu_spike,
                        item.camera_active,
                        item.min_obstacle_m,
                        item.obstacle_brake,
                        item.vo_valid,
                    ]
                    for item in validation_log
                ]
            )
            np.savetxt(VALIDATION_CSV_FILE, csv_rows, delimiter=",", header=csv_header, comments="")

        for sensor in (gnss, imu, left_camera, right_camera):
            sensor.stop()

        actors = [vehicle, gnss, imu, left_camera, right_camera]
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(False)

        print("Actors destroyed safely.")

    # ==========================================================
    # CEP95 + ISO-style industrial validation report
    # ==========================================================
    if not drift_history:
        print("No drift samples collected. Skipping report generation.")
        return

    drift_np = np.array(drift_history)
    cep95 = float(np.percentile(drift_np, 95))
    mean_drift = float(np.mean(drift_np))
    max_drift = float(np.max(drift_np))

    obstacle_events = int(sum(1 for row in validation_log if row.obstacle_brake == 1))
    vo_valid_count = int(sum(1 for row in validation_log if row.vo_valid == 1))
    vo_ratio = 100.0 * vo_valid_count / max(1, len(validation_log))

    # ISO-style pass/fail gates (example quality gates for industrial validation).
    gate_cep95 = "PASS" if cep95 <= 3.0 else "FAIL"
    gate_obstacle = "PASS" if obstacle_events >= 1 else "WARN"
    gate_vo = "PASS" if vo_ratio >= 50 else "WARN"

    doc = SimpleDocTemplate(PDF_REPORT_FILE)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Autonomous Mining Vehicle - Industrial Validation Report", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("Validation Profile: ISO-style safety and localization gates", styles["Heading3"]))
    elements.append(Spacer(1, 0.2 * inch))

    metrics_table = [
        ["Metric", "Value"],
        ["CEP95 Drift (m)", f"{cep95:.3f}"],
        ["Mean Drift (m)", f"{mean_drift:.3f}"],
        ["Max Drift (m)", f"{max_drift:.3f}"],
        ["GNSS Failure Injected at (s)", str(GNSS_FAIL_TIME)],
        ["IMU Spike Injected at (s)", str(IMU_SPIKE_TIME)],
        ["Camera Failure Injected at (s)", str(CAMERA_FAIL_TIME)],
        ["Obstacle Brake Events", str(obstacle_events)],
        ["VO Valid Ratio (%)", f"{vo_ratio:.1f}"],
    ]

    gates_table = [
        ["Validation Gate", "Threshold", "Result"],
        ["Localization CEP95", "<= 3.0 m", gate_cep95],
        ["Obstacle Emergency Brake", ">= 1 event", gate_obstacle],
        ["Visual Odometry Availability", ">= 50% frames", gate_vo],
    ]

    elements.append(Table(metrics_table))
    elements.append(Spacer(1, 0.25 * inch))
    elements.append(Table(gates_table))

    doc.build(elements)

    print(f"PDF Generated: {PDF_REPORT_FILE}")
    print(f"Validation CSV: {VALIDATION_CSV_FILE}")
    print("Simulation Complete.")


if __name__ == "__main__":
    main()
