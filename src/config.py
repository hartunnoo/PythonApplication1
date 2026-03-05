"""
Configuration loader.
Reads settings.yaml and exposes typed, validated dataclasses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple

import yaml


@dataclass
class CameraConfig:
    device_index: int = 0
    label: str = "Camera 0"
    width: int = 1280
    height: int = 720
    fps: int = 30
    rtsp_url: str = ""          # If set, overrides device_index (e.g. rtsp://admin:pass@192.168.1.100/stream)


@dataclass
class DetectionConfig:
    frame_skip: int = 1


@dataclass
class MatchingConfig:
    tolerance: float = 0.55      # LBPH threshold = 170 * tolerance  (~93.5 at default)
    min_confidence: float = 0.20  # Lowered — confidence now uses fixed 250-ref, not threshold
    confirm_frames: int = 3


@dataclass
class AlertConfig:
    cooldown_seconds: int = 30
    popup_duration_ms: int = 5000
    popup_width: int = 460
    popup_height: int = 220


@dataclass
class LoggingConfig:
    level: str = "INFO"
    max_bytes: int = 10_485_760
    backup_count: int = 10
    log_dir: str = "logs"
    log_file: str = "face_recognition.log"
    match_log_file: str = "matches.log"


@dataclass
class PathsConfig:
    whitelist_dir: str = "known_faces/whitelist"
    blacklist_dir: str = "known_faces/blacklist"
    unknown_faces_dir: str = "known_faces/unknown"
    cache_dir: str = "cache"
    screenshots_dir: str = "screenshots"


@dataclass
class DisplayConfig:
    window_title: str = "Face Recognition System"
    show_fps: bool = True
    show_stats: bool = True
    unknown_label: str = "UNKNOWN VISITOR"
    box_thickness: int = 2
    font_scale: float = 0.70
    grid_cell_width: int = 960
    grid_cell_height: int = 540
    panel_width: int = 280   # Left scan-info panel width in pixels


@dataclass
class DeepFaceConfig:
    model: str = "ArcFace"
    detector_backend: str = "opencv"
    distance_metric: str = "cosine"
    threshold: float = 0.40


@dataclass
class SoundConfig:
    enabled: bool = True
    whitelist_beep: bool = True     # Play soft beep for whitelist too


@dataclass
class EmailConfig:
    enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    use_tls: bool = True            # True=STARTTLS, False=SSL
    username: str = ""
    password: str = ""              # Use Gmail App Password
    from_addr: str = ""
    to_addrs: List[str] = field(default_factory=list)


@dataclass
class DashboardConfig:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 5000


@dataclass
class ReportConfig:
    enabled: bool = True
    reports_dir: str = "reports"
    export_time: str = "00:05"      # HH:MM — export previous day's data


@dataclass
class ICScanConfig:
    enabled: bool = True
    scans_dir: str = "scans"
    scans_log: str = "scans/ic_scans.txt"
    tesseract_path: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


@dataclass
class AppConfig:
    cameras: List[CameraConfig] = field(default_factory=lambda: [CameraConfig()])
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    deepface: DeepFaceConfig = field(default_factory=DeepFaceConfig)
    sound: SoundConfig = field(default_factory=SoundConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    ic_scan: ICScanConfig = field(default_factory=ICScanConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/settings.yaml") -> AppConfig:
    if not os.path.exists(config_path):
        return AppConfig()

    with open(config_path, "r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    det_raw = raw.get("detection", {})
    mat_raw = raw.get("matching", {})
    ale_raw = raw.get("alerts", {})
    log_raw = raw.get("logging", {})
    pth_raw = raw.get("paths", {})
    dis_raw = raw.get("display", {})
    df_raw  = raw.get("deepface", {})
    snd_raw = raw.get("sound", {})
    eml_raw = raw.get("email", {})
    dsh_raw = raw.get("dashboard", {})
    rpt_raw = raw.get("report", {})
    ics_raw = raw.get("ic_scan", {})

    cameras: List[CameraConfig] = []
    if "cameras" in raw:
        for i, cam in enumerate(raw["cameras"]):
            cameras.append(CameraConfig(
                device_index=cam.get("device_index", i),
                label=cam.get("label", f"Camera {i}"),
                width=cam.get("width", 1280),
                height=cam.get("height", 720),
                fps=cam.get("fps", 30),
                rtsp_url=cam.get("rtsp_url", ""),
            ))
    elif "camera" in raw:
        cam = raw["camera"]
        cameras.append(CameraConfig(
            device_index=cam.get("device_index", 0),
            label=cam.get("label", "Camera 0"),
            width=cam.get("width", 1280),
            height=cam.get("height", 720),
            fps=cam.get("fps", 30),
            rtsp_url=cam.get("rtsp_url", ""),
        ))
    if not cameras:
        cameras = [CameraConfig()]

    return AppConfig(
        cameras=cameras,
        detection=DetectionConfig(
            frame_skip=det_raw.get("frame_skip", 1),
        ),
        matching=MatchingConfig(
            tolerance=mat_raw.get("tolerance", 0.85),
            min_confidence=mat_raw.get("min_confidence", 0.25),
            confirm_frames=mat_raw.get("confirm_frames", 3),
        ),
        alerts=AlertConfig(
            cooldown_seconds=ale_raw.get("cooldown_seconds", 30),
            popup_duration_ms=ale_raw.get("popup_duration_ms", 5000),
            popup_width=ale_raw.get("popup_width", 460),
            popup_height=ale_raw.get("popup_height", 220),
        ),
        logging=LoggingConfig(
            level=log_raw.get("level", "INFO"),
            max_bytes=log_raw.get("max_bytes", 10_485_760),
            backup_count=log_raw.get("backup_count", 10),
            log_dir=log_raw.get("log_dir", "logs"),
            log_file=log_raw.get("log_file", "face_recognition.log"),
            match_log_file=log_raw.get("match_log_file", "matches.log"),
        ),
        paths=PathsConfig(
            whitelist_dir=pth_raw.get("whitelist_dir", "known_faces/whitelist"),
            blacklist_dir=pth_raw.get("blacklist_dir", "known_faces/blacklist"),
            unknown_faces_dir=pth_raw.get("unknown_faces_dir", "known_faces/unknown"),
            cache_dir=pth_raw.get("cache_dir", "cache"),
            screenshots_dir=pth_raw.get("screenshots_dir", "screenshots"),
        ),
        display=DisplayConfig(
            window_title=dis_raw.get("window_title", "Face Recognition System"),
            show_fps=dis_raw.get("show_fps", True),
            show_stats=dis_raw.get("show_stats", True),
            unknown_label=dis_raw.get("unknown_label", "UNKNOWN VISITOR"),
            box_thickness=dis_raw.get("box_thickness", 2),
            font_scale=dis_raw.get("font_scale", 0.65),
            grid_cell_width=dis_raw.get("grid_cell_width", 640),
            grid_cell_height=dis_raw.get("grid_cell_height", 360),
            panel_width=dis_raw.get("panel_width", 220),
        ),
        deepface=DeepFaceConfig(
            model=df_raw.get("model", "ArcFace"),
            detector_backend=df_raw.get("detector_backend", "opencv"),
            distance_metric=df_raw.get("distance_metric", "cosine"),
            threshold=df_raw.get("threshold", 0.40),
        ),
        sound=SoundConfig(
            enabled=snd_raw.get("enabled", True),
            whitelist_beep=snd_raw.get("whitelist_beep", True),
        ),
        email=EmailConfig(
            enabled=eml_raw.get("enabled", False),
            smtp_host=eml_raw.get("smtp_host", "smtp.gmail.com"),
            smtp_port=int(eml_raw.get("smtp_port", 587)),
            use_tls=eml_raw.get("use_tls", True),
            username=eml_raw.get("username", ""),
            password=eml_raw.get("password", ""),
            from_addr=eml_raw.get("from_addr", ""),
            to_addrs=eml_raw.get("to_addrs", []),
        ),
        dashboard=DashboardConfig(
            enabled=dsh_raw.get("enabled", True),
            host=dsh_raw.get("host", "0.0.0.0"),
            port=int(dsh_raw.get("port", 5000)),
        ),
        report=ReportConfig(
            enabled=rpt_raw.get("enabled", True),
            reports_dir=rpt_raw.get("reports_dir", "reports"),
            export_time=rpt_raw.get("export_time", "00:05"),
        ),
        ic_scan=ICScanConfig(
            enabled=ics_raw.get("enabled", True),
            scans_dir=ics_raw.get("scans_dir", "scans"),
            scans_log=ics_raw.get("scans_log", "scans/ic_scans.txt"),
            tesseract_path=ics_raw.get("tesseract_path", r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        ),
    )
