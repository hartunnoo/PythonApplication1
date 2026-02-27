"""
Email notifier — sends SMTP alert when a blacklist face is detected.

Attaches the screenshot JPEG to an HTML email.
Runs in a daemon thread so detection is never blocked.

Supports any SMTP server (Gmail, Outlook, custom).
For Gmail: enable App Passwords in Google Account settings.
"""

from __future__ import annotations

import os
import smtplib
import ssl
import threading
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from src.config import EmailConfig
from src.logger_setup import get_logger

log = get_logger(__name__)


class EmailNotifier:
    def __init__(self, cfg: EmailConfig) -> None:
        self._cfg = cfg
        if cfg.enabled:
            log.info(
                "EmailNotifier ready — SMTP %s:%d → %s",
                cfg.smtp_host, cfg.smtp_port,
                ", ".join(cfg.to_addrs),
            )

    def send_blacklist_alert(
        self,
        name: str,
        confidence: float,
        camera_label: str,
        timestamp: datetime,
        screenshot_path: Optional[str] = None,
        age: Optional[int] = None,
        emotion: Optional[str] = None,
    ) -> None:
        if not self._cfg.enabled:
            return
        threading.Thread(
            target=self._send,
            kwargs=dict(
                name=name, confidence=confidence,
                camera_label=camera_label, timestamp=timestamp,
                screenshot_path=screenshot_path,
                age=age, emotion=emotion,
            ),
            daemon=True,
            name="email-alert",
        ).start()

    # ── Private ───────────────────────────────────────────────────────

    def _send(
        self,
        name: str,
        confidence: float,
        camera_label: str,
        timestamp: datetime,
        screenshot_path: Optional[str] = None,
        age: Optional[int] = None,
        emotion: Optional[str] = None,
    ) -> None:
        try:
            cfg = self._cfg
            ts_str  = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            subject = f"[SECURITY ALERT] Blacklist detected — {name} @ {camera_label}"

            # ── Build HTML body ───────────────────────────────────────
            age_row     = f"<tr><td><b>Age (est.)</b></td><td>~{age} yrs — {'Child' if age < 18 else 'Adult'}</td></tr>" if age is not None else ""
            emotion_row = f"<tr><td><b>Expression</b></td><td>{emotion.capitalize()}</td></tr>" if emotion else ""
            img_tag     = '<br><img src="cid:screenshot" style="max-width:480px;border:2px solid #c00;border-radius:4px;" />' if screenshot_path and os.path.isfile(screenshot_path) else ""

            html = f"""
<html><body style="font-family:Segoe UI,Arial,sans-serif;background:#f5f5f5;padding:20px;">
<div style="max-width:520px;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px #0002;">
  <div style="background:#b01c2e;padding:18px 24px;">
    <h2 style="color:#fff;margin:0;font-size:18px;">&#x26A0; SECURITY ALERT — BLACKLIST DETECTED</h2>
  </div>
  <div style="padding:24px;">
    <h3 style="color:#b01c2e;font-size:22px;margin:0 0 16px;">{name}</h3>
    <table style="width:100%;border-collapse:collapse;font-size:14px;">
      <tr style="background:#fafafa;"><td style="padding:6px 10px;width:130px;color:#666;"><b>Camera</b></td><td style="padding:6px 10px;">{camera_label}</td></tr>
      <tr><td style="padding:6px 10px;color:#666;"><b>Detected at</b></td><td style="padding:6px 10px;">{ts_str}</td></tr>
      <tr style="background:#fafafa;"><td style="padding:6px 10px;color:#666;"><b>Confidence</b></td><td style="padding:6px 10px;">{confidence:.1%}</td></tr>
      {age_row}
      {emotion_row}
    </table>
    {img_tag}
    <p style="color:#888;font-size:12px;margin-top:20px;">This is an automated alert from the Face Recognition Security System.</p>
  </div>
</div>
</body></html>
"""

            # ── Assemble MIME message ─────────────────────────────────
            msg = MIMEMultipart("related")
            msg["Subject"] = subject
            msg["From"]    = cfg.from_addr
            msg["To"]      = ", ".join(cfg.to_addrs)

            msg.attach(MIMEText(html, "html"))

            if screenshot_path and os.path.isfile(screenshot_path):
                with open(screenshot_path, "rb") as fh:
                    img = MIMEImage(fh.read(), _subtype="jpeg")
                img.add_header("Content-ID", "<screenshot>")
                img.add_header(
                    "Content-Disposition", "inline",
                    filename=os.path.basename(screenshot_path),
                )
                msg.attach(img)

            # ── Send ──────────────────────────────────────────────────
            context = ssl.create_default_context()
            if cfg.use_tls:
                with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=15) as server:
                    server.starttls(context=context)
                    if cfg.username:
                        server.login(cfg.username, cfg.password)
                    server.sendmail(cfg.from_addr, cfg.to_addrs, msg.as_string())
            else:
                with smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port,
                                       context=context, timeout=15) as server:
                    if cfg.username:
                        server.login(cfg.username, cfg.password)
                    server.sendmail(cfg.from_addr, cfg.to_addrs, msg.as_string())

            log.info("Email alert sent to %s for %s.", cfg.to_addrs, name)

        except Exception as exc:
            log.warning("Email alert failed: %s", exc)
