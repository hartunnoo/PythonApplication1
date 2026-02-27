"""
Quick face capture utility.
Opens the webcam, shows a 3-second countdown, auto-captures and saves.
Usage: python capture_face.py --name "Tuno"
"""

import argparse
import os
import sys
import time

import cv2


def _next_path(output_dir: str, base: str) -> str:
    """
    Return a non-colliding file path.
    Sharol_Naim.jpg → Sharol_Naim_2.jpg → Sharol_Naim_3.jpg …
    """
    candidate = os.path.join(output_dir, base + ".jpg")
    if not os.path.exists(candidate):
        return candidate
    n = 2
    while True:
        candidate = os.path.join(output_dir, f"{base}_{n}.jpg")
        if not os.path.exists(candidate):
            return candidate
        n += 1


def capture(name: str, output_dir: str = "known_faces") -> str:
    os.makedirs(output_dir, exist_ok=True)
    base     = name.strip().replace(" ", "_")
    out_path = _next_path(output_dir, base)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)

    # Warm up camera (first few frames are often dark/blurry)
    for _ in range(10):
        cap.read()

    countdown_secs = 4
    start = time.time()
    captured_frame = None

    print(f"\n  Look at the camera — capturing in {countdown_secs} seconds...")
    print("  Press ESC to cancel.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        elapsed = time.time() - start
        remaining = max(0, countdown_secs - int(elapsed))

        h, w = display.shape[:2]

        if elapsed < countdown_secs:
            # Countdown overlay
            num = str(remaining if remaining > 0 else 1)
            (tw, th), _ = cv2.getTextSize(num, cv2.FONT_HERSHEY_SIMPLEX, 6, 8)
            cv2.putText(
                display, num,
                ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 220, 255), 8,
            )
            msg = f"  Capturing as: {name}  —  hold still!"
            cv2.putText(display, msg, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        else:
            # Flash white then save
            if captured_frame is None:
                captured_frame = frame.copy()
                cv2.imwrite(out_path, captured_frame)
                print(f"  Saved → {out_path}")

            cv2.putText(display, f"  Saved as: {name}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)
            cv2.putText(display, "  Closing in 2 seconds...", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            if time.time() - start > countdown_secs + 2:
                break

        cv2.imshow("Face Capture — Look at camera!", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("  Cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="Person", help="Person name for the face file")
    parser.add_argument(
        "--list", choices=["whitelist", "blacklist"], default="whitelist",
        help="Which list to add this person to"
    )
    args = parser.parse_args()
    out_dir = os.path.join("known_faces", args.list)
    path = capture(args.name, output_dir=out_dir)
    print(f"\n  Done! Face saved to: {path}  ({args.list.upper()})")
    print("  Now run:  python main.py  (or press R in the live window to reload)\n")
