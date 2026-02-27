"""
Auto-rename tool for unknown faces.

Loads unknown face crops, groups similar faces using ArcFace embeddings,
shows each group in an OpenCV window, and lets you name + move them
to whitelist or blacklist via terminal input.

Usage:
    python label_unknowns.py
    python label_unknowns.py --threshold 0.35
    python label_unknowns.py --no-group
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import cv2
import numpy as np

from src.arcface_onnx import get_arcface
from src.config import load_config


# ── Helpers ─────────────────────────────────────────────────────────────────


def _next_path(output_dir: str, base: str) -> str:
    """
    Return a non-colliding file path.
    Sharol_Naim.jpg -> Sharol_Naim_2.jpg -> Sharol_Naim_3.jpg ...
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


def _load_images(unknown_dir: str) -> list[tuple[str, np.ndarray]]:
    """Load all images from the unknown directory. Returns list of (path, bgr)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    results = []
    if not os.path.isdir(unknown_dir):
        return results
    for fname in sorted(os.listdir(unknown_dir)):
        if os.path.splitext(fname)[1].lower() not in exts:
            continue
        fpath = os.path.join(unknown_dir, fname)
        img = cv2.imread(fpath)
        if img is not None:
            results.append((fpath, img))
        else:
            print(f"  WARNING: Could not read {fname}, will be placed in its own group.")
            results.append((fpath, None))
    return results


def _compute_embeddings(
    images: list[tuple[str, np.ndarray]],
) -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    """Compute ArcFace embeddings for all images. Returns list of (path, bgr, embedding)."""
    arcface = get_arcface()
    results = []
    total = len(images)
    for i, (path, img) in enumerate(images):
        print(f"\rComputing embeddings: {i + 1}/{total}...", end="", flush=True)
        emb = None
        if img is not None:
            emb = arcface.get_embedding(img)
        results.append((path, img, emb))
    print(" done.")
    return results


def _group_faces(
    entries: list[tuple[str, np.ndarray, np.ndarray | None]],
    threshold: float,
) -> list[list[tuple[str, np.ndarray]]]:
    """
    Greedy centroid clustering: cosine distance <= threshold -> same group.
    Returns groups sorted by size (largest first).
    """
    # Separate entries with valid embeddings from those without
    with_emb: list[tuple[str, np.ndarray, np.ndarray]] = []
    without_emb: list[tuple[str, np.ndarray]] = []

    for path, img, emb in entries:
        if emb is not None:
            with_emb.append((path, img, emb))
        else:
            without_emb.append((path, img))

    groups: list[list[tuple[str, np.ndarray]]] = []
    centroids: list[np.ndarray] = []

    for path, img, emb in with_emb:
        placed = False
        for gi, centroid in enumerate(centroids):
            dist = 1.0 - float(np.dot(emb, centroid))
            if dist <= threshold:
                groups[gi].append((path, img))
                # Update centroid as running mean (re-normalise)
                n = len(groups[gi])
                new_centroid = centroid * ((n - 1) / n) + emb * (1 / n)
                norm = float(np.linalg.norm(new_centroid))
                if norm > 0:
                    new_centroid /= norm
                centroids[gi] = new_centroid
                placed = True
                break
        if not placed:
            groups.append([(path, img)])
            centroids.append(emb.copy())

    # Each corrupt/unreadable image becomes its own group
    for path, img in without_emb:
        groups.append([(path, img)])

    # Sort by group size descending
    groups.sort(key=len, reverse=True)
    return groups


def _make_montage(
    faces: list[tuple[str, np.ndarray]],
    target_h: int = 200,
    max_show: int = 8,
) -> np.ndarray:
    """Create a horizontal montage of face images, resized to target_h."""
    show = faces[:max_show]
    resized = []
    for _, img in show:
        if img is None:
            # Placeholder for corrupt image
            placeholder = np.zeros((target_h, target_h, 3), dtype=np.uint8)
            cv2.putText(
                placeholder, "?", (target_h // 3, target_h * 2 // 3),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4,
            )
            resized.append(placeholder)
            continue
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            continue
        scale = target_h / h
        new_w = max(1, int(w * scale))
        r = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized.append(r)

    if not resized:
        blank = np.zeros((target_h, 200, 3), dtype=np.uint8)
        cv2.putText(blank, "No images", (10, target_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return blank

    # Add "+N more" tile if there are extra faces
    extra = len(faces) - max_show
    if extra > 0:
        tile = np.zeros((target_h, 120, 3), dtype=np.uint8)
        cv2.putText(
            tile, f"+{extra}", (10, target_h // 2 + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2,
        )
        cv2.putText(
            tile, "more", (15, target_h // 2 + 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1,
        )
        resized.append(tile)

    return np.hstack(resized)


def _move_group(
    faces: list[tuple[str, np.ndarray]],
    dest_dir: str,
    name: str,
) -> int:
    """Move all files in a group to dest_dir with the given name. Returns count moved."""
    os.makedirs(dest_dir, exist_ok=True)
    base = name.strip().replace(" ", "_")
    moved = 0
    for path, _ in faces:
        dest = _next_path(dest_dir, base)
        try:
            shutil.move(path, dest)
            moved += 1
        except Exception as exc:
            print(f"  WARNING: Failed to move {os.path.basename(path)}: {exc}")
    return moved


def _delete_group(faces: list[tuple[str, np.ndarray]]) -> int:
    """Delete all files in a group. Returns count deleted."""
    deleted = 0
    for path, _ in faces:
        try:
            os.remove(path)
            deleted += 1
        except Exception as exc:
            print(f"  WARNING: Failed to delete {os.path.basename(path)}: {exc}")
    return deleted


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review and label unknown face crops."
    )
    parser.add_argument(
        "--config", default="config/settings.yaml",
        help="Path to settings.yaml (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Cosine distance threshold for grouping (default: from config)",
    )
    parser.add_argument(
        "--no-group", action="store_true",
        help="Disable grouping — review each file individually",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    unknown_dir = cfg.paths.unknown_faces_dir
    whitelist_dir = cfg.paths.whitelist_dir
    blacklist_dir = cfg.paths.blacklist_dir
    threshold = args.threshold if args.threshold is not None else cfg.deepface.threshold

    # Load images
    images = _load_images(unknown_dir)
    if not images:
        print(f"No unknown face files found in {unknown_dir}/")
        return
    print(f"\nFound {len(images)} unknown face file(s).")

    # Compute embeddings and group
    if args.no_group:
        groups = [[(path, img)] for path, img in images]
        print(f"No-group mode: {len(groups)} individual file(s).\n")
    else:
        entries = _compute_embeddings(images)
        groups = _group_faces(entries, threshold)
        print(f"Grouped into {len(groups)} group(s) (threshold={threshold:.2f}).\n")

    # Counters
    whitelist_count = 0
    blacklist_count = 0
    skipped_count = 0
    deleted_count = 0

    window_name = "Label Unknowns"

    try:
        for gi, group in enumerate(groups):
            # Show montage
            montage = _make_montage(group)
            cv2.imshow(window_name, montage)
            cv2.waitKey(1)  # Pump event loop so window appears

            print(f"--- Group {gi + 1}/{len(groups)} ({len(group)} face(s)) ---")
            print("  <name>       -> Move to WHITELIST")
            print("  b:<name>     -> Move to BLACKLIST")
            print("  s            -> Skip")
            print("  d            -> Delete all in group")
            print("  q            -> Quit")

            choice = input("\nYour choice: ").strip()

            if not choice or choice.lower() == "s":
                skipped_count += len(group)
                print(f"  Skipped {len(group)} file(s).\n")

            elif choice.lower() == "q":
                print("  Quitting early.")
                break

            elif choice.lower() == "d":
                confirm = input(f"  Delete {len(group)} file(s)? (y/n): ").strip().lower()
                if confirm == "y":
                    n = _delete_group(group)
                    deleted_count += n
                    print(f"  Deleted {n} file(s).\n")
                else:
                    skipped_count += len(group)
                    print("  Cancelled — skipped.\n")

            elif choice.lower().startswith("b:"):
                name = choice[2:].strip()
                if not name:
                    print("  No name given — skipped.\n")
                    skipped_count += len(group)
                    continue
                n = _move_group(group, blacklist_dir, name)
                blacklist_count += n
                print(f"  Moved {n} file(s) to blacklist as \"{name}\".\n")

            else:
                name = choice
                n = _move_group(group, whitelist_dir, name)
                whitelist_count += n
                print(f"  Moved {n} file(s) to whitelist as \"{name}\".\n")

    except KeyboardInterrupt:
        print("\n\n  Interrupted — showing partial summary.")

    finally:
        cv2.destroyAllWindows()

    # Summary
    print("\n=== Summary ===")
    print(f"  Moved to whitelist: {whitelist_count}")
    print(f"  Moved to blacklist: {blacklist_count}")
    print(f"  Skipped:            {skipped_count}")
    print(f"  Deleted:            {deleted_count}")
    print()


if __name__ == "__main__":
    main()
