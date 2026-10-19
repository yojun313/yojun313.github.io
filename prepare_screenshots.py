#!/usr/bin/env python3
"""크롬 스크린샷 → 여백·그림자를 정리한 둥근 WebP.

설치: python3 -m pip install Pillow numpy
실행: python3 prepare_screenshots.py /이미지/폴더
직접 높이 지정: python3 prepare_screenshots.py /이미지/폴더 --top 140
이미 크롬이 잘린 이미지: python3 prepare_screenshots.py /이미지/폴더 --top 0
확인만: --dry-run / 하위 폴더 포함: --recursive

같은 폴더에 같은 이름의 .webp를 저장합니다. 기존 WebP는 덮어씁니다.
PNG/JPEG 원본은 삭제하지 않습니다. WebP는 투명도를 포함해 무손실 저장합니다.
자동 크롬 감지는 macOS 창 버튼 기준이며 불확실하면 건너뜁니다.
바깥 여백·그림자 감지는 투명도가 있는 스크린샷을 기준으로 합니다.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import tempfile

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
except ImportError as exc:
    raise SystemExit("필요 패키지 설치: python3 -m pip install Pillow numpy") from exc

EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def detect_top(image: Image.Image) -> int | None:
    """Return a conservative toolbar boundary in original pixel coordinates."""
    scale = min(1.0, 1600 / image.width)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    thumb = image.convert("RGB").resize((width, height))
    rgb = np.asarray(thumb, dtype=np.int16)
    # Traffic lights must be close together, horizontally ordered and aligned.
    area = rgb[: min(height // 3, 300), : max(1, width // 3)]
    if area.size == 0:
        return None
    r, g, b = area[..., 0], area[..., 1], area[..., 2]
    masks = (
        (r > 190) & (g > 45) & (g < 160) & (b < 145) & (r - g > 65),
        (r > 180) & (g > 140) & (b < 100) & (r - g < 95),
        (r < 130) & (g > 140) & (b < 145) & (g - r > 45),
    )
    candidates = []
    for mask in masks:
        components = []
        # Compact connected components avoid mistaking long colored UI strips
        # for the circular window buttons. Only the small search area is walked.
        visited = np.zeros(mask.shape, dtype=bool)
        ys, xs = np.nonzero(mask)
        for sy, sx in zip(ys, xs):
            if visited[sy, sx]:
                continue
            stack = [(int(sy), int(sx))]
            visited[sy, sx] = True
            points = []
            while stack:
                y, x = stack.pop()
                points.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (
                        0 <= ny < mask.shape[0]
                        and 0 <= nx < mask.shape[1]
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            p = np.asarray(points)
            lo, hi = p.min(axis=0), p.max(axis=0)
            h, w = hi - lo + 1
            if (
                5 <= w <= 28
                and 5 <= h <= 28
                and 0.65 <= w / h <= 1.5
                and len(p) / (h * w) > 0.45
            ):
                components.append(
                    (float(p[:, 1].mean()), float(p[:, 0].mean()), float(max(h, w)))
                )
        candidates.append(components)

    for red in sorted(candidates[0], key=lambda p: p[1]):
        for yellow in candidates[1]:
            for green in candidates[2]:
                diameter = float(np.median([red[2], yellow[2], green[2]]))
                gap1, gap2 = yellow[0] - red[0], green[0] - yellow[0]
                if not (
                    1.1 * diameter < gap1 < 3 * diameter and 0.65 < gap2 / gap1 < 1.5
                ):
                    continue
                if (
                    max(red[1], yellow[1], green[1]) - min(red[1], yellow[1], green[1])
                    > diameter * 0.35
                ):
                    continue
                # Ignore image margins when measuring the horizontal boundary.
                left = max(0, round(red[0] - 1.8 * diameter))
                right = width - left
                if right - left < width * 0.65:
                    continue
                start = round(red[1] + 3 * diameter)
                end = min(height - 1, round(red[1] + 10 * diameter))
                edges = []
                for y in range(start, end):
                    above = np.median(rgb[max(0, y - 3) : y, left:right], axis=0)
                    below = np.median(rgb[y : min(height, y + 3), left:right], axis=0)
                    difference = np.max(np.abs(below - above), axis=1)
                    coverage = float(np.mean(difference > 25))
                    if coverage < 0.7:
                        continue
                    # Browser toolbars are predominantly neutral gray. A site
                    # navigation boundary after a colored navbar is rejected.
                    neutral = np.max(above, axis=1) - np.min(above, axis=1)
                    if float(np.mean(neutral < 22)) < 0.8:
                        continue
                    edges.append((coverage * float(np.median(difference)), y))
                if edges:
                    _, boundary = max(edges)
                    # Choose the strongest row transition, not its 3-row halo.
                    low, high = max(1, boundary - 2), min(height, boundary + 3)
                    boundary = max(
                        range(low, high),
                        key=lambda y: float(
                            np.median(
                                np.max(
                                    np.abs(rgb[y, left:right] - rgb[y - 1, left:right]),
                                    axis=1,
                                )
                            )
                        ),
                    )
                    return round(boundary / scale)
    return None


def window_bounds(image: Image.Image) -> tuple[int, int, int, int]:
    """Ignore outer translucent window shadows; preserve opaque content."""
    alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
    ys, xs = np.nonzero(alpha >= 250)
    if not len(xs):
        raise ValueError("불투명한 창 영역을 찾을 수 없습니다.")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def clean_corners(image: Image.Image, radius: int) -> Image.Image:
    """Mask all four corners, removing old shadow wedges outside each arc."""
    result = image.convert("RGBA")
    radius = min(radius, result.width // 2, result.height // 2)
    if radius <= 0:
        return result
    mask = Image.new("L", (radius * 4, radius * 4), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, radius * 8 - 1, radius * 8 - 1), fill=255)
    mask = mask.resize((radius, radius), Image.Resampling.LANCZOS)
    alpha = result.getchannel("A")
    for right, bottom in ((False, False), (True, False), (False, True), (True, True)):
        corner = ImageOps.mirror(mask) if right else mask
        if bottom:
            corner = ImageOps.flip(corner)
        x = result.width - radius if right else 0
        y = result.height - radius if bottom else 0
        existing = np.asarray(alpha.crop((x, y, x + radius, y + radius)))
        alpha.paste(Image.fromarray(np.minimum(existing, np.asarray(corner))), (x, y))
    result.putalpha(alpha)
    return result


def prepare_image(
    image: Image.Image, top: int, radius: int | None = None, max_width: int = 1920
) -> Image.Image:
    if not 0 <= top < image.height:
        raise ValueError(f"--top은 0~{image.height - 1}px 범위여야 합니다.")
    cropped = image.convert("RGBA").crop((0, top, image.width, image.height))
    cropped = cropped.crop(window_bounds(cropped))
    # 36px for the 3420px-wide reference windows; scale with the source.
    radius = max(1, round(cropped.width * 36 / 3420)) if radius is None else radius
    result = clean_corners(cropped, radius)
    if max_width and result.width > max_width:
        result = result.resize(
            (max_width, max(1, round(result.height * max_width / result.width))),
            Image.Resampling.LANCZOS,
        )
    return result


def process_file(
    path: Path, top: int | None, radius: int | None, max_width: int, dry_run: bool
) -> str:
    target = path.with_suffix(".webp")
    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) > 1:
            return "건너뜀: 다중 프레임 이미지"
        image = ImageOps.exif_transpose(source)
        cut = detect_top(image) if top is None else top
        if cut is None:
            return "건너뜀: 크롬 자동 감지 불확실 (--top 픽셀값, 이미 잘랐다면 --top 0)"
        result = prepare_image(image, cut, radius, max_width)
        description = f"{target.name}: 상단 {cut}px 제거 → {result.width}×{result.height}, 투명 WebP"
        if dry_run:
            return "확인: " + description
        profile = source.info.get("icc_profile")
    fd, name = tempfile.mkstemp(
        prefix=".screenshot-", suffix=".webp", dir=target.parent
    )
    os.close(fd)
    try:
        options = {"icc_profile": profile} if profile else {}
        result.save(name, "WEBP", lossless=True, exact=True, method=4, **options)
        os.chmod(name, (target if target.exists() else path).stat().st_mode & 0o777)
        os.replace(name, target)
    finally:
        Path(name).unlink(missing_ok=True)
    return "저장: " + description


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", nargs="?", help="이미지 파일 또는 폴더")
    parser.add_argument(
        "--top", type=int, help="상단 제거 높이(px), 0은 크롬 제거 생략"
    )
    parser.add_argument(
        "--radius", type=int, help="모서리 반지름(px), 기본값은 창 너비에 비례"
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1920,
        help="최대 출력 너비, 기본 1920px; 0은 원본 크기",
    )
    parser.add_argument("--recursive", "-r", action="store_true", help="하위 폴더 포함")
    parser.add_argument(
        "--dry-run", action="store_true", help="파일 저장 없이 결과 확인"
    )
    args = parser.parse_args()
    if any(
        value is not None and value < 0
        for value in (args.top, args.radius, args.max_width)
    ):
        parser.error("높이·반지름·최대 너비는 0 이상이어야 합니다.")
    path = (
        Path(args.path or input("이미지 파일 또는 폴더 경로: ").strip().strip("\"'"))
        .expanduser()
        .resolve()
    )
    if not path.exists():
        parser.error(f"경로가 없습니다: {path}")
    paths = (
        list(path.rglob("*") if args.recursive else path.iterdir())
        if path.is_dir()
        else [path]
    )
    # A PNG and its generated WebP must not process the same output twice.
    targets = {}
    for item in sorted(paths, key=lambda p: (p.suffix.lower() == ".webp", str(p))):
        if (
            item.is_file()
            and not item.is_symlink()
            and item.suffix.lower() in EXTENSIONS
            and not item.name.startswith(".")
        ):
            targets.setdefault(item.with_suffix(".webp"), item)
    failed = 0
    for item in targets.values():
        try:
            print(
                f"{item}: {process_file(item, args.top, args.radius, args.max_width, args.dry_run)}",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            print(f"{item}: 오류: {exc}", flush=True)
    print(f"대상 {len(targets)}개, 오류 {failed}개")
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
