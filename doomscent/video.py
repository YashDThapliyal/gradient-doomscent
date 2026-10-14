"""H.264 MP4 writing through PyAV, plus Pillow drawing helpers for tags and panels."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

import av
import numpy as np
from PIL import Image, ImageDraw

from doomscent.style import DARK, Theme, font, hex_to_rgb


class VideoWriter:
    """Write RGB uint8 frames (H, W, 3) to an H.264 MP4."""

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: int = 35,
        crf: int = 20,
        gop: int | None = None,
    ):
        if width % 2 or height % 2:
            raise ValueError("H.264 needs even frame sizes")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_name(self.path.stem + ".partial.mp4")
        self.container = av.open(str(self.tmp), mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": str(crf), "preset": "medium"}
        if gop:
            self.stream.codec_context.gop_size = gop  # keyframe spacing, for fast seeking
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self.stream.height, self.stream.width):
            raise ValueError(f"frame {frame.shape} does not match the writer size")
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
        for pkt in self.stream.encode(vf):
            self.container.mux(pkt)
        self.frames += 1

    def close(self) -> None:
        for pkt in self.stream.encode():
            self.container.mux(pkt)
        self.container.close()
        self.tmp.replace(self.path)  # only complete files get the final name

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self.container.close()


def probe_video(path: str | Path) -> dict[str, float]:
    """Decode a clip end to end; returns frame count, size, fps, and duration."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        n = sum(1 for _ in c.decode(s))
        return {
            "frames": n,
            "width": s.codec_context.width,
            "height": s.codec_context.height,
            "fps": float(s.average_rate),
            "duration_s": n / float(s.average_rate),
        }


def iter_frames(path: str | Path) -> Iterator[np.ndarray]:
    """Decode a clip lazily, one RGB frame at a time."""
    with av.open(str(path)) as c:
        for f in c.decode(c.streams.video[0]):
            yield f.to_ndarray(format="rgb24")


class FrameReader:
    """Random access to a clip's frames: sequential reads are cheap, jumps seek to a keyframe."""

    def __init__(self, path: str | Path):
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.rate = float(self.stream.average_rate)
        self.tb = float(self.stream.time_base)
        self.n = self.stream.frames or int(probe_video(path)["frames"])
        self._iter = None
        self._pos = -1
        self._frame: np.ndarray | None = None

    def _index(self, frame: av.VideoFrame) -> int:
        return round(frame.pts * self.tb * self.rate)

    def get(self, i: int) -> np.ndarray:
        i = max(0, min(int(i), self.n - 1))
        if self._frame is not None and i == self._pos:
            return self._frame
        if self._iter is None or i < self._pos or i > self._pos + 70:
            self.container.seek(int(i / self.rate / self.tb), stream=self.stream, backward=True)
            self._iter = self.container.decode(self.stream)
            self._pos = -1
        for f in self._iter:
            self._pos = self._index(f)
            if self._pos >= i:
                self._frame = f.to_ndarray(format="rgb24")
                return self._frame
        assert self._frame is not None, "empty clip"
        return self._frame

    def close(self) -> None:
        self.container.close()


def read_frames(path: str | Path) -> list[np.ndarray]:
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]


def draw_tag(
    img: np.ndarray,
    lines: list[str],
    corner: str = "tl",
    theme: Theme = DARK,
    size: int = 22,
    margin: int = 16,
    alpha: float = 0.78,
) -> np.ndarray:
    """A small corner label: semi-opaque panel, first line semibold, rest regular."""
    pil = Image.fromarray(img).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    fonts = [font(size, "semibold")] + [font(int(size * 0.82), "regular")] * (len(lines) - 1)
    pad, gap = int(size * 0.5), int(size * 0.25)
    widths = [d.textbbox((0, 0), t, font=f)[2] for t, f in zip(lines, fonts, strict=True)]
    heights = [f.size + gap for f in fonts]
    w, h = max(widths) + 2 * pad, sum(heights) - gap + 2 * pad
    x = margin if corner[1] == "l" else pil.width - w - margin
    y = margin if corner[0] == "t" else pil.height - h - margin
    bg = (*hex_to_rgb(theme.background), int(255 * alpha))
    d.rounded_rectangle([x, y, x + w, y + h], radius=int(size * 0.3), fill=bg)
    ty = y + pad
    for i, (t, f) in enumerate(zip(lines, fonts, strict=True)):
        color = theme.text if i == 0 else theme.subtext
        d.text((x + pad, ty), t, font=f, fill=(*hex_to_rgb(color), 255))
        ty += heights[i]
    return np.asarray(Image.alpha_composite(pil, overlay).convert("RGB"))


def paste(dst: np.ndarray, src: np.ndarray, x: int, y: int, alpha: float = 1.0) -> np.ndarray:
    """Paste src onto a copy of dst at (x, y), optionally blended."""
    out = dst.copy()
    h, w = src.shape[:2]
    region = out[y : y + h, x : x + w]
    if alpha >= 1.0:
        region[:] = src
    else:
        region[:] = (alpha * src + (1 - alpha) * region).astype(np.uint8)
    return out
