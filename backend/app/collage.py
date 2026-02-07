from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .retrieval import RetrievalHit



def _fit_tile(image: Image.Image, tile_size: tuple[int, int]) -> Image.Image:
    tile = ImageOps.pad(image.convert("RGB"), tile_size, method=Image.Resampling.BICUBIC, color=(8, 8, 8))
    return tile



def build_collage_jpeg(
    *,
    query_rgb,
    topk_hits: list[RetrievalHit],
    tile_w: int = 240,
    tile_h: int = 240,
) -> bytes:
    if len(topk_hits) == 0:
        topk_hits = []

    font = ImageFont.load_default()

    query_tile = _fit_tile(Image.fromarray(query_rgb), (tile_w, tile_h))
    right_tiles: list[Image.Image] = []

    for idx, hit in enumerate(topk_hits, start=1):
        path = Path(hit.exemplar_path)
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        tile = _fit_tile(img, (tile_w, tile_h))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, tile_h - 26, tile_w, tile_h), fill=(0, 0, 0))
        draw.text((8, tile_h - 21), f"{idx}. {hit.letter} ({hit.score:.3f})", fill=(255, 255, 255), font=font)
        right_tiles.append(tile)

    columns = 1 + max(1, len(right_tiles))
    canvas = Image.new("RGB", (columns * tile_w, tile_h + 30), (18, 20, 24))

    draw = ImageDraw.Draw(canvas)
    canvas.paste(query_tile, (0, 30))
    draw.text((10, 7), "QUERY", fill=(255, 255, 255), font=font)

    if right_tiles:
        for i, tile in enumerate(right_tiles, start=1):
            canvas.paste(tile, (i * tile_w, 30))
        draw.text((tile_w + 10, 7), "TOP-K ETALONS", fill=(255, 255, 255), font=font)
    else:
        placeholder = Image.new("RGB", (tile_w, tile_h), (30, 30, 30))
        ph_draw = ImageDraw.Draw(placeholder)
        ph_draw.text((12, 12), "NO TOP-K", fill=(220, 220, 220), font=font)
        canvas.paste(placeholder, (tile_w, 30))

    buffer = BytesIO()
    canvas.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()
