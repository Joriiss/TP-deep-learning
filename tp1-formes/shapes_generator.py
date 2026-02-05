"""
Interactive shape generator: triangles, squares, circles, rectangles, and stars.
User inputs shape type, color (hex), size, and position; can add multiple shapes.
Final image is generated and saved.
"""

from PIL import Image, ImageDraw
import math
from pathlib import Path


def hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
    """Convert hex color (e.g. #FF0000 or FF0000) to RGB tuple."""
    hex_code = hex_code.lstrip("#")
    if len(hex_code) != 6:
        raise ValueError("Hex code must be 6 characters (e.g. #FF0000)")
    return tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))


def draw_triangle(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int, color: tuple) -> None:
    """Draw an equilateral triangle centered at (cx, cy) with given size (side length)."""
    h = size * math.sqrt(3) / 2  # height of equilateral triangle
    # Top, bottom-left, bottom-right
    top = (cx, cy - h / 2)
    left = (cx - size / 2, cy + h / 2)
    right = (cx + size / 2, cy + h / 2)
    draw.polygon([top, left, right], fill=color, outline=color)


def draw_square(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int, color: tuple) -> None:
    """Draw a square centered at (cx, cy) with given side length."""
    half = size / 2
    bbox = (cx - half, cy - half, cx + half, cy + half)
    draw.rectangle(bbox, fill=color, outline=color)


def draw_circle(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int, color: tuple) -> None:
    """Draw a circle centered at (cx, cy) with given diameter (size)."""
    half = size / 2
    bbox = (cx - half, cy - half, cx + half, cy + half)
    draw.ellipse(bbox, fill=color, outline=color)


def draw_rectangle(
    draw: ImageDraw.ImageDraw, cx: int, cy: int, width: int, height: int, color: tuple
) -> None:
    """Draw a rectangle centered at (cx, cy) with given width and height (pixels)."""
    x1 = cx - width / 2
    y1 = cy - height / 2
    bbox = (x1, y1, x1 + width, y1 + height)
    draw.rectangle(bbox, fill=color, outline=color)


def draw_star(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int, color: tuple) -> None:
    """Draw a 5-pointed star centered at (cx, cy). size = outer radius in pixels."""
    n = 5
    inner_r = size * 0.4
    points = []
    for i in range(2 * n):
        r = size if i % 2 == 0 else inner_r
        angle = math.pi / 2 + (2 * math.pi * i) / (2 * n)
        px = cx + r * math.cos(angle)
        py = cy - r * math.sin(angle)
        points.append((px, py))
    draw.polygon(points, fill=color, outline=color)


def prompt_shape() -> dict | None:
    """Ask user for one shape's attributes. Returns dict or None if invalid."""
    VALID_SHAPES = ("triangle", "square", "circle", "rectangle", "star")
    print("\n--- New shape ---")
    shape = input(f"Shape ({' / '.join(VALID_SHAPES)}): ").strip().lower()
    if shape not in VALID_SHAPES:
        print(f"Invalid shape. Use: {', '.join(VALID_SHAPES)}.")
        return None

    color_input = input("Color (hex, e.g. #FF0000): ").strip()
    try:
        color = hex_to_rgb(color_input)
    except (ValueError, TypeError) as e:
        print(f"Invalid hex color: {e}")
        return None

    if shape == "rectangle":
        try:
            size = float(input("Width (pixels): "))
            size2 = float(input("Height (pixels): "))
            if size <= 0 or size2 <= 0:
                raise ValueError("Width and height must be positive")
        except ValueError as e:
            print(f"Invalid size: {e}")
            return None
    else:
        try:
            size = float(
                input(
                    "Size (side/diameter/radius for triangle/square/circle/star): "
                )
            )
            size2 = size
            if size <= 0:
                raise ValueError("Size must be positive")
        except ValueError as e:
            print(f"Invalid size: {e}")
            return None

    try:
        x = float(input("Position X (center): "))
        y = float(input("Position Y (center): "))
    except ValueError as e:
        print(f"Invalid position: {e}")
        return None

    out = {"shape": shape, "color": color, "size": size, "x": x, "y": y}
    if shape == "rectangle":
        out["size2"] = size2
    return out


def generate_image(shapes: list[dict], width: int = 800, height: int = 600) -> Image.Image:
    """Create a white background image and draw all shapes."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    for s in shapes:
        cx, cy = int(s["x"]), int(s["y"])
        size = int(s["size"])
        color = s["color"]
        if s["shape"] == "triangle":
            draw_triangle(draw, cx, cy, size, color)
        elif s["shape"] == "square":
            draw_square(draw, cx, cy, size, color)
        elif s["shape"] == "circle":
            draw_circle(draw, cx, cy, size, color)
        elif s["shape"] == "rectangle":
            height = int(s.get("size2", size))
            draw_rectangle(draw, cx, cy, size, height, color)
        elif s["shape"] == "star":
            draw_star(draw, cx, cy, size, color)

    return img


def main() -> None:
    print("Shape generator — add triangles, squares, circles, rectangles, and stars.")
    print("Image size: 800x600. Positions are in pixels (center of shape).\n")

    shapes: list[dict] = []

    while True:
        one = prompt_shape()
        if one is not None:
            shapes.append(one)
            print("Shape added.")

        again = input("Add another shape? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            break

    if not shapes:
        print("No shapes added. Exiting.")
        return

    img = generate_image(shapes)
    out_path = Path(__file__).parent / "tp-formes_output.png"
    img.save(out_path)
    print(f"\nImage saved as: {out_path}")


if __name__ == "__main__":
    main()
