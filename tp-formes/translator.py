"""
Natural language → shape instructions for the shape generator.
Uses Gemini (API key from project root .env) to translate user text into
shape list, then generates the image via shapes_generator.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# Load .env from project root (parent of tp-formes)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

from shapes_generator import generate_image, hex_to_rgb

SYSTEM_PROMPT = """You translate natural language into shape-drawing instructions.
Canvas size: 800 x 600 pixels. Origin (0,0) is top-left.

Output ONLY a valid JSON array. No markdown, no explanation. Each element is an object with:
- "shape": one of "triangle", "square", "circle", "rectangle", "star"
- "color": hex string (e.g. "#FF0000" for red, "#0000FF" for blue)
- "size": number in pixels. Use: small ≈ 50, medium ≈ 100, large ≈ 150. For rectangle this is width; for star this is outer radius.
- "size2": only for "rectangle" — height in pixels (omit for other shapes)
- "x": number (center X in pixels)
- "y": number (center Y in pixels)

Position hints (approximate center):
- top left: (150, 150)
- top right: (650, 150)
- middle / center: (400, 300)
- bottom left: (150, 450)
- bottom right: (650, 450)
"""


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex string (e.g. #FF0000)."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def extract_json(text: str) -> str:
    """Get JSON from model output, stripping markdown code blocks if present."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def translate_to_shapes(natural_language: str, api_key: str | None = None) -> list[dict]:
    """Call Gemini to convert natural language into a list of shape dicts (with RGB color)."""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=natural_language,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
    )
    raw = response.text
    json_str = extract_json(raw)
    data = json.loads(json_str)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of shapes")

    shapes = []
    for item in data:
        if not isinstance(item, dict):
            continue
        shape = item.get("shape", "").lower()
        if shape not in ("triangle", "square", "circle", "rectangle", "star"):
            continue
        color_hex = item.get("color", "#000000")
        if isinstance(color_hex, str):
            color_hex = color_hex.strip()
        color = hex_to_rgb(color_hex)
        size = float(item.get("size", 100))
        x = float(item.get("x", 400))
        y = float(item.get("y", 300))
        s = {"shape": shape, "color": color, "size": size, "x": x, "y": y}
        if shape == "rectangle":
            s["size2"] = float(item.get("size2", size))
        shapes.append(s)

    return shapes


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set. Add it to the project root .env file.")
        return

    print("Describe the shapes you want (e.g. 'a large red square in the top right and a small blue circle in the middle')")
    user_input = input("You: ").strip()
    if not user_input:
        print("No input. Exiting.")
        return

    print("Translating with Gemini...")
    try:
        shapes = translate_to_shapes(user_input, api_key=api_key)
    except Exception as e:
        print(f"Translation failed: {e}")
        return

    if not shapes:
        print("No valid shapes were parsed. Try being more explicit (shape, color, size, position).")
        return

    print(f"Generated {len(shapes)} shape(s). Drawing...")
    img = generate_image(shapes)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = results_dir / timestamp

    img_path = base_name.with_suffix(".png")
    img.save(img_path)

    translated = []
    for s in shapes:
        t = {
            "shape": s["shape"],
            "color": rgb_to_hex(s["color"]),
            "size": s["size"],
            "x": s["x"],
            "y": s["y"],
        }
        if s["shape"] == "rectangle":
            t["size2"] = s.get("size2", s["size"])
        translated.append(t)
    result = {"prompt": user_input, "shapes": translated}
    json_path = base_name.with_suffix(".json")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Image saved as: {img_path}")
    print(f"Result saved as: {json_path}")


if __name__ == "__main__":
    main()
