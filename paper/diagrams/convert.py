#!/usr/bin/env python3
"""Optionally convert the five supplied SVGs to vector PDFs with Typst."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
STEMS = (
    "fig01-benchmark-overview",
    "fig02-data-pipeline",
    "fig03-encoder-tokenization",
    "fig04-ssl-objectives",
    "fig05-protocol-ab",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    expected_path = ROOT / "SOURCE_MANIFEST.json"
    expected = json.loads(expected_path.read_text()) if expected_path.exists() else None
    if expected:
        for name, info in expected.get("conversion_files", {}).items():
            if sha(ROOT / name) != info["sha256"]:
                raise ValueError(f"Conversion asset differs from the pinned source: {name}")
    result = {}
    with tempfile.TemporaryDirectory(prefix=".convert-", dir=ROOT) as temporary:
        source = Path(temporary) / "figure.typ"
        for stem in STEMS:
            svg = ROOT / f"{stem}.svg"
            if expected and sha(svg) != expected["files"][stem + ".svg"]["sha256"]:
                raise ValueError(f"SVG differs from the pinned source: {svg.name}")
            element = ET.parse(svg).getroot()
            width, height = float(element.attrib["width"]), float(element.attrib["height"])
            pdf = out / f"{stem}.pdf"
            if expected and stem in expected.get("figma_converted_stems", []):
                conversion = ROOT / "figma-conversion"
                conversion_svg = svg
                if stem == "fig05-protocol-ab":
                    fragment_root = ET.parse(conversion / "stop-gradient-native-path.svg").getroot()
                    fragment = ET.tostring(fragment_root[0], encoding="unicode")
                    text, count = re.subn(
                        r'<text id="stop-gradient-label"[\s\S]*?</text>',
                        lambda _: fragment,
                        svg.read_text(),
                    )
                    if count != 1:
                        raise ValueError("Expected exactly one native stop-gradient label")
                    conversion_svg = Path(temporary) / svg.name
                    conversion_svg.write_text(text)
                subprocess.run(
                    ["typst", "compile", "--root", str(ROOT),
                     "--font-path", str(conversion / "fonts"),
                     "--input", "stem=" + stem,
                     "--input", "svg=/" + str(conversion_svg.relative_to(ROOT)),
                     "--input", "width=" + str(width),
                     "--input", "height=" + str(height),
                     str(conversion / "convert-svg.typ"), str(pdf)],
                    check=True,
                )
            else:
                source.write_text(
                    '#set document(date: none)\n'
                    '#set text(font: "Arial")\n'
                    f'#set page(width: {width}pt, height: {height}pt, margin: 0pt)\n'
                    f'#image("/{stem}.svg", width: 100%, height: 100%)\n'
                )
                subprocess.run(
                    ["typst", "compile", "--root", str(ROOT), str(source), str(pdf)],
                    check=True,
                )
            result[stem] = {
                "source_sha256": sha(svg),
                "pdf_sha256": sha(pdf),
                "matches_supplied_pdf": (
                    sha(pdf) == expected["files"][stem + ".pdf"]["sha256"]
                    if expected else None
                ),
            }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
