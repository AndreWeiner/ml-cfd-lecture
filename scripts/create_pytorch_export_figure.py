"""Run with Python 3 to generate the two SVGs in notebooks/media. Uses native SVG sans-serif text."""
from html import escape
from pathlib import Path

rows = [
    ("AOTInductor · Lecture 5", "#1f77b4",
     ("torch.export → AOTInductor", "Capture, compile and package"),
     (".pt2 package", "Compiled model and weights"),
     ("LibTorch + AOTI loader", "Run inside OpenFOAM")),
    ("TorchScript · legacy", "#ff7f0e",
     ("torch.jit.script / trace", "Capture and save TorchScript"),
     (".pt archive", "TorchScript model and weights"),
     ("LibTorch + torch::jit::load", "Run the TorchScript module")),
    ("ONNX · exchange format", "#2ca02c",
     ("torch.onnx.export", "Convert to an ONNX graph"),
     (".onnx model", "Graph and weights"),
     ("ONNX Runtime C++ API", "Load and run the ONNX graph")),
]

def draw(theme, bg, fg, neutral):
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="430" viewBox="0 0 1060 430">',
           '<title>Three PyTorch export paths to C++</title>',
           f'<rect width="1060" height="430" fill="{bg}"/>',
           f'<g font-family="DejaVu Sans, sans-serif" fill="{fg}">']

    def label(x, y, value, size, weight=400):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}">{escape(value)}</text>')

    for x, title in [(30, "Export in Python"), (387, "Exported artifact"), (706, "Load and run in C++")]:
        label(x, 33, title, 18, 700)
    for i, (name, color, *cells) in enumerate(rows):
        y = 73 + i * 127
        svg.append(f'<rect x="30" y="{y-15}" width="6" height="18" fill="{color}"/>')
        label(45, y, name, 20, 700)
        for (x, width), (title, detail) in zip([(30,314),(387,276),(706,324)], cells):
            svg.append(f'<rect x="{x}" y="{y+12}" width="{width}" height="65" rx="7" fill="none" stroke="{neutral}"/>')
            label(x+12,y+38,title,18,700)
            label(x+12,y+61,detail,16)
        for x in (352,671):
            h=y+44
            svg.append(f'<path d="M{x} {h} h26 m-7 -6 l7 6 -7 6" fill="none" stroke="{neutral}" stroke-width="2"/>')
    svg.append('</g></svg>')
    (Path(__file__).resolve().parents[1] / "notebooks" / "media" / f"pytorch-to-cpp-{theme}.svg").write_text('\n'.join(svg))

draw("light", "#ffffff", "#171717", "#676767")
draw("dark", "#111111", "#f4f4f4", "#a0a0a0")
