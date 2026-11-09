"""Create a source-only download, excluding build products and existing runs."""
from pathlib import Path
import zipfile
root = Path(__file__).resolve().parents[1]
source = root / "test_cases/single_phase_approach"
with zipfile.ZipFile(root / "downloads/bubble-single-phase.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if not path.is_file() or any(part in {"build", "lnInclude", ".vscode"}
                                    or part.startswith("linux") for part in relative.parts):
            continue
        if path.suffix in {".so", ".o", ".dep"}:
            continue
        archive.write(path, Path("single_phase_approach") / relative)
