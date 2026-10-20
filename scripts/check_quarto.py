"""Check Python syntax, saved figures, source downloads, and local website links."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import ast
import json
import re

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []
    def handle_starttag(self, tag, attrs):
        self.urls.extend(v for k, v in attrs if k in ('href', 'src') and v)


def main():
    manifest = json.loads((ROOT / 'quarto-migration.json').read_text())
    existing = {item['slug'] for item in manifest}
    manifest.extend({'slug': slug} for slug in ('ml_intro_exercise', 'python_intro', 'neural_network_training')
                    if slug not in existing)
    code_count = 0
    image_count = 0
    for item in manifest:
        slug = item['slug']
        path = ROOT / 'notebooks' / f'{slug}.py'
        ast.parse(path.read_text())
        blocks = re.split(r'^# %%', path.read_text(), flags=re.M)[1:]
        code_count += sum('[markdown]' not in block.partition('\n')[0] for block in blocks)
        frozen = ROOT / '_freeze' / 'notebooks' / slug / 'execute-results/html.json'
        assert json.loads(frozen.read_text())['result']['engine'] == 'jupyter', slug
        assert (ROOT / 'docs/notebooks' / f'{slug}.html').exists(), slug
        assert (ROOT / 'docs/notebooks' / f'{slug}.py').read_bytes() == path.read_bytes(), slug
    figures = [figure for item in manifest for figure in (ROOT / '_freeze' / 'notebooks' / item['slug'] / 'figure-html').glob('*')]
    image_count = len(figures)
    broken = []
    for page in (ROOT / 'docs').rglob('*.html'):
        parser = Links(); parser.feed(page.read_text())
        for url in parser.urls:
            parts = urlsplit(url)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            path = unquote(parts.path)
            target = ROOT / 'docs' / path.lstrip('/') if path.startswith('/') else page.parent / path
            if not target.exists():
                broken.append((str(page.relative_to(ROOT)), url))
    assert not broken, broken
    print(f'Checked {len(manifest)} chapters, {code_count} code cells, {image_count} saved figures, and all local links.')


if __name__ == '__main__':
    main()
