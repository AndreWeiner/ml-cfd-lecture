"""Rotate the attributed TUHH GIF 90 degrees clockwise for page display.

Reproduce: python3 scripts/rotate_taylor_bubble_animation.py
Original retained; decoded frame colors, frame durations, and loop count checked.
Rounded corners and shadow belong to quarto.css, not the scientific image.
"""
from pathlib import Path
from PIL import Image, ImageSequence, ImageChops

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'notebooks/media/reactive-oxygen-taylor-bubble-tuhh.gif'
target = source.with_name(source.stem + '-horizontal.gif')
with Image.open(source) as animation:
    loop = animation.info.get('loop', 0)
    frames, durations = [], []
    for frame in ImageSequence.Iterator(animation):
        frames.append(frame.convert('RGB').transpose(Image.Transpose.ROTATE_270))
        durations.append(frame.info.get('duration', 0))
frames[0].save(target, save_all=True, append_images=frames[1:], duration=durations,
               loop=loop, disposal=2, optimize=False)
with Image.open(target) as check:
    assert check.n_frames == len(frames)
    assert check.info.get('loop', 0) == loop
    for index, frame in enumerate(ImageSequence.Iterator(check)):
        assert frame.info.get('duration', 0) == durations[index]
        assert ImageChops.difference(frame.convert('RGB'), frames[index]).getbbox() is None
print(f'Checked {len(frames)} rotated frames, exact decoded colors, timing, and loop count.')
