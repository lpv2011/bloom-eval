import numpy as np
from PIL import Image

from bloom_eval.checks.color import check_color, ciede2000, srgb_to_lab

# Reference pairs from Sharma, Wu, Dalal (2005) CIEDE2000 test data
SHARMA = [
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
    ((50.0, 2.5, 0.0), (50.0, 0.0, -2.5), 4.3065),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
]


def test_ciede2000_matches_reference_values():
    for a, b, expected in SHARMA:
        assert abs(float(ciede2000(np.array(a), np.array(b))) - expected) < 1e-3


def test_lab_of_white_and_black():
    lab = srgb_to_lab(np.array([[255, 255, 255], [0, 0, 0]], dtype=float))
    assert abs(lab[0, 0] - 100) < 0.1 and abs(lab[1, 0]) < 0.1


def test_color_check_passes_on_palette_and_fails_off_palette():
    pal = ["#0B3D2E", "#F4E9D8"]
    good = Image.new("RGB", (64, 64), "#0B3D2E")
    bad = Image.new("RGB", (64, 64), "#E040FB")  # saturated magenta, nowhere near the palette
    assert check_color(good, pal).passed is True
    assert check_color(bad, pal).passed is False


def test_color_check_uncertain_without_palette():
    assert check_color(Image.new("RGB", (8, 8)), []).passed is None
