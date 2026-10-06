"""End-to-end check quality on synthetic faults (also what `bloom-eval validate` reports)."""
import shutil

import pytest

from bloom_eval.validate import validate


@pytest.fixture(scope="module")
def res():
    return validate(6)


def test_no_false_positives_on_clean(res):
    for check in ("color", "logo"):
        assert res["clean"][check]["clean_fail"] == 0


def test_color_and_logo_faults_caught(res):
    assert res["faults"]["hue_shift"]["color"]["caught"] == 6
    assert res["faults"]["logo_stretched"]["logo"]["caught"] == 6
    assert res["faults"]["logo_missing"]["logo"]["caught"] == 6


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract binary not installed")
def test_text_typos_caught(res):
    assert res["clean"]["text"]["clean_fail"] == 0
    assert res["faults"]["headline_typo"]["text"]["caught"] == 6
