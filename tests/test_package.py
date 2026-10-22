from importlib import metadata
from pathlib import Path

import segmentscan


def test_version_matches_package_metadata():
    assert segmentscan.__version__ == metadata.version("segmentscan")


def test_package_is_typed():
    assert (Path(segmentscan.__file__).parent / "py.typed").exists()


def test_public_names_exist():
    for name in segmentscan.__all__:
        assert hasattr(segmentscan, name)
