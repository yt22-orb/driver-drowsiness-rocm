from PIL import Image

from drowsiness.audit import difference_hash, image_digest


def test_image_hashes_are_deterministic() -> None:
    image = Image.new("RGB", (16, 16), color=(10, 20, 30))
    assert image_digest(image) == image_digest(image.copy())
    assert difference_hash(image) == difference_hash(image.copy())
