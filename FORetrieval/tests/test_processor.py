from foretrieval.generic_processor import text_to_image
from PIL import Image


def test_text_to_image():
    text = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.\n"
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n"
        " Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.\n"
        " Duis"
    )
    image = text_to_image(text)
    assert isinstance(image, Image.Image)
    assert image.size == (800, 600)
    assert image.mode == "RGB"
    # show image
    image.show()


if __name__ == "__main__":
    test_text_to_image()
    print("All tests passed.")
