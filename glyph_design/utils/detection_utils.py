def write_file(label_file, text):
    with open(label_file, "a+") as f:
        f.write(text + "\n")
    f.close()


def is_collision(rect1, rect2):
    """
    Determines if two rectangles overlap. Each rectangle is defined by a tuple of (x, y, width, height).

    Args:
    rect1 (tuple): The x, y coordinates, width, and height of the first rectangle.
    rect2 (tuple): The x, y coordinates, width, and height of the second rectangle.

    Returns:
    bool: True if the rectangles overlap, False otherwise.
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Check if one rectangle is to the left of the other
    if x1 + w1 < x2 or x2 + w2 < x1:
        return False

    # Check if one rectangle is above the other
    if y1 + h1 < y2 or y2 + h2 < y1:
        return False

    return True

