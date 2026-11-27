"""White or not: measured in the image, not asked.

The labeling guideline defines a white backdrop by a number: brightness 250 or more on the 0 to 255 scale.
A language model cannot tell 247 from 251, a pixel measurement can.
"""

import numpy as np

WHITE = 250  # the threshold of the labeling guideline
PATCH = 0.03  # side of a corner patch, as a share of the image's shorter side
STEPS_INSIDE = (0.0, 0.01, 0.02, 0.04)  # how far to step in from the border, as a share of the shorter side


def brightness(rgb_image):
    """Brightness per pixel, 0 to 255."""
    return np.asarray(rgb_image, dtype=np.float32).mean(axis=2)


def corner_medians(gray, step=0.0):
    """Median brightness of the four corner patches, sorted from dark to bright."""
    height, width = gray.shape
    side = max(2, int(PATCH * min(height, width)))
    offset = int(round(step * min(height, width)))
    patches = [
        gray[offset : offset + side, offset : offset + side],
        gray[offset : offset + side, width - offset - side : width - offset],
        gray[height - offset - side : height - offset, offset : offset + side],
        gray[height - offset - side : height - offset, width - offset - side : width - offset],
    ]
    return sorted(float(np.median(patch)) for patch in patches)


def corners_are_white(gray, step=0.0):
    """True when at least two of the four corner patches are white. The product may cover the other two."""
    return corner_medians(gray, step)[2] >= WHITE


def white_backdrop(rgb_image, tolerate_frame=True):
    """True when the backdrop is white.

    With tolerate_frame the corners are also checked a few steps inside the border, and one white position is
    enough. A thin frame around the image then no longer turns a white backdrop into a colored one.
    """
    gray = brightness(rgb_image)
    steps = STEPS_INSIDE if tolerate_frame else (0.0,)
    return any(corners_are_white(gray, step) for step in steps)


def with_frame(gray, width, value=0.0):
    """A copy of the brightness image with a frame of the given width drawn around it. Used to test the rule."""
    framed = gray.copy()
    framed[:width] = framed[-width:] = value
    framed[:, :width] = framed[:, -width:] = value
    return framed
