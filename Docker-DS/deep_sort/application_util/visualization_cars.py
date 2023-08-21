import colorsys
import cv2
import numpy as np

l = set()

def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


def draw_rectangle(x, y, w, h, image, label=None, thickness = 2, color = (0,0,255)):
    """Draw a rectangle.

    Parameters
    ----------
    x : float | int
        Top left corner of the rectangle (x-axis).
    y : float | int
        Top let corner of the rectangle (y-axis).
    w : float | int
        Width of the rectangle.
    h : float | int
        Height of the rectangle.
    label : Optional[str]
        A text label that is placed at the top left corner of the
        rectangle.

    """
    #region_1 = [[55,380], [620,260], [623,253], [60,370]]
    region_1 = [[45,475], [730,320], [710,240], [30,350]]
    region_1 = np.array(region_1)
    region_1 = region_1.reshape((-1, 1, 2))


    pt1 = int(x), int(y)
    pt2 = int(x + w), int(y + h)
    cv2.rectangle(image, pt1, pt2, color, thickness)
    if label is not None:
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_PLAIN, 1, thickness)

        center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
        pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
        cv2.rectangle(image, pt1, pt2, color, -1)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness)
                

        inside_region = cv2.pointPolygonTest(np.array(region_1), (int(x), int(h)), False)

        if inside_region > 0:
            l.add(label)

        cv2.putText(image, "Cars:" + str(len(l)), (20,160), 0, 1.5, (0, 255, 255), 2)

        cv2.polylines(image, [np.array(region_1)], True, (0, 255, 255), 4)
        print(l)
        cv2.imshow("frame", image)
        #print(label)

def draw_detections(detections, image):
    thickness = 2
    color = 0, 0, 255
    for i, detection in enumerate(detections):
        draw_rectangle(*detection.tlwh, image, thickness=thickness, color=color)

def draw_trackers(tracks, image):
    thickness = 2
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        color = create_unique_color_uchar(track.track_id)
        draw_rectangle(*track.to_tlwh(), image,
                  label=str(track.track_id), thickness=thickness, color=color)
