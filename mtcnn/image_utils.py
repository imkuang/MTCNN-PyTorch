from PIL import ImageDraw, Image
from .box_utils import convert_to_square


def draw_bboxes(img, bounding_boxes, facial_landmarks=[]):
    """Draw bounding boxes and facial landmarks.

    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].
    Returns:
        an instance of PIL.Image.
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline="white")

    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse(
                [(p[i] - 1.0, p[i + 5] - 1.0), (p[i] + 1.0, p[i + 5] + 1.0)],
                outline="blue",
            )

    return img_copy


def crop_img(img, bounding_boxes, resize=False, crop_size=(64, 64)):
    """Crop all face pictures.

    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        resize: whether to resize the images.
        crop_size: the size of output images, this parameter is required if you set resized = True
    Returns:
        a list of PIL.Image instances
    """
    img_list = []
    # convert bboxes to square
    square_bboxes = convert_to_square(bounding_boxes)
    for b in square_bboxes:
        face_img = img.crop((b[0], b[1], b[2], b[3]))
        if resize:
            face_img = face_img.resize(crop_size, Image.BILINEAR)
        img_list.append(face_img)
    return img_list
