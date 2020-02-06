from PIL import ImageDraw, Image


def draw_bboxes(img, bounding_boxes, facial_landmarks=[]):
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


def crop_img(img, bounding_boxes, size=(64, 64)):
    img_list = []
    for b in bounding_boxes:
        face_img = img.crop(((b[0], b[1], b[2], b[3])))
        face_img = face_img.resize(size, Image.BILINEAR)
        img_list.append(face_img)
    return img_list
