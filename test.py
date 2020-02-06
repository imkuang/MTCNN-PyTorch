from PIL import Image
from mtcnn import detect_faces, draw_bboxes, crop_img


if __name__ == "__main__":
    # draw bboxes
    for i in range(3):
        example_img = Image.open("./images/example" + str(i + 1) + ".jpg")
        bboxes, landmarks = detect_faces(example_img)
        drawed_img = draw_bboxes(example_img, bboxes, landmarks)
        drawed_img.save("./images/example" + str(i + 1) + "_result.jpg")

    example_img = Image.open("./images/example3.jpg")
    bboxes, _ = detect_faces(example_img)
    face_img_list = crop_img(example_img, bboxes, (64, 64))
    for i in range(len(face_img_list)):
        face_img_list[i].save("./images/example3_croped/" + str(i + 1) + ".jpg")
