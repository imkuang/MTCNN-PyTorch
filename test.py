from mtcnn import FaceDetector
from PIL import Image


if __name__ == "__main__":
    # Face detection object.
    # automatically detect if the GPU is available.
    # you can specify the device by setting FaceDetector("cpu") or FaceDetector("cuda")
    detector = FaceDetector()

    # draw bboxes on example images
    for i in range(3):
        example_img = Image.open("./images/example" + str(i + 1) + ".jpg")
        drawed_img = detector.draw_bboxes(example_img)
        drawed_img.save("./images/example" + str(i + 1) + "_result.jpg")

    # crop face images from example3.jpg
    example_img = Image.open("./images/example3.jpg")
    face_img_list = detector.crop_image(example_img, resize=True, crop_size=(64, 64))
    for i in range(len(face_img_list)):
        face_img_list[i].save("./images/example3_croped/face_" + str(i + 1) + ".jpg")
