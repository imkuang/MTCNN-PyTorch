# MTCNN PyTorch

MTCNN 推理阶段的 PyTorch 实现。支持 GPU（CUDA）计算，兼容 PyTorch 1.4.0。  
原作者论文：[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks.](https://arxiv.org/abs/1604.02878)

## 示例效果

![](https://raw.githubusercontent.com/xirikm/mtcnn-pytorch/master/images/drawed_image.jpg)


## 使用方式

```python
from mtcnn import FaceDetector
from PIL import Image

# 人脸检测对象。优先使用GPU进行计算（会自动判断GPU是否可用）
# 你也可以通过设置 FaceDetector("cpu") 或者 FaceDetector("cuda") 手动指定计算设备
detector = FaceDetector()

image = Image.open("./images/image.jpg")

# 检测人脸
# 其中bboxes是一个n*5的列表、landmarks是一个n*10的列表，n表示检测出来的人脸个数，数据详细情况如下：
# bbox：[左上角x坐标, 左上角y坐标, 右下角x坐标, 右下角y坐标, 检测评分]
# landmark：[右眼x, 左眼x, 鼻子x, 右嘴角x, 左嘴角x, 右眼y, 左眼y, 鼻子y, 右嘴角y, 左嘴角y]
bboxes, landmarks = detector.detect(image)

# 绘制并保存标注图
drawed_image = detector.draw_bboxes(image)
drawed_image.save("./images/drawed_image.jpg")

# 裁剪人脸图片并保存
face_img_list = detector.crop_image(image, resize=True, crop_size=(64, 64))
for i in range(len(face_img_list)):
    face_img_list[i].save("./images/face_" + str(i + 1) + ".jpg")
```

将上述代码保存为 `demo.py`，运行后可以在[imges文件夹](https://github.com/xirikm/mtcnn-pytorch/tree/master/demo)下查看保存的图片。

## 依赖

- PyTorch 1.4.0
- Pillow
- NumPy

## 感谢

本项目实现离不开以下项目的启发，特此感谢：

- [kpzhang93/MTCNN_face_detection_alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
- [TropComplique/mtcnn-pytorch](https://github.com/TropComplique/mtcnn-pytorch)
