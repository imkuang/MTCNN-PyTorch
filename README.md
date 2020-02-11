# MTCNN PyTorch

MTCNN 推理阶段的 PyTorch 实现。主要代码参考自 [mtcnn-pytorch](https://github.com/TropComplique/mtcnn-pytorch)，做了一些小调整以优化使用、兼容 PyTorch 1.4.0。  
原作者论文：[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks.](https://arxiv.org/abs/1604.02878)

## 示例效果

![](https://raw.githubusercontent.com/xirikm/mtcnn-pytorch/master/images/example1_result.jpg)

更多示例效果可以查看[images文件夹](https://github.com/xirikm/mtcnn-pytorch/tree/master/images)。

## 使用方式

```python
from mtcnn import detect_faces, draw_bboxes, crop_img
from PIL import Image

# 检测人脸
image = Image.open("image.jpg")
bboxes, landmarks = detect_faces(image)

# 绘制标注图
drawed_image = draw_bboxes(image, bboxes, landmarks)
drawed_image.save("drawed_image.jpg")

# 裁剪人脸图片
face_img_list = crop_img(image, bboxes, resize=True, crop_size=(64, 64))
for i in range(len(face_img_list)):
    face_img_list[i].save("face_"+str(i+1)+".jpg")
```

其中`bboxes`是一个n\*5的列表、`landmarks`是一个n\*10的列表，n表示检测出来的人脸的个数，详细情况如下：

- bbox：[左上角x坐标, 左上角y坐标, 右下角x坐标, 右下角y坐标, 检测评分]
- landmark：[右眼x, 左眼x, 鼻子x, 右嘴角x, 左嘴角x, 右眼y, 左眼y, 鼻子y, 右嘴角y, 左嘴角y]

## 依赖

- pytorch 1.4.0
- Pillow
- numpy

## 感谢

本项目实现离不开以下项目的启发，特此感谢：

- [kpzhang93/MTCNN_face_detection_alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
- [TropComplique/mtcnn-pytorch](https://github.com/TropComplique/mtcnn-pytorch)
