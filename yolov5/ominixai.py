import torch
import sys
from omnixai.data.image import Image
from omnixai.explainers.vision import IntegratedGradientImage
from omnixai.preprocessing.image import Resize
from models.yolo import Model

# 加载您的训练好的图像分类模型
# 例如：model = torch.load("your_model.pth")
sys.path.append('E:/yolov5')

model = Model(cfg="E:/yolov5/models/yolov5m.yaml", nc=80)



# 加载模型权重
checkpoint = torch.load('E:/yolov5/runs/train/exp5/weights/best.pt', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])
model.eval()


# 确保模型处于评估模式
model.eval()


# 创建 IntegratedGradients 实例
ig = IntegratedGradientImage(
    pretrained_model=model,
    preprocess=Resize((640, 640)),  # 根据您的模型输入调整
    use_cuda=torch.cuda.is_available()
)

# 加载要解释的图像
image_path = "test5.jpg"
image = Image.from_file(image_path)

# 生成解释
explanation = ig.explain([image])

# 展示解释结果
explanation.show()
