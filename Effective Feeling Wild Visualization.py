import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
import torch
from torchvision import transforms
import numpy as np
import torchvision
from PIL import Image
import cv2
import os
from models.experimental import attempt_load
#这里随机拿100张测试集图像，放到一个文件夹中，img_dir是文件夹路径
#可以放测试集的图片，图片越多，速读越慢
img_dir = "e:/yolov5/yolov5-6.1/data/images/vs" #修改成自己的
images=os.listdir(img_dir)
#自己训练好的best.pt模型，修改成自己的
# model=attempt_load(weights=r'e:\yolov5\yolov5-6.1\best.pt')
model=attempt_load(weights=r'e:\yolov5\yolov5-6.1\runs\train\exp146\weights\best.pt',
                            inplace=True,
                            fuse=False)
# model = deeplabv3_resnet50(pretrained=True, progress=False)
model = model.eval()
#定义输入图像的长宽，这里需要保证每张图像都要相同，可以修改
input_H, input_W =320, 320
#生成一个和输入图像大小相同的0矩阵，用于更新梯度
heatmap = np.zeros([input_H, input_W])
#打印一下模型，选择其中的一个层
print(model)

#-----最重要的的地方-------#
#根据网络输出结构的名称选择可是的层数，此时选择的是骨干的SPFF上一层
layer = model.model[6]
print(layer)


def farward_hook(module, data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input)
    
#为了简单，这里直接一张一张图来算，遍历文件夹中所有图像  
for img in images:
    read_img = os.path.join(img_dir,img)
    image = Image.open(read_img)
    
    #图像预处理，将图像缩放到固定分辨率，并进行标准化
    image = image.resize((input_H, input_W))
    image = np.float32(image) / 255
    input_tensor = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])(image)
    
    #添加batch维度
    input_tensor = input_tensor.unsqueeze(0)
    
    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()
        
    #输入张量需要计算梯度
    input_tensor.requires_grad = True
    fmap_block = list()
    input_block = list()
    
    #对指定层获取特征图
    layer.register_forward_hook(farward_hook)
    
    #进行一次正向传播
    output = model(input_tensor)
    
    #特征图的channel维度算均值且去掉batch维度，得到二维张量
    feature_map = fmap_block[0].mean(dim=1,keepdim=False).squeeze()
    
    #对二维张量中心点（标量）进行backward
    feature_map[(feature_map.shape[0]//2-1)][(feature_map.shape[1]//2-1)].backward(retain_graph=True)

    #对输入层的梯度求绝对值
    grad = torch.abs(input_tensor.grad)
    
    #梯度的channel维度算均值且去掉batch维度，得到二维张量，张量大小为输入图像大小
    grad = grad.mean(dim=1,keepdim=False).squeeze()
    
    #累加所有图像的梯度，由于后面要进行归一化，这里可以不算均值
    heatmap = heatmap + grad.cpu().numpy()
    
    
cam = heatmap

#对累加的梯度进行归一化
cam = cam / cam.max()

#可视化，蓝色值小，红色值大
cam = cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_JET)
cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
cam_image=Image.fromarray(cam)

#输入自己要保存的路径，修改
cam_image.save(f'yolov5s.png')
print("有效感受野可视化图片已经保存")