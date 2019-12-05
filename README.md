# captcha
 基于CNN的简单验证码识别
 
 目前仅支持单一种类验证码识别
 
 参考项目： https://github.com/ice-tong/pytorch-captcha

 ## 数据集
  
 ## 文件说明
 + cnn_captcha.py 顶层文件，调用接口
 + models.py CNN模型，由卷积层，池化层，激活函数以及BatchNorm组成
 + dataset.py 数据样本封装，Dataset子类
 + model.pth 已训练好的模型
