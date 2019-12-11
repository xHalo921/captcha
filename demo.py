import cnn_captcha as captcha
import matplotlib.pyplot as plot # plt 用于显示图片
from PIL import Image

if __name__ == '__main__':
    img_path = './testdata/test1.jpg'
    print(captcha.captchaByPath(img_path))
    img = plot.imread(img_path)
    plot.imshow(img)
    plot.axis('off')
    plot.show()