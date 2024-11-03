% 读取彩色图像
img_color = imread('10 plant512_rgb.jpg');

% 将彩色图像转换为灰度图像
img_gray = rgb2gray(img_color);

% 显示灰度图像
imshow(img_gray);
title('灰度图像');

% 将灰度图像保存为文件
imwrite(img_gray, '10 plant512_gray.jpg');
