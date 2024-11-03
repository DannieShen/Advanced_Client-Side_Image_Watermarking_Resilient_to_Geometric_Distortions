clc;
clear;
tic;
% 读取图片
img = imread('dog.jpg');
imshow(img)

% 获取图片的尺寸
[height, width, ~] = size(img);

% 设置裁剪后的大小
crop_size = 1010;

% 计算裁剪的起始点（居中裁剪）
x_start = floor((width - crop_size) / 2) + 1;
y_start = floor((height - crop_size) / 2) + 1;

% 如果图片尺寸小于1024×1024，进行相应的调整
if width < crop_size || height < crop_size
    error('Image dimensions are smaller than the required crop size of 1024x1024.');
end

% 裁剪图片为 1024×1024
cropped_img = imcrop(img, [x_start, y_start, crop_size-1, crop_size-1]);

% 显示裁剪后的图片
imshow(cropped_img);

% 保存裁剪后的图片
imwrite(cropped_img, 'output_image_1024x1024.jpg');

