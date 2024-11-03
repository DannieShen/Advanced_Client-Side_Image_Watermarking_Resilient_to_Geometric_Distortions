% 读取原始图像
img = imread('1 teacup1279.jpg'); % 替换为你的图像文件名

% 缩小图像尺寸至 512x512
resized_img = imresize(img, [4096 4096]);

% 显示缩小后的图像
imshow(resized_img);

% 若是彩色图像则转为灰度图像
% if length(size(img)) == 3
%     img = rgb2gray(img);
% end

% 可选：保存缩小后的图像
imwrite(resized_img, '1 teacup4096_rgb.jpg'); % 替换为你想保存的文件名


% % 读取图像
% % img = imread('image.jpg');
% 
% % 获取图像的尺寸
% [height, width, ~] = size(img);
% crop_width = 1279;
% 
% % 计算裁剪区域的起始x坐标和y坐标
% x_start = round((width - crop_width) / 2) + 1;  % 水平居中裁剪
% y_start = round((height - crop_width) / 2) + 1; % 垂直居中裁剪
% 
% % 裁剪图像
% cropped_img = imcrop(img, [x_start y_start crop_width-1 crop_width-1]);
% 
% % 显示裁剪后的图像
% imshow(cropped_img);
% imwrite(cropped_img, '10 leaf1279.jpg');
