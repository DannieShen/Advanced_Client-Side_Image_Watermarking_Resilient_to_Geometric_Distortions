function [cropped_img] = clip(img,crop_width, crop_height, center_x, center_y)
% 获取原图像的尺寸
[rows, columns] = size(img);

% 要裁剪图像的尺寸
% crop_width = 256;  % 裁剪图像的宽度
% crop_height = 256; % 裁剪图像的高度

% 裁剪区域的中心位置
% center_x = 300;  % 裁剪图像的中心列坐标
% center_y = 300;  % 裁剪图像的中心行坐标

% 计算裁剪区域的左上角坐标
x_start = center_x - floor(crop_width / 2);
y_start = center_y - floor(crop_height / 2);

% 确保裁剪区域不超出原图像边界
x_start = max(1, x_start);
y_start = max(1, y_start);
x_end = min(columns, x_start + crop_width - 1);
y_end = min(rows, y_start + crop_height - 1);

% 计算裁剪图像的实际宽度和高度
actual_width = x_end - x_start + 1;
actual_height = y_end - y_start + 1;

% 进行裁剪
cropped_img = img(y_start:y_end, x_start:x_end);

% 显示裁剪后的图像
figure;
imshow(cropped_img);
title('裁剪后的图像');

% 保存裁剪后的图像
imwrite(cropped_img, 'cropped_image.png');


end

