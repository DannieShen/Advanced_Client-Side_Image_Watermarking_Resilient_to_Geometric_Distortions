function [restored_img] = padding(orign_height,orign_width,center_height,center_width,cropped_img)
% 假设原图像的尺寸
rows = orign_height;   % 原图像的行数
columns = orign_width; % 原图像的列数

% 裁剪图像的尺寸
img_size = size(cropped_img);
x1 = img_size(2);  % 裁剪图像的宽度
y1 = img_size(1);  % 裁剪图像的高度

% 裁剪区域的中心位置
c1 = center_height;  % 裁剪图像的中心列坐标
c2 = center_width;  % 裁剪图像的中心行坐标

% 创建一个与原图像相同大小的全零矩阵
restored_img = zeros(rows, columns, 'like', uint8(0));

% 计算裁剪图像在原图像中的放置位置
x_start = max(1, c1 - floor(x1 / 2));
y_start = max(1, c2 - floor(y1 / 2));

% 确保裁剪图像的放置位置在原图像中有效
x_end = min(columns, x_start + x1 - 1);
y_end = min(rows, y_start + y1 - 1);

% 计算裁剪图像的实际宽度和高度（考虑边界问题）
crop_width = x_end - x_start + 1;
crop_height = y_end - y_start + 1;

% 将裁剪后的图像数据放入全零矩阵中
restored_img(y_start:y_start+crop_height-1, x_start:x_start+crop_width-1) = cropped_img;

% 显示恢复后的图像
figure;
imshow(restored_img);
title('恢复后的图像');

% 保存恢复后的图像
imwrite(restored_img, 'restored_image.png');



end

