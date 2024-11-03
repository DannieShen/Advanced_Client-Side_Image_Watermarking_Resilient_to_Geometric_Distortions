function [cropped_image] = cropimg1(I_rotation,img_size)
    % 获取图像的尺寸
    [height, width] = size(I_rotation);
    
    % 定义裁剪区域的尺寸
    M = img_size(1); % 裁剪区域的高度
    N = img_size(2); % 裁剪区域的宽度
    
%     % 计算图像中心的坐标
%     center_x = round(width / 2);
%     center_y = round(height / 2);
%     
%     % 计算裁剪区域的边界
%     x_start = max(1, center_x - floor(N / 2));
%     x_end = min(width, center_x + floor(N / 2) - 1);
%     y_start = max(1, center_y - floor(M / 2));
%     y_end = min(height, center_y + floor(M / 2) - 1);
%     
    % 裁剪图像
    cropped_image = I_rotation(-2/M:2/M, -2/N:2/N);

end

