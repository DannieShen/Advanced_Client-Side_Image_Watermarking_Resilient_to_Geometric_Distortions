function [result_img] = filter_img(img)

% 获取图像的尺寸
[rows, columns] = size(img);

% 创建一个副本来存储结果
result_img = img;

% 遍历每个像素点，从第2列开始，因为第1列没有左侧像素点
for row = 1:rows
    for col = 2:columns
        % 计算当前像素与左侧像素的差值
        diff = double(result_img(row, col)) - double(result_img(row, col - 1));
        
        % 根据差值调整当前像素值
        if diff > 240
            result_img(row, col) = 0;
        elseif diff < -240
            result_img(row, col) = 255;
        end
    end
end

end

