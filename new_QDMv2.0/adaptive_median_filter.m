function filtered_image = adaptive_median_filter(image, max_window_size)
    % 确保输入图像是uint8类型
    if ~isa(image, 'uint8')
        error('输入图像必须为uint8类型');
    end

    % 获取图像大小
    [rows, cols] = size(image);
    filtered_image = image;  % 复制图像作为输出

    % 遍历图像中的每个像素
    for i = 1:rows
        for j = 1:cols
            window_size = 3;  % 初始窗口大小
            while window_size <= max_window_size
                % 获取当前窗口的边界
                half_size = floor(window_size / 2);
                row_min = max(i - half_size, 1);
                row_max = min(i + half_size, rows);
                col_min = max(j - half_size, 1);
                col_max = min(j + half_size, cols);
                
                % 提取窗口中的像素
                window = image(row_min:row_max, col_min:col_max);
                
                % 计算窗口的中值、最小值和最大值
                window_median = median(window(:));
                window_min = min(window(:));
                window_max = max(window(:));
                
                % 自适应滤波判断
                if window_min < window_median && window_median < window_max
                    if window_min < image(i, j) && image(i, j) < window_max
                        filtered_image(i, j) = image(i, j);
                    else
                        filtered_image(i, j) = window_median;
                    end
                    break;
                else
                    window_size = window_size + 2;  % 增加窗口大小
                end
                
                % 如果超过最大窗口大小，直接使用中值
                if window_size > max_window_size
                    filtered_image(i, j) = window_median;
                end
            end
        end
    end
end

