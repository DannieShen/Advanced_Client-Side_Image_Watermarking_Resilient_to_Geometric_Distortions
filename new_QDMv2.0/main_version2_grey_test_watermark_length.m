clc;
clear;
img_name = "05.boat.bmp";
Delta = 18;
L = 10;
for i = 1:3
%     tic;
    img_psnr = func_version2_grey_watermark_length(img_name,L);
%     time = toc;
%     fprintf("Delta: %d, psnr: %.4d, time: %d\n",Delta,img_psnr,time);
    fprintf("watermark_length: %d, psnr: %.4f\n",L,img_psnr);
    L = L + 10;
end