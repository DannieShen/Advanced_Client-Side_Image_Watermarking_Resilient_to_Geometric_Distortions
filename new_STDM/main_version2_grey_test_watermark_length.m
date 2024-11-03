clc;
clear;
img_name = "05.boat.bmp";
L = 10;
len_skm = 20;
M_D = 30;
M1 = 40;
for i = 1:10
%     tic;
    img_psnr = func_version2_grey_watermark_length(img_name,L,len_skm,M_D,M1);
%     time = toc;
%     fprintf("Delta: %d, psnr: %.4d, time: %d\n",Delta,img_psnr,time);
    fprintf("watermark_length: %d, psnr: %.4f\n",L,img_psnr);
    len_skm = len_skm + 10;
    M_D = M_D + 10;
    L = L + 10;
    M1 = M1 + 10;
end