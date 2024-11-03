clc;
clear;
img_name = "5 cat256_rgb.jpg";
Delta = 50;
for i = 1:1
%     tic;
    img_psnr = psnr_func2(img_name,Delta);
%     time = toc;
%     fprintf("Delta: %d, psnr: %d, time: %d\n",Delta,img_psnr,time);
    fprintf("Delta: %d, psnr: %d\n",Delta,img_psnr);
    Delta = Delta + 150;
end