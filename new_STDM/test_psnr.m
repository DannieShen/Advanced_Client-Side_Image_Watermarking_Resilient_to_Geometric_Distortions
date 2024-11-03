clc;
clear;
img_name = "5 cat256_rgb.jpg";
Delta = 30;
for i = 1:1
%     tic;
    img_psnr = psnr_fun_rgb2(img_name,Delta);
%     time = toc;
%     fprintf("Delta: %d, psnr: %d, time: %.2f\n",Delta,img_psnr,time);
    fprintf("Delta: %d, psnr: %d\n",Delta,img_psnr);
    Delta = Delta + 100;
end