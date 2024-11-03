clc;
clear;
img_name = "goldhill.bmp";
Delta = 26;
T_multiply = 1000;
avg_psnr = 0;
for i = 1:5
    img_psnr = func_version2_grey_psnr(img_name,T_multiply,Delta);
    fprintf("Delta: %d, psnr: %.1f\n",Delta,img_psnr);
%     Delta = Delta + 2;
    avg_psnr = avg_psnr + img_psnr;
end
avg_psnr = avg_psnr/5;
fprintf("Delta: %d, avg_psnr: %.1f\n",Delta,avg_psnr);