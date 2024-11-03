clear;
clc;
tic

L = 30;
watermark = randi([0,1],L,1);
% b_k = [1,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,1,1,1,1,0,1,1,0,1,0,0,1,1];
% watermark = b_k';
% watermark=[1,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,1,1,1,1,0,1,1,0,1,0,0,1,1,1,0,1,0,0,0,0,1,0,1,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,0,1,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,1,1,1,1,0,1,1,0,1,0,0,1,1,1,0,1,0,0,0,0,1,0,1,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,0];

img_name='Peppers.bmp';
img = imread(img_name);
% 若是彩色图像则转为灰度图像
if length(size(img)) == 3
    img = rgb2gray(img);
end
img = double(img);	
figure;imshow(uint8(img));
order = 31;     %设定zernike变换的阶数
step=20;         %量化步长
T_multiply=1000;



%%
%----------------2.嵌入水印----------------
%zernike正变换，得到Anm, Vnm
[V,PQ]=zernike_V(img, order);
[A]=zernike_A(img, V, PQ);
img_size = size(img);        %图像大小	

%选择合适的zernike矩,生成zernike矩的索引
in=find(PQ(:,2)>5&PQ(:,1)>2&mod(PQ(:,2),4)~=0);
Irw=zeros(img_size);
length_watermark = length(watermark);  %水印长度
quantified_error = zeros(size(A));    %矩系数绝对值量化损失 dq
%水印的嵌入,对zernike矩的幅度值进行量化
for k=1:length_watermark
    A_enc_complex=A(in(k));
    A_enc_abs = abs(T_multiply*A_enc_complex/A(1)); %A(1)便是A00
    %选择A的整数部分进行水印嵌入，嵌入后对虚部进行恢复
    q = floor(A_enc_abs/step)*step;         %构建量化器q
    quantified_error(in(k)) = A_enc_abs - q;   %保存整数部分的量化损失
    if watermark(k)==0                  %进行量化水印嵌入
        A_embed=q+step/4;
    else
        A_embed=q+3*step/4;
    end
    Ain=find(abs(A)==abs(A_enc_complex));
    Irw=Irw+(A_embed/A_enc_abs - 1)*(A(in(k))*V(:,:,in(k))+conj(A(in(k)))*V(:,:,Ain(1)));   
end
%将高频成分和带有水印的低频成分加入，得到水印图像Iw
Iw=img+Irw;
%对Iw进行round处理，保证Iw为整数
Iw1=round(Iw);
figure;
imshow(uint8(Iw1));

peak_snr_wn = psnr(uint8(Iw1), uint8(img));
fprintf("peak_snr_wn:%.4f\n",peak_snr_wn);
