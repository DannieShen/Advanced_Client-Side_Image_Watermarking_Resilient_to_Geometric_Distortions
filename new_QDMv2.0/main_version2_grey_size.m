clc;
clear;
order = 17;
T_multiply = 1000;
sigma_E = 10^5;
T = 1000;
R = 3;
L = 30; % 水印长度
Delta = 20;
ro_angle = 0;
% img_resize = 0.5;
q = 20;
dbw_noise = 4; %the variance of the noise

T1 = floor(T/2);
T2 = T-T1;
% % -------------密钥生成----------------
% 选取低阶zernike系数嵌入水印
tic;
PQ = zernike_orderlist(order,1);

in = find(PQ(:,2)>5 & PQ(:,2)>0 & mod(PQ(:,2),4)~=0);
if length(in) < L
    fprintf("wrong:length(in) < water_length\n");
end

% 标记高阶和低阶系数的下标
flag = zeros(1,length(PQ));   % flag: 1×len(PQ)
flag(in(1:L)) = 1;

G_zero = 1;
% G
while isempty(G_zero) == 0
    %G：T×L
    G = zeros(T,L);
    for i=T1+1:T
        j=randi([1,L]);
        G(i,j)=1;
    end
    sum_G =sum(G);
    G_zero = find(sum_G<R);
end

% 生成密钥
[sk_a,sk_w] = Sk_Gen(G,PQ,flag,T,R);

% 设置要嵌入的水印信息
b_k = randi([0,1],L,1);
m_k = 2*b_k-1;
W =  floor(Delta/4) * (m_k);
% 生成水印查找表
WLUT = floor(Delta/(4*R)) * G * (m_k);
time2 = toc;
% % -------------图像加密----------------
tic;
img = imread("1 teacup512_rgb.jpg");
% 若是彩色图像则转为灰度图像
if length(size(img)) == 3
    img = rgb2gray(img);
end
% uint8->double
img = double(img);
% 展示图像
figure;imshow(uint8(img));

%zernike正变换，得到Anm, Vnm
% [V,PQ]=zernike_V(img, order);
[A]=zernike_A2(img, PQ);
img_size = size(img);        %图像大小



% 计算除去order内像素的图像
img_else = img;
for k = 1:L
    A_enc_complex=A(in(k));
    Ain=find(abs(A)==abs(A_enc_complex));
    Vi = zernike_V2(img_size, PQ,in(k));
    Vj = zernike_V2(img_size, PQ,Ain(1));
    img_else=img_else-(A(in(k))*Vi+conj(A(in(k)))*Vj);
end

A_enc_abs = abs(T_multiply*A/A(1));

% delta = zeros(L,1);
delta = round(Delta*rand(L,1)) - Delta/2;
embed_q = water_quantizer_e(A_enc_abs,flag,delta,Delta);
embed_q = symmetri(embed_q,PQ,flag);

quan_img = img_else;
M = length(PQ);
for i = 1:M
    Vi = zernike_V2(img_size, PQ,i);
    quan_img = quan_img + embed_q(i)/A_enc_abs(i)*A(i)*Vi;
end

% 生成加密查找表
ELUT = ELUT_Gen(sigma_E,T);


% 图像加密
M = length(PQ);

c_img = quan_img;
pad_e =zeros(M,1);
for i = 1:M
    for j = 1:R
        pad_e(i) = pad_e(i) + ELUT(sk_a((i-1)*R+j));
    end
    Vi = zernike_V2(img_size, PQ,i);
    c_img = c_img + pad_e(i)/A_enc_abs(i)*A(i)*Vi;
end

xstep = 2/(img_size(1)-1);
ystep = 2/(img_size(2)-1);
[x,y] = meshgrid(-1:xstep:1,-1:ystep:1);
circle1 = sqrt(x.^2+y.^2);
inside = find(circle1<1.0001);
outside = find(circle1>=1.0001);
outside_len = length(outside);
% 内切圆外像素加密（流密码加密）
key = rng;
text = randi([0,255],outside_len,1);
% cipher = zeros(outside_len,1);
cipher = c_img;
for i = 1:outside_len
    cipher(outside(i)) = bitxor(cipher(outside(i)),text(i));
end
% cipher_memory = (round(real(cipher)));
cipher1 = uint8(mod(round(real(cipher)),256));
% cipher1 = round(real(cipher));
% aa=int16((round(real(cipher))-double(cipher1))/256);
% figure;imshow(cipher1);
time1 = toc;

% ------------------------------------------------- %
% 图像解密
% ------------------------------------------------- %
tic;

% 生成解密查找表
DLUT = -ELUT + WLUT;

% 图像解密
% w_img = cipher_memory;
w_img = double(cipher1);
img_size = size(w_img);
% w_img = double(cipher1) + 256 * double(aa);
pad_d = zeros(M,1);
PQ = zernike_orderlist(order,1);
for i = 1:M
    for j = 1:R
        pad_d(i) = pad_d(i) + DLUT(sk_a((i-1)*R+j));
    end
    Vi = zernike_V2(img_size, PQ,i);
    w_img = w_img + pad_d(i)*A(i)/A_enc_abs(i)*Vi;
end

% 内切圆外像素解密（流密码解密）
rng(key);
text2 = randi([0,255],outside_len,1);
decryption = w_img;
for i = 1:outside_len
    decryption(outside(i)) = bitxor(decryption(outside(i)),text2(i));
end

Img_w = uint8(mod(round(real(decryption)),256));
figure;imshow(Img_w);

% 计算两张图像的差异
diff_img = imabsdiff(Img_w, uint8(img));  % 计算绝对差值
    
% 显示差异图像
figure;
imshow(diff_img);
title('两张图像的差异');

% 滤波
Img_w = filter_img(Img_w);
% filtered_image = Img_w;


figure;imshow(Img_w);
% peak_snr_wn = psnr(Img_w, uint8(img));
peak_snr_wn = psnr(Img_w, uint8(img));

time3 = toc;

% ------------------------------------------------- %
% 提取水印信息
% ------------------------------------------------- %
tic;
% Bm1 = Bm_Gen(L,T,R,sk_w);
Bm = Bm_Gen1(L,T,R,sk_a,flag);
% img_w = double(Img_w);
img_w = double(Img_w);
[w_A] = zernike_A2(img_w,PQ);
w_A_abs = abs(T_multiply*w_A/w_A(1));
[arbitration] = detection_my2(Bm,G,w_A_abs,b_k,Delta,delta,flag);
%[arbitration] = detection_my1(Bm,G,w_A_abs,b_k,Delta,delta,flag);
if arbitration == 1
    disp("图像水印提取成功")
else
    disp("图像水印提取失败")
end
time4 = toc;

fprintf("time1:%.4f\n",time1);
fprintf("time2:%.4f\n",time2);
fprintf("time3:%.4f\n",time3);
fprintf("time4:%.4f\n",time4);

u = zeros(M,1);
for i = 1:M
    u(i) = A(i)/A_enc_abs(i);
end