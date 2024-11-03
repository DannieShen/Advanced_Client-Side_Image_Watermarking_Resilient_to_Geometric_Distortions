clc;
clear;
tic;
order = 31;
T_multiply = 1000;
sigma_E = 10^5;
T = 3000;
R = 3;
L = 30; % 水印长度
Delta = 20;
ro_angle = 0;
% img_resize = 0.5;
q = 20;
dbw_noise = 4; %the variance of the noise

T1 = floor(T/2);
T2 = T-T1;

img = imread("Lena.bmp");
% 若是彩色图像则转为灰度图像
if length(size(img)) == 3
    img = rgb2gray(img);
end
% uint8->double
img = double(img);
% 展示图像
figure;imshow(uint8(img));

%zernike正变换，得到Anm, Vnm
[V,PQ]=zernike_V(img, order);
[A]=zernike_A(img, V, PQ);
img_size = size(img);        %图像大小

% 选取低阶zernike系数嵌入水印
in = find(PQ(:,1)<25 & PQ(:,2)>0 & mod(PQ(:,2),4)~=0);
if length(in) < L
    fprintf("wrong:length(in) < water_length\n");
end

% 标记高阶和低阶系数的下标
flag = zeros(1,length(PQ));   % flag: 1×len(PQ)
flag(in(1:L)) = 1;

% 计算除去order内像素的图像
img_else = img;
for k = 1:L
    A_enc_complex=A(in(k));
    Ain=find(abs(A)==abs(A_enc_complex));
    img_else=img_else-(A(in(k))*V(:,:,in(k))+conj(A(in(k)))*V(:,:,Ain(1)));
end

A_enc_abs = abs(T_multiply*A/A(1));

% delta = zeros(L,1);
delta = round(Delta*rand(L,1)) - Delta/2;
embed_q = water_quantizer_e(A_enc_abs,flag,delta,Delta);
embed_q = symmetri(embed_q,PQ,flag);

quan_img = img_else;
M = length(PQ);
for i = 1:M
    quan_img = quan_img + embed_q(i)/A_enc_abs(i)*A(i)*V(:,:,i);
end

% 生成加密查找表
ELUT = ELUT_Gen(sigma_E,T);
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

% 图像加密
M = length(PQ);

c_img = quan_img;
pad_e =zeros(M,1);
for i = 1:M
    for j = 1:R
        pad_e(i) = pad_e(i) + ELUT(sk_a((i-1)*R+j));
    end
    c_img = c_img + pad_e(i)/A_enc_abs(i)*A(i)*V(:,:,i);
end


xstep = 2/(img_size(1)-1);
ystep = 2/(img_size(2)-1);
[x,y] = meshgrid(-1:xstep:1,-1:ystep:1);
circle1 = sqrt(x.^2+y.^2);
inside = find(circle1<1.0001);
outside = find(circle1>=1.0001);
% total = length(inside) + length(outside);
img_outside = img(outside);
outside_len = length(img_outside);
% 内切圆外像素加密（流密码加密）
key = rng;
text = randi([0,255],outside_len,1);
% cipher = zeros(outside_len,1);
cipher = c_img;
for i = 1:outside_len
    cipher(outside(i)) = bitxor(c_img(outside(i)),text(i));
end

cipher1 = mod(round(real(cipher)),256);
figure;imshow(uint8(cipher1));


% 设置要嵌入的水印信息
b_k = randi([0,1],L,1);
m_k = 2*b_k-1;
W =  floor(Delta/4) * (m_k);
% 生成水印查找表
WLUT = floor(Delta/(4*R)) * G * (m_k);
% 生成解密查找表
DLUT = -ELUT + WLUT;

% 图像解密
w_img = cipher1;
pad_d = zeros(M,1);
for i = 1:M
    for j = 1:R
        pad_d(i) = pad_d(i) + DLUT(sk_a((i-1)*R+j));
    end
    w_img = w_img + pad_d(i)*A(i)/A_enc_abs(i)*V(:,:,i);
end

% 内切圆外像素解密（流密码解密）
rng(key);
text1 = randi([0,255],outside_len,1);
decryption = w_img;
for i = 1:outside_len
    decryption(outside(i)) = bitxor(w_img(outside(i)),text1(i));
end

img_w = mod(round(real(decryption)),256);
figure;imshow(uint8(img_w));
peak_snr_wn = psnr(uint8(img_w), uint8(img));

% ------------------------------------------------- %
% 提取水印信息
% ------------------------------------------------- %
% Bm1 = Bm_Gen(L,T,R,sk_w);
Bm = Bm_Gen1(L,T,R,sk_a,flag);
[w_A] = zernike_A(img_w, V, PQ);
w_A_abs = abs(T_multiply*w_A/w_A(1));
[arbitration] = detection_my2(Bm,G,w_A_abs,b_k,Delta,delta,flag);
%[arbitration] = detection_my1(Bm,G,w_A_abs,b_k,Delta,delta,flag);
if arbitration == 1
    disp("图像水印提取成功")
else
    disp("图像水印提取失败")
end

% % 加噪声
% img_noise = awgn(img_w, -dbw_noise); % generate Gaussian noise
% Img_noise = uint8(img_noise);
% figure;imshow(Img_noise);
% peak_snr_wn = psnr(Img_noise, uint8(img));
% [A_noise]=zernike_A(img_noise, V, PQ);
% A_noise_abs = abs(T_multiply*A_noise/A_noise(1));
% [arbitration_noise] = detection_my(Bm,G,A_noise_abs,b_k,Delta,delta,flag);
% if arbitration_noise == 1
%     disp("图像加噪声后水印提取成功")
% else
%     disp("图像加噪声后水印提取失败")
% end

img_resize = 0.1;
% 缩放
for i = 1:19
    I_resize = imresize(img_w,img_resize);
    figure;imshow(uint8(I_resize));
    [V_resize,PQ]=zernike_V(I_resize, order);
    [A_resize]=zernike_A(I_resize, V_resize, PQ);
    A_resize_abs = abs(T_multiply*A_resize/A_resize(1));
    [arbitration_resize] = detection_my2(Bm,G,A_resize_abs,b_k,Delta,delta,flag);
    if arbitration_resize == 1
        disp("图像缩放后水印提取成功")
    else
        disp("图像缩放后水印提取失败")
    end
    img_resize = img_resize + 0.1;
end


% % 旋转
% for i =1:10
%     fprintf("ro_angle:%d\n",ro_angle);
%     I_rotation = imrotate(img_w,ro_angle, 'bilinear', 'crop');
%     figure;imshow(uint8(I_rotation));
%     [V_rotation,PQ]=zernike_V(I_rotation, order);
%     [A_rotation]=zernike_A(I_rotation, V_rotation, PQ);
%     A_rotation_abs = abs(T_multiply*A_rotation/A_rotation(1));
%     [arbitration_rotation] = detection_my2(Bm,G,A_rotation_abs,b_k,Delta,delta,flag);
%     if arbitration_rotation == 1
%         disp("图像旋转后水印提取成功")
%     else
%         disp("图像旋转后水印提取失败")
%     end
%     ro_angle = ro_angle + 37;
% end

% % JEPG压缩
% imwrite(uint8(img_w),'img_w.jpg','quality',q);
% img1_w = imread('img_w.jpg');
% img1_w = double(img1_w);
% figure;imshow(uint8(img1_w));
% [V_jepg,PQ1]=zernike_V(img1_w, order);
% [A_jepg]=zernike_A(img1_w, V_jepg, PQ1);
% A_jepg_abs = abs(T_multiply*A_jepg/A_jepg(1));
% [arbitration_jepg] = detection_my(Bm,G,A_jepg_abs,b_k,Delta,delta,flag);
% if arbitration_jepg == 1
%     disp("图像JEPG压缩后水印提取成功")
% else
%     disp("图像JEPG压缩后水印提取失败")
% end
time = toc;