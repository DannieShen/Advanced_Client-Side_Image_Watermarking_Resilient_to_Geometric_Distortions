clear;
clc;
tic;
order = 31;
T_multiply = 10000;
sigma_E = 10^5;
T = 3000;
R = 3;
% M1 >= M_D > len_skm >= L
L = 30; % 水印长度
M_D = 50;  % 所有标准正交向量个数
len_skm = 40; % 要嵌入水印的标准正交向量个数
M1 = 60; % 低频zernike矩系数的个数
Delta = 50;
% ro_angle = 44;
% img_resize = 0.1;
% q = 50;
% dbw_noise = 4; %the variance of the noise

T1 = floor(T/2);
% T2 = T-T1;

img_name = "5 cat512_rgb.jpg";
img_rgb = imread(img_name);
% 将 RGB 图像转换为 YCbCr 色彩空间
img_ycbcr = rgb2ycbcr(img_rgb);
% uint8->double
img_ycbcr = double(img_ycbcr);
img = img_ycbcr(:,:,1);
% 展示图像
% figure('Name', '原始图像');
% imshow(uint8(img));

%zernike正变换，得到Anm, Vnm
PQ = zernike_orderlist(order,1); 
% [V]=zernike_V(img, order);
[A]=zernike_A2(img, PQ);
% [A]=zernike_A(img, V, PQ);
img_size = size(img);        %图像大小

% 选取低阶zernike系数嵌入水印
in = find(PQ(:,1)<31 & PQ(:,2)>0 & mod(PQ(:,2),4)~=0);
if length(in) < L
    fprintf("wrong:length(in) < water_length\n");
end

% 标记高阶和低阶系数的下标
flag = zeros(1,length(PQ));   % flag: 1×len(PQ)
flag(in(1:M1)) = 1;

% 计算除去order内像素的图像
img_else = img;
for k = 1:M1
    A_enc_complex=A(in(k));
    Ain=find(abs(A)==abs(A_enc_complex));
    V1 = zernike_V2(img_size, PQ, in(k));
    V2 = zernike_V2(img_size, PQ, Ain(1));
    img_else=img_else-(A(in(k))*V1+conj(A(in(k)))*V2);
%     img_else=img_else-(A(in(k))*V(:,:,in(k))+conj(A(in(k)))*V(:,:,Ain(1)));
end
% figure;imshow(uint8(img_else));

A_enc_abs = abs(T_multiply*A/A(1));

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

% 生成标准正交向量S
[S_l,Sym_l,SA_l,SAl_index] = S_Gen(flag,M_D,len_skm,PQ);
% 生成密钥
[sk_a,sk_w] = Sk_Gen(G,M_D,SAl_index,T1,T,L,len_skm,R);

delta = round(Delta*rand(len_skm,1)) - Delta/2;
% delta = zeros(len_skm,1);
% Quantizer_0
embed_q = water_quantizer_e(A_enc_abs,SA_l,delta,Delta);

% 图像加密
M = length(PQ);

c_img = img_else;
ppad_e = zeros(M,1);
for i = 1:M
    ppad_e(i) = A_enc_abs(i)*flag(i) + embed_q(i);
end
ppad_e = symmetri(ppad_e,PQ,flag);
pad_e = zeros(2*M_D,1);
for i = 1:M_D
    for j = 1:R
        pad_e(i) = pad_e(i) + ELUT(sk_a((i-1)*R+j));
    end
    ppad_e = ppad_e +  pad_e(i)*S_l(:,i);
end
for i = 1:M_D
    for j = 1:R
        pad_e(i+M_D) = pad_e(i+M_D) + ELUT(sk_a((i-1)*R+j));
    end
    ppad_e = ppad_e +  pad_e(i+M_D)*Sym_l(:,i);
end
for i = 1:M
%     c_img = c_img + (embed_q(i)+ppad_e(i))*A(i)/A_enc_abs(i)*V(:,:,i);
    V = zernike_V2(img_size, PQ, i);
    c_img = c_img + ppad_e(i)*A(i)/A_enc_abs(i)*V;
end
% c_img = img_construction(M,PQ,c_img,ppad_e,A,A_enc_abs);

% figure('Name', '第一次加密图像');
% c1_img = mod(round(real(c_img)),256);
% imshow(uint8(c1_img));

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
img_length2 = img_size(1)*img_size(2);
img_length3 = img_size(1)*img_size(2);
total_length = outside_len+img_length2+img_length3;

text = randi([0,255],total_length,1);

% cipher = zeros(outside_len,1);
cipher2_img = c_img;
for i = 1:outside_len
    cipher2_img(outside(i)) = bitxor(c_img(outside(i)),text(i));
end

% 图像二通道、三通道流密码加密

cipher1_img = zeros(img_size(1),img_size(2));
img1 = img_ycbcr(:,:,2);
for i = 1:img_length2
    cipher2_img(i) = bitxor(img1(i),text(i+outside_len));
end

cipher3_img = zeros(img_size(1),img_size(2));
img3 = img_ycbcr(:,:,3);
for i = 1:img_length3
    cipher3_img(i) = bitxor(img3(i),text(i+outside_len+img_length2));
end

cipher = zeros(img_size(1),img_size(2),3);
cipher(:,:,1) = cipher1_img;
cipher(:,:,2) = cipher2_img;
cipher(:,:,3) = cipher3_img;

cipher1 = uint8(mod(round(real(cipher)),256));
% cipher1 = ycbcr2rgb(cipher1);
% cipher1 = round(real(cipher));
% cipher1(cipher1<0) = 0;
% cipher1(cipher1>255) = 255;

figure('Name', '第二次加密图像');
imshow(cipher1);

time1 = toc;
fprintf("所有者端运行时间：%.2f\n",time1);


tic;
% 设置要嵌入的水印信息
b_k = randi([0,1],L,1);
m_k = 2*b_k-1;
W =  floor(Delta/4) * (m_k);
% 生成水印查找表
WLUT = floor(Delta/(4*R)) * G * (m_k);
% 生成解密查找表
DLUT = -ELUT + WLUT;

% 图像解密
% cipher1 = rgb2ycbcr(cipher1);
cipher1 = double(cipher1);
w_img = cipher1(:,:,2);
pad_d =zeros(2*M_D,1);
ppad_d = zeros(M,1);
for i = 1:M_D
    for j = 1:R
        pad_d(i) = pad_d(i) + DLUT(sk_a((i-1)*R+j));
    end
    ppad_d = ppad_d +  pad_d(i)*S_l(:,i);
end
for i = 1:M_D
    for j = 1:R
        pad_d(i+M_D) = pad_d(i+M_D) + DLUT(sk_a((i-1)*R+j));
    end
    ppad_d = ppad_d +  pad_d(i+M_D)*Sym_l(:,i);
end
for i = 1:M
    V = zernike_V2(img_size, PQ, i);
    w_img = w_img + ppad_d(i)*A(i)/A_enc_abs(i)*V;
end
% w_img = img_construction(M,PQ,w_img,ppad_e,A,A_enc_abs);

% figure('Name', '第一次解密图像');
% w1_img = mod(round(real(w_img)),256);
% imshow(uint8(w1_img));


% 内切圆外像素解密（流密码解密）
rng(key);
img_length2 = img_size(1)*img_size(2);
img_length3 = img_size(1)*img_size(2);
total_length = outside_len+img_length2+img_length3;
text = randi([0,255],total_length,1);
decryption2 = w_img;
for i = 1:outside_len
    decryption2(outside(i)) = bitxor(w_img(outside(i)),text(i));
end

% 图像二通道、三通道流密码解密
img_size = size(cipher1);
decryption1 = zeros(img_size(1),img_size(2));
img2 = cipher1(:,:,2);
for i = 1:img_length2
    decryption1(i) = bitxor(img2(i),text(i+outside_len));
end

decryption3 = zeros(img_size(1),img_size(2));
img3 = cipher1(:,:,3);
for i = 1:img_length3
    decryption3(i) = bitxor(img3(i),text(i+outside_len+img_length2));
end

decryption_img = zeros(img_size(1),img_size(2),3);
decryption_img(:,:,1) = decryption1;
decryption_img(:,:,2) = decryption2;
decryption_img(:,:,3) = decryption3;

img_w1 = uint8(mod(round(real(decryption_img)),256));
filtered_img = filter_img(img_w1(:,:,1));
img_w1(:,:,1) = filtered_img;
img_w1 = ycbcr2rgb(img_w1);
% img_w = round(real(decryption));
% img_w(img_w<0)=0;
% img_w(img_w>255)=255;
figure('Name', '第二次解密图像');
imshow(img_w1);
time2 = toc;
fprintf("用户端运行时间：%.2f\n",time2);


% figure('Name', '水印信息');
% imshow(uint8(mod(round(real(img_w-img)),256)));

% ------------------------------------------------- %
% 提取水印信息
% ------------------------------------------------- %
Bm = Bm_Gen1(T,R,sk_a,SAl_index);

% img_w1 = rgb2ycbcr(img_w1);
% 加噪声
img_w = double(img_w1);

    img_w = img_w(:,:,2);
    [w_A]=zernike_A2(img_w,PQ);
    w_A_abs = abs(T_multiply*w_A/w_A(1));
    [arbitration] = detection_my(Bm,G,SA_l,w_A_abs,b_k,Delta,delta);
    if arbitration == 1
        disp("图像水印提取成功")
    else
        disp("图像水印提取失败")
    end


% JEPG压缩
q = 100;
for i = 1:10
    fprintf("q:%d\n",q);
    imwrite(img_w1,'img_w.jpg','quality',q);
    img1_w = imread('img_w.jpg');
    figure;imshow(uint8(img1_w));
    img_jepg = double(img1_w);
    img_jepg = img_jepg(:,:,1);
    [A_jepg]=zernike_A2(img_jepg,PQ);
    A_jepg_abs = abs(T_multiply*A_jepg/A_jepg(1));
    [arbitration_jepg] = detection_my(Bm,G,SA_l,A_jepg_abs,b_k, Delta, delta);
    if arbitration_jepg == 1
        disp("图像JEPG压缩后水印提取成功")
    else
        disp("图像JEPG压缩后水印提取失败")
    end
    q = q - 10;
end




