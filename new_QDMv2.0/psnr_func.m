function [peak_snr_wn] = psnr_func(img_name,Delta)
order = 31;
T_multiply = 10000;
sigma_E = 10^5;
T = 3000;
R = 3;
L = 30; % 水印长度
% Delta = 50;
% ro_angle = 44;
% img_resize = 1.5;
% q = 20;
% dbw_noise = 4; %the variance of the noise

T1 = floor(T/2);
% T2 = T-T1;

% img_name = "lena512.bmp";
img_rgb = imread(img_name);
% uint8->double
img_rgb = double(img_rgb);
img = img_rgb(:,:,1);
% 展示图像
% figure('Name', '原始图像');
% imshow(uint8(img));

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
img_length2 = 512*512;
img_length3 = 512*512;
total_length = outside_len+img_length2+img_length3;

text = randi([0,255],total_length,1);

% cipher = zeros(outside_len,1);
cipher1_img = c_img;
for i = 1:outside_len
    cipher1_img(outside(i)) = bitxor(c_img(outside(i)),text(i));
end

% 图像二通道、三通道流密码加密

cipher2_img = zeros(img_size(1),img_size(2));
img2 = img_rgb(:,:,2);
for i = 1:img_length2
    cipher2_img(i) = bitxor(img2(i),text(i+outside_len));
end

cipher3_img = zeros(img_size(1),img_size(2));
img3 = img_rgb(:,:,3);
for i = 1:img_length3
    cipher3_img(i) = bitxor(img3(i),text(i+outside_len+img_length2));
end

cipher = zeros(img_size(1),img_size(2),3);
cipher(:,:,1) = cipher1_img;
cipher(:,:,2) = cipher2_img;
cipher(:,:,3) = cipher3_img;

cipher1 = mod(round(real(cipher)),256);
% cipher1 = round(real(cipher));
% cipher1(cipher1<0) = 0;
% cipher1(cipher1>255) = 255;

figure('Name', '第二次加密图像');
imshow(uint8(cipher1));


% 设置要嵌入的水印信息
b_k = randi([0,1],L,1);
m_k = 2*b_k-1;
W =  floor(Delta/4) * (m_k);
% 生成水印查找表
WLUT = floor(Delta/(4*R)) * G * (m_k);
% 生成解密查找表
DLUT = -ELUT + WLUT;

% 图像解密
cipher1 = double(cipher1);
w_img = cipher1(:,:,1);
pad_d = zeros(M,1);
for i = 1:M
    for j = 1:R
        pad_d(i) = pad_d(i) + DLUT(sk_a((i-1)*R+j));
    end
    w_img = w_img + pad_d(i)*A(i)/A_enc_abs(i)*V(:,:,i);
end

% 内切圆外像素解密（流密码解密）
rng(key);
img_length2 = 512*512;
img_length3 = 512*512;
total_length = outside_len+img_length2+img_length3;
text = randi([0,255],total_length,1);
decryption1 = w_img;
for i = 1:outside_len
    decryption1(outside(i)) = bitxor(w_img(outside(i)),text(i));
end

% 图像二通道、三通道流密码解密
img_size = size(cipher1);
decryption2 = zeros(img_size(1),img_size(2));
img2 = cipher1(:,:,2);
for i = 1:img_length2
    decryption2(i) = bitxor(img2(i),text(i+outside_len));
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
% img_w = round(real(decryption));
% img_w(img_w<0)=0;
% img_w(img_w>255)=255;
figure('Name', '第二次解密图像');
imshow(img_w1);

% figure('Name', '水印信息');
% imshow(uint8(mod(round(real(img_w-img)),256)));

% ------------------------------------------------- %
% 提取水印信息
% ------------------------------------------------- %
img_w = double(img_w1(:,:,1));
% Bm1 = Bm_Gen(L,T,R,sk_w);
Bm = Bm_Gen1(L,T,R,sk_a,flag);
[w_A] = zernike_A(img_w, V, PQ);
w_A_abs = abs(T_multiply*w_A/w_A(1));
[arbitration] = detection_my(Bm,G,w_A_abs,b_k,Delta,delta,flag);
%[arbitration] = detection_my1(Bm,G,w_A_abs,b_k,Delta,delta,flag);
if arbitration == 1
    disp("图像水印提取成功")
else
    disp("图像水印提取失败")
end

peak_snr_wn = psnr(img_w1, uint8(img_rgb));
end

