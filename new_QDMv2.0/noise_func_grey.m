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
% ro_angle = 44;
% img_resize = 1.5;
% q = 20;
% dbw_noise = 4; %the variance of the noise

T1 = floor(T/2);
% T2 = T-T1;

img_name = "Peppers.bmp";
img = imread(img_name);
% 若是彩色图像则转为灰度图像
if length(size(img)) == 3
    img = rgb2gray(img);
end
img = double(img);
% 展示图像
% figure('Name', '原始图像');
% imshow(uint8(img));

%zernike正变换，得到Anm, Vnm
% [V,PQ]=zernike_V(img, order);
% [A]=zernike_A(img, V, PQ);
PQ = zernike_orderlist(order,1); 
[A]=zernike_A2(img, PQ);
img_size = size(img);        %图像大小

% 选取低阶zernike系数嵌入水印
% in = find(PQ(:,2)>0 & mod(PQ(:,2),4)~=0);
in=find(PQ(:,2)>5&PQ(:,1)>2&mod(PQ(:,2),4)~=0);
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
    V1 = zernike_V2(img_size, PQ, in(k));
    V2 = zernike_V2(img_size, PQ, Ain(1));
    img_else=img_else-(A(in(k))*V1+conj(A(in(k)))*V2);
end

A_enc_abs = abs(T_multiply*A/A(1));

% delta = zeros(L,1);
delta = round(Delta*rand(L,1)) - Delta/2;
embed_q = water_quantizer_e(A_enc_abs,flag,delta,Delta);
embed_q = symmetri(embed_q,PQ,flag);

quan_img = img_else;
M = length(PQ);
for i = 1:M
    V = zernike_V2(img_size, PQ, i);
    quan_img = quan_img + embed_q(i)/A_enc_abs(i)*A(i)*V;
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
    V = zernike_V2(img_size, PQ, i);
    c_img = c_img + pad_e(i)/A_enc_abs(i)*A(i)*V;
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
cipher_img = c_img;
for i = 1:outside_len
    cipher_img(outside(i)) = bitxor(c_img(outside(i)),text(i));
end

cipher = mod(round(real(cipher_img)),256);

figure('Name', '第二次加密图像');
imshow(uint8(cipher));

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
w_img = double(cipher);
pad_d = zeros(M,1);
for i = 1:M
    for j = 1:R
        pad_d(i) = pad_d(i) + DLUT(sk_a((i-1)*R+j));
    end
    V = zernike_V2(img_size, PQ, i);
    w_img = w_img + pad_d(i)*A(i)/A_enc_abs(i)*V;
end

% 内切圆外像素解密（流密码解密）
rng(key);

text = randi([0,255],outside_len,1);
decryption = w_img;
for i = 1:outside_len
    decryption(outside(i)) = bitxor(w_img(outside(i)),text(i));
end

img_w1 = uint8(mod(round(real(decryption)),256));
img_w1 = filter_img(img_w1);
figure('Name', '第二次解密图像');
imshow(img_w1);
time2 = toc;
fprintf("用户端运行时间：%.2f\n",time2);

peak_snr_wn = psnr(img_w1, uint8(img));

% ------------------------------------------------- %
% 提取水印信息
% ------------------------------------------------- %
Bm = Bm_Gen1(L,T,R,sk_a,flag);
% % [w_A] = zernike_A(img_w, V, PQ);
% [w_A] = zernike_A2(img_w,PQ);
% w_A_abs = abs(T_multiply*w_A/w_A(1));
% [arbitration] = detection_my(Bm,G,w_A_abs,b_k,Delta,delta,flag);
% %[arbitration] = detection_my1(Bm,G,w_A_abs,b_k,Delta,delta,flag);
% if arbitration == 1
%     disp("图像水印提取成功")
% else
%     disp("图像水印提取失败")
% end

% 加噪声
img_w = double(img_w1);
dbw_noise = 26;
for i = 1:10
    fprintf("dbw_noise:%d\n",dbw_noise);
    img_noise = awgn(img_w, -dbw_noise); % generate Gaussian noise
    figure;imshow(uint8(img_noise));
    [A_noise]=zernike_A2(img_noise,PQ);
    A_noise_abs = abs(T_multiply*A_noise/A_noise(1));
    [arbitration_noise] = detection_my(Bm,G,A_noise_abs,b_k,Delta,delta,flag);
    if arbitration_noise == 1
        disp("图像加噪声后水印提取成功")
    else
        disp("图像加噪声后水印提取失败")
    end
    dbw_noise = dbw_noise + 2;
end