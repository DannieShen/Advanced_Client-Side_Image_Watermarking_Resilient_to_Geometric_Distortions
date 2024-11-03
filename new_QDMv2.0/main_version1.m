% clc;
% clear;
% 
% sigma_E = 10^5;
% T = 1000;
% L = 30;
% R = 3;
% Delta = 20;
% order = 31;
% 
% T1 = floor(T/2);
% G_zero = 1;
% % G
% while isempty(G_zero) == 0
%     %G：T×L
%     G = zeros(T,L);
%     for i=T1+1:T
%         j=randi([1,L]);
%         G(i,j)=1;
%     end
%     sum_G =sum(G);
%     G_zero = find(sum_G<R);
% end
% 
% % 生成加密查找表
% ELUT = ELUT_Gen(sigma_E,T);
% % 设置要嵌入的水印信息
% % watermark=[1,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,1,1,1,1,0,1,1,0,1,0,0,1,1,1,0,1,0,0,0,0,1,0,1,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,0,1,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,1,1,1,1,0,1,1,0,1,0,0,1,1,1,0,1,0,0,0,0,1,0,1,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,0];
% watermark = randi([0,1],L,1);
% m_k = 2*watermark-1;
% W =  floor(Delta/4) * (m_k);
% % 生成水印查找表
% WLUT = floor(Delta/(4*R)) * G * (m_k);
% % 生成解密查找表
% DLUT = -ELUT + WLUT;
% 
% % -------------------------------------------------------------------------
% 
% img_name = "lena512.bmp";
% img = imread(img_name);
% % 若是彩色图像则转为灰度图像
% if length(size(img)) == 3
%     img = rgb2gray(img);
% end
% img = double(img);
% % 展示图像
% figure('Name', '原始图像');
% imshow(uint8(img));
% 
% %zernike正变换，得到Anm, Vnm
% [V,PQ]=zernike_V(img, order);
% [A]=zernike_A(img, V, PQ);
% img_size = size(img);        %图像大小
% 
% %选择合适的zernike矩,生成zernike矩的索引
% in=find(PQ(:,2)>5&PQ(:,1)>2&mod(PQ(:,2),4)~=0);
% % 标记高阶和低阶系数的下标
% flag = zeros(1,length(PQ));   % flag: 1×len(PQ)
% flag(in(1:L)) = 1;
% 
% % 生成密钥
% [sk_a,sk_w] = Sk_Gen(G,PQ,flag,T,R);
% 
% % 计算除去order内像素的图像
% img_else = img;
% for k = 1:L
%     A_enc_complex=A(in(k));
%     Ain=find(abs(A)==abs(A_enc_complex));
%     img_else=img_else-(A(in(k))*V(:,:,in(k))+conj(A(in(k)))*V(:,:,Ain(1)));
% end
% 
% A_enc_abs = abs(T_multiply*A/A(1));
% 
% % delta = zeros(L,1);
% delta = round(Delta*rand(L,1)) - Delta/2;
% embed_q = water_quantizer_e(A_enc_abs,flag,delta,Delta);
% embed_q = symmetri(embed_q,PQ,flag);











% -------------------------------------------------------------------------

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

img = imread("Peppers.bmp");
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
    img_else=img_else-(A(in(k))*V(:,:,in(k))+conj(A(in(k)))*V(:,:,Ain(1)));
end

% -------------------------------------------------------------------------
b_k = randi([0,1],L,1);

A_enc_abs = abs(T_multiply*A/A(1));
Irw = img_else;
for k = 1:L
    Ain=find(abs(A)==abs(A(in(k))));
    q = floor(A_enc_abs(in(k))/Delta)*Delta;
    if b_k(k)==0
        A_embed=q+Delta/4;
    else
        A_embed=q+3*Delta/4;
    end
    Irw=Irw+(A_embed/A_enc_abs(in(k)))*(A(in(k))*V(:,:,in(k))+conj(A(in(k)))*V(:,:,Ain(1)));
end
[Aw_dec]=zernike_A(Irw, V, PQ);
Aw_dec_abs = abs(T_multiply*Aw_dec/Aw_dec(1));

img_w = mod(round(real(Irw)),256);
figure;imshow(uint8(img_w));
peak_snr_wn = psnr(uint8(img_w), uint8(img));
fprintf("peak_snr_wn:%.4f\n",peak_snr_wn);

% Iw1 = double(img_w);
[Aw_dec]=zernike_A(img_w, V, PQ);
watermark_extracted = zeros(1,L);
for k=1:L
    Aw_dec_complex=Aw_dec(in(k));
    Aw_dec_abs = abs(T_multiply*Aw_dec_complex/Aw_dec(1));
    q = floor(Aw_dec_abs/Delta)*Delta;
    delta = Aw_dec_abs - q;
    if delta >= Delta/2
        watermark_extracted(k) = 1; 
    elseif delta < Delta/2
        watermark_extracted(k) = 0;
    end  
end

%恢复出来的水印与原始水印的差距

fprintf("watermark_extracted:");
fprintf("%d",watermark_extracted);
fprintf("\n");
fprintf("b_k  :");
fprintf("%d",b_k);
fprintf("\n");
sub_w = sum(abs(watermark_extracted - b_k'));
fprintf('恢复出来的水印与原始水印的差距:\t\t%d\n',sub_w);

% -------------------------------------------------------------------------

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
% b_k = randi([0,1],L,1);
% b_k = [1,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,1,1,1,1,0,1,1,0,1,0,0,1,1];
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
fprintf("peak_snr_wn:%.4f\n",peak_snr_wn);