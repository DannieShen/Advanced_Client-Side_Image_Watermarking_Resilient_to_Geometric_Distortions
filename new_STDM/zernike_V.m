function [V,PQ]=zernike_V(img, Order)
PQ = zernike_orderlist(Order,1);    %求出阶数n以及满足对应关系的m
img = double(img);
img_size = size(img);       %图像大小			
xstep = 2/(img_size(1)-1);  %x方向上步长
ystep = 2/(img_size(2)-1);  %y方向上步长
[x,y] = meshgrid(-1:xstep:1,-1:ystep:1);%以图像中心为原点，每个坐标到图像中心的距离，（规划到单位圆以内）
circle1 = sqrt(x.^2 + y.^2);      %每个坐标到图像中心的距离，（规划到单位圆以内）
inside = find(circle1<=1.0001);    %inside是一个列向量，代表该图像中位于单位圆内的点
mask = zeros(img_size);      %创建全零的mask矩阵，用来存放该图像中位于单位圆内的点
mask(inside) = ones(size(inside));  %单位圆内的坐标标记为1，否则标记为0
Z=zeros(img_size);
Z(inside)=x(inside)+1i*y(inside);
p=abs(Z);
theta=angle(Z);

V=zeros(img_size(1),img_size(2),length(PQ));
%按照公式推导，先求R，再求V，最后求A
for k=1:length(PQ)
    n=PQ(k,1);
    m=PQ(k,2);
    a=(n-abs(m))/2;b=(n+abs(m))/2;
    R=zeros(img_size);
    for s=0:a      
        num=power(-1,s).*factorial(n-s)*(p.^(n-2*s));
        den=factorial(s)*factorial(b-s)*factorial(a-s);
        R=R+(num/den);
    end
    phase=m*theta;
    R(mask==0)=0;
    phase(mask==0)=0;
    V(:,:,k)=(R.*exp(1i*phase));     
end



