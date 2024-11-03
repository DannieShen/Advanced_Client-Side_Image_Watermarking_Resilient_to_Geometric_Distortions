function [A]=zernike_A2(img, PQ)
img = double(img);
img_size = size(img);       %ͼ���С			
xstep = 2/(img_size(1)-1);  %x�����ϲ���
ystep = 2/(img_size(2)-1);  %y�����ϲ���
[x,y] = meshgrid(-1:xstep:1,-1:ystep:1);%��ͼ������Ϊԭ�㣬ÿ�����굽ͼ�����ĵľ��룬���滮����λԲ���ڣ�
circle1 = sqrt(x.^2 + y.^2);      %ÿ�����굽ͼ�����ĵľ��룬���滮����λԲ���ڣ�
inside = find(circle1<=1.0001);    %inside��һ���������������ͼ����λ�ڵ�λԲ�ڵĵ�
mask = zeros(img_size);      %����ȫ���mask����������Ÿ�ͼ����λ�ڵ�λԲ�ڵĵ�
mask(inside) = ones(size(inside));  %��λԲ�ڵ�������Ϊ1��������Ϊ0
Z=zeros(img_size);
Z(inside)=x(inside)+1i*y(inside);
p=abs(Z);
theta=angle(Z);


A=zeros(length(PQ),1);
%���չ�ʽ�Ƶ�������R������V�������A
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
    V=(R.*exp(1i*phase));    
    A(k)=((n+1)/pi*sum(sum(xstep*ystep*img.*conj(V))));  %conj���㸴������ֵ
end

   


