function [A]=zernike_A(img, V, PQ)
img = double(img);
img_size = size(img);       %图像大小			
xstep = 2/(img_size(1)-1);  %x方向上步长
ystep = 2/(img_size(2)-1);  %y方向上步长
%按照公式推导，先求R，再求V，最后求A
A=zeros(1,length(PQ));
for k=1:length(PQ)
 n=PQ(k,1);
 A(k)=((n+1)/pi*sum(sum(xstep*ystep*img.*conj(V(:,:,k)))));  %conj计算复数共扼值
end

   


