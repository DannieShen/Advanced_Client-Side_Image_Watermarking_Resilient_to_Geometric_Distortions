function [A]=zernike_A(img, V, PQ)
img = double(img);
img_size = size(img);       %ͼ���С			
xstep = 2/(img_size(1)-1);  %x�����ϲ���
ystep = 2/(img_size(2)-1);  %y�����ϲ���
%���չ�ʽ�Ƶ�������R������V�������A
A=zeros(1,length(PQ));
for k=1:length(PQ)
 n=PQ(k,1);
 A(k)=((n+1)/pi*sum(sum(xstep*ystep*img.*conj(V(:,:,k)))));  %conj���㸴������ֵ
end

   


