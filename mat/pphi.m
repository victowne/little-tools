A=load('./xy0050.dat');
p=load('./Profile0.dat');
r=p(:,1);                                       %����Χ
r=r';
eq=p(:,16);                                      %phi0
a=A(:,3);
b=A(:,4);                                       %ѡ��ͼ��ֵ  
a=reshape(a,500,4);              
b=reshape(b,500,4);                             %�������
a=a+1i*b;                     
n=0:3;                                          %ģ��
k=2*n;                                          %kn
k=k';
theta=0:0.01:2.005*pi;                              %����Χ
b=2*exp(1i*k*theta);
b(1,:)=b(1,:)/2;
PHI=a*b;                      
PHI=real(PHI);
PHI0=eq';                                        %ƽ����
PHI0=PHI0';
PHI0=repmat(PHI0,1,630);      
% PHI=PHI+PHI0;
%r=r(:,1:300);PHI=PHI(1:300,:);             %��ȡ��Χ
[tt, rr] = meshgrid(theta, r);
[x, y] = pol2cart(tt, rr);
contourf(y,x,PHI,50,'linecolor','none')
% contourf(x,y,PHI,50)