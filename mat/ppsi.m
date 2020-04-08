Lmax=3;               
Imax=499;
m=2;
% % % % % % % % % % % Plot Psi % % % % % % % % % % % % % % 
A=load('xy2590.dat');
p=load('./Profile0.dat');
r=p(:,1);                                       %����Χ
r=r';
eq=p(:,4);                                       %psi0
a=A(:,5);
b=A(:,6);                                       %ѡ��ͼ��ֵ  
a=reshape(a,Imax+1,Lmax+1);               
b=reshape(b,Imax+1,Lmax+1);                             %�������
a=a+1i*b;                     
k=m*(0:Lmax);                    %ģ��
% k=[0,2,0,0,0,0,0];
k=k';
theta=0:0.01:2.005*pi;                              %����Χ
b=2*exp(1i*k*theta);
b(1,:)=b(1,:)/2;
PSI=a*b;                      
PSI=real(PSI);
PSI0=0.8*eq'+0.25*r.*r./4;                            %ƽ����
PSI0=PSI0';
PSI0=repmat(PSI0,1,630);      
PSI=PSI+PSI0;
% r=r(:,1:270);PSI=PSI(1:270,:);             %��ȡ��Χ
[tt, rr] = meshgrid(theta, r);
[x, y] = pol2cart(tt, rr);
contourf(y,x,PSI,50,'linecolor','none')
% contourf(tt,rr,PSI,50,'linecolor','none')