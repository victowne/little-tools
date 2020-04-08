% ����3D���п����λ��ͼ�� (V 0.1 by Jiale Chan for Y. H. Huang)
% Dee Formula
% ��������
    rzero = 2.0;
    rmax = 0.75;
    eshape = 2.0;
    xshape = 0.5;
% ������뻷���
    theta = linspace(0, 2*pi, 50);
    phi = linspace(0, 1.3*pi, 50);
% �����ݾ���
    [phi_g, theta_g ] = meshgrid(phi, theta);
% �ο�TOQ��ҳ���ļ򵥹�ʽ���ɴ������ݣ�https://fusion.gat.com/THEORY/toq/geometry.html
    z = eshape*rmax*sin(theta_g);
    R = rzero + rmax*( cos(theta_g)-xshape*sin(theta_g).^2 ) ;
    x = R.*cos(phi_g);
    y = R.*sin(phi_g);
% ����3D����
    surf(x,y,z); shading interp; %���� surf(x,y,z, 'EdgeColor','none');
%     hold on;
%��㻭��һ����
%      R_line = rzero + rmax*( cos(theta)-xshape*sin(theta).^2 ) ;
%      x_line = R_line.*cos(phi*1.5);
%      y_line = R_line.*sin(phi*1.5);
%      z_line = eshape*rmax*sin(theta);
%      plot3(x_line, y_line, z_line, 'k'); 
 
% �趨ͼ��һЩ��������
%     colorbar; %����ɫ��
    xlabel('x');   ylabel('y');   zlabel('z');%�����������
    axis equal; %�������������Ϊ����ȡ�
    view(2,40); %�ı��ӽ�