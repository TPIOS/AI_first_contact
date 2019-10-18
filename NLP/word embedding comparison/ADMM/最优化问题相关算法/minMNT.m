function [x,minf]=minMNT(f,x0,var,eps)
%Ŀ�꺯����f;
%��ʼ�㣺x0��
%�Ա���������var��
%���ȣ�eps��
%Ŀ�꺯��ȡ��Сֵʱ�Ա�����ֵ��x��
%Ŀ�꺯������Сֵ��minf

format long;
if nargin==3
    eps=1.0e-6;
end
tol=1;
x0=transpose(x0);
syms 1;

while tol>eps
    gradf=jacobian(f,var);      %�ݶȷ���
    jacf=jacobian(gradf,var);   %�ſ˱Ⱦ���
    v=Funval(gradf,var,x0);
    tol=norm(v);
    pv=Funval(jacf,var,x0);
    p=-inv(pv)*transpose(v);    %��������
    y=x0+1*p;
    yf=Funval(f,var,y);
    [a,b]=minJT(yf,0,0.1);
    xm=minHJ(yf,a,b);           %�ƽ�ָ����һά����
    x1=x0+xm*p;
    x0=x1;
end

x=x1;
minf=Funval(f,var,x);
format short;