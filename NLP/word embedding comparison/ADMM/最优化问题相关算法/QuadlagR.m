function [xv,fv]=QuadlagR(H,c,A,b)
%��������H
%���ι滮һ����ϵ��������c
%��ʽԼ������A
%��ʽԼ���Ҷ�������b
%Ŀ�꺯��ȡ��Сֵʱ���Ա���ֵ��xv
%Ŀ�꺯������Сֵ��fv

invH=inv(H);
F=invH*transpose(A)*inv(A*invH*transpose(A))*A*invH-invH;
D=inv(A*invH*transpose(A))*A*invH;
xv=F*c+transpose(D)*b;
fv=transpose(xv)*H*xv/2+transpose(c)*xv;