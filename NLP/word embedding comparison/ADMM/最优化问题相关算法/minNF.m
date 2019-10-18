function [x,minf]=minNF(f,x0,g,u,v,var,eps)
%Ŀ�꺯����f
%��ʼ�㣺x0
%Լ��������g
%�����ӣ�u
%��Сϵ����v
%�Ա���������var
%���ȣ�eps
%Ŀ�꺯��ȡ��Сֵʱ�Ա�����ֵ��x
%Ŀ�꺯������Сֵ��minf

format long;
if nargin==6;
    eps=1.0e-4;
end
k=0;
FE=0;
for i=1:length(g)
    FE=FE+1/g(i);               %���췣����
end
x1=transpose(x0);
x2=inf;

while 1
    FF=u*FE;
    SumF=f+FF;
    [x2,minf]=minNT(SumF,transpose(x1),var);    %��ţ�ٷ������Լ���滮
    Bx=Funval(FE,var,x2);
    if u*Bx<eps
        if norm(x2-x1)<=eps
            x=x2;
            break;
        else
            u=v*u;                              %��������
            x1=x2;
        end
    else
        if norm(x2-x1)<=eps
            x=x2;
            break;
        else
            u=v*u;
            x1=x2;
        end
    end
end
minf=Funval(f,var,x);
format short;