function [x,minf]=minGeneralPF(f,x0,h,c1,p,var,eps)
%�ú��������һ���ʽ����Ч��=��=
%��ʼ�㣺x0;
%��ʽԼ��������h
%�������ĳ�ʼ������c1
%�������ı���ϵ����p
%�Ա���������var
%���ȣ�eps
%Ŀ�꺯��ȥ��Сֵʱ���Ա�����ֵ��x
%Ŀ�꺯������Сֵ��minf

format long
if nargin==6
    eps=1.0e-4;
end
k=0;
FE=0;
for i=1:length(h)
    FE=FE+(h(i))^2;         %���췣����
end
x1=transpose(x0);
x2=inf;

while 1
    M=c1*p;
    FF=M*FE;
    SumF=f+FF;
    [x2,minf]=minNT(SumF,transpose(x1),var);    %��ţ�ٷ������Լ���滮
    if norm(x2-x1)<=eps                         %�����ж�
        x=x2;
        break;
    else
        c1=M;
        x1=x2;
    end
end
minf=Funval(f,var,x);
format short;