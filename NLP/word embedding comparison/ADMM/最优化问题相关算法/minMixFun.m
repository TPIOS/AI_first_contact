function [x,minf]=minMixFun(f,g,h,x0,r0,c,var,eps)
%Ŀ�꺯����f
%����ʽԼ����g
%��ʽԼ����h
%��ʼ�㣺x0
%�����ӣ�r0
%��Сϵ����c
%�Ա���������var
%���ȣ�eps
%Ŀ�꺯��ȡ��Сֵʱ���Ա�����ֵ��x
%Ŀ�꺯������Сֵ��minf

gx0=Funval(g,var,x0);
if gx0>=0
    ;
else
    disp('��ʼ��������㲻��ʽԼ����');          %��ʼ����
    x=NaN;
    minf=NaN;
    return;
end

if c>=1||c<0
    disp('��Сϵ���������0��С��1��');         %��С���ϵ��
    x=NaN;
    minf=NaN;
    return;
end

if nargin==7
    eps=1.0e-6;
end

FE=0;
for i=1:length(g)
    FE=FE+1/g(i);
end
FH=transpose(h)*h;

x1=transpose(x0);
x2=inf;

while 1
    FF=r0*FE+FH/sqrt(r0);                   %��������Ŀ�꺯��
    SumF=f+FF;
    [x2,minf]=minNT(SumF,transpose(x1),var);    %��ţ�ٷ������Լ���滮
    
    if norm(x2-x1)<=eps                         %�����ж�
        x=x2;
        break;
    else
        r0=c*r0;
        x1=x2;
    end
end
min=Funval(f,var,x);