function [x,minf]=minJSMixFun(f,g,h,x0,r0,c,var,eps)
%Ŀ�꺯����f
%����ʽԼ����g
%��ʽԼ����h
%��ʼ�㣺x0
%�����ӣ�r0
%��Сϵ����c
%�Ա���������var
%���ȣ�eps
%Ŀ�꺯��ȡ��Сֵʱ���Ա���ʱ���Ա�����ֵ��x
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

if r0<=0
    disp('��ʼ�ϰ����Ӳ������0��');
    x=NaN;
    minf=NaN;
    return;
end

if c>=1||c<0
    disp('��Сϵ���������0��С��1!');
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
FF=r0*FE+FH/sqrt(r0);
SumF=f+FF;
[x2,minf]=minNT(SumF,transpose(x1),var);

while 1
    FF=r0*FE+FH/sqrt(r0);                   %��������Ŀ�꺯��
    SumF=f+FF;
    a0=(c*x1-x2)/(c-1);
    x2=a0+(x1-a0)*c^2;                      %��幫ʽ
    [x3,minf]=minNT(SumF,transpose(x2),var);    %��ţ�ٷ�����Լ���滮
    
    if norm(x3-x2)<=eps
        x=x3;
        break;
    else
        r0=c*r0;
        x1=x2;
        x2=x3;
    end
end
minf=Funval(f,var,x);