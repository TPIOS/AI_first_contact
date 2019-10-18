function [x,minf]=minHJ(f,a,b,eps)
%Ŀ�꺯����f
%��ֵ������˵㣺a
%��ֵ�����Ҷ˵㣺b
%���ȣ�eps
%Ŀ�꺯��ȡ��Сֵʱ�Ա�����ֵ��x
%Ŀ�꺯����ȡ����Сֵ��minf

format long;
if nargin==3
    eps=1.0e-6;
end

l=a+0.382*(b-a);            %��̽��
u=a+0.618*(b-a);            %��̽��
k=1;
tol=b-a;

while tol>eps&&k<100000
    fl=subs(f,symvar(f),l);        %��̽�㺯��ֵ
    fu=subs(f,symvar(f),u);        %��̽�㺯��ֵ
    if fl>fu
        a=1;                        %�ı�������˵�
        l=u;
        u=a+0.618*(b-a);            %������������
    else
        b=u;                        %�ı������Ҷ˵�
        u=l;
        l=a+0.382*(b-a);             %������������
    end
    k=k+1;
    tol=abs(b-a);
end
if k==100000
    disp('�Ҳ�����Сֵ��');
    x=NaN;
    minf=NaN;
    return;
end
x=(a+b)/2;
minf=subs(f,symvar(f),x);
format short;