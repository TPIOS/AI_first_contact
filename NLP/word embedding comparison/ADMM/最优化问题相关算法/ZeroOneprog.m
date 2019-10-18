function [intx,intf]=ZeroOneprog(c,A,b,x0)
%Ŀ�꺯��ϵ��������c
%����ʽԼ������A
%����ʽԼ���Ҷ�������b
%��ʼ�������н⣺x0
%Ŀ�꺯��ȡ��Сֵʱ���Ա���ֵ��intx��
%Ŀ�꺯������Сֵ��intf

sz=size(A);
if sz(2)<3
    [intx,intf]=Allprog(c,A,b);     %��ٷ�
else
    [intx,intf]=Implicitprog(c,A,b,x0);  %��ö�ٷ�
end

function [intx,inf]=Allprog(c,A,b)
sz_A=size(A);
rw=sz_A(1);
col=sz_A(2);

minf=inf;
for i=0:(2^(col)-1)             %ö�ٿռ�
    x1=myDec2Bin(i,col);        %ʮ����ת��Ϊ������
    if A*x1>=b                  %�Ƿ�����Լ������
        f_tmp=c*x1;
        if f_tmp<minf
            minf=f_tmp;
            intx=x1;
            intf=minf;
        else
            continue;
        end
    else
        continue;
    end
end

function [intx,intf]=Implicitprog(c,A,b,x0)         %��ö�ٷ�
sz_A=size(A);
rw=sz_A(1);
col=sz_A(2);

minf=c*x0;
A=[A;-c];
b=[b;-minf];                            %������һ�����Ʒ���
for i=0:(2^(col)-1)
    x1=myDec2Bin(i,col);
    if A*x1>=b
        f_tmp=c*x1;
        if f_tmp<minf
            minf=f_tmp;
            b(rw+1,1)=-minf;            %��ö�ٷ�����ٷ��������ڴʾ�
            intx=x1;
            intf=minf;
        else
            continue;
        end
    else
        continue;
    end
end

function y=myDec2Bin(x,n)               %ʮ����ת��Ϊ������
str=dec2bin(x,n);
for j=1:n
    y(j)=str2num(str(j));
end
y=transpose(y);
       