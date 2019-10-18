function [x,minf]=minRosen(f,A,b,x0,var,eps)
%�õ��˻ƽ��и�ͽ��˷�
%Ŀ�꺯����f
%Լ������A
%Լ���Ҷ�������b
%��ʼ���е㣺x0
%�Ա���������var
%���ȣ�eps
%Ŀ�꺯��ȡ��Сֵʱ���Ա���ֵ��x
%Ŀ�꺯������Сֵ��minf

format long
if nargin==5
    eps=1.0e-6;
end

syms l;
x0=transpose(x0);
n=length(var);
sz=size(A);
m=sz(1);

gf=jacobian(f,var);
bConti=1;

while bConti
    k=0;
    s=0;
    A1=A;
    A2=A;
    b1=b;
    b2=b;
    for i=1:m
        dfun=A(i,:)*x0-b(i);
        if abs(dfun)<0.000000001        %��Լ��������ϵ��������������ֽ�
            k=k+1;
            A1(k,:)=A(i,:);             %A1�����ʽԼ��ϵ������
            b1(k,1)=b(i);               %b1�����ʽԼ������
        else
            s=s+1;
            A2(s,:)=A(i,:);              %A2������ʽԼ��ϵ������
            b2(s,1)=b(i);               %b2������ʽԼ��ϵ������
        end
    end
    if k>0
        A1=A1(1:k,:);
        b1=b1(1:k,:);
    end
    if s>0
        A2=A2(1:s,:);
        b2=b2(1:s,:);
    end
    
    while 1
        P=eye(n,n);
        if k>0
            tM=transpose(A1);
            P=P - tM*inv(A1*tM)*A1;
        end
        gv=Funval(gf,var,x0);
        gv=transpose(gv);
        d=-P*gv;                        %dΪ��������
        if d==0
            if k==0
                x=x0;
                bConti=0;
                break;
            else
                w=inv(A1*tM)*A1*gv;
                if w>=0                 %w����ȫΪ��
                    x=x0;
                    bConti=0;
                    break;
                else
                    [u,index]=min(w);
                    sA1=size(A1);
                    if sA1(1)==1
                        k=0;
                    else
                        k=sA1(2);           %ѡ��w��һ��������
                        A1=[ A1(1:(index-1),:); A1((index+1):sA1(2),:)];
                            %ȥ��A1��Ӧ����
                    end
                end
            end
        else
            break;
        end
    end
    
    yl=x0+l*d;
    tmpf=Funval(f,var,yl);
    bb=b2-A2*x0;
    dd=A2*d;
    if dd>=0
        [tmpI,lm]=minJT(tmpf,0,0.1);%�ý��˷�ȷ��һά��ֵ����ļ�ֵ����
    else
        lm=inf;
        for i=1:length(dd)
            if dd(i)<0
                if bb(i)/dd(i)<lm
                   lm=bb(i)/dd(i);
                end
            end
        end
    end
    [xm,minf]=minHJ(tmpf,0,lm,1.0e-14);     %�ûƽ�ָ���һά��ֵ����
    tol=norm(xm*d);
    if tol<eps
       x=x0;
       break;
    end
    x0=x0+xm*d;
end
    
minf=Funval(f,var,x);