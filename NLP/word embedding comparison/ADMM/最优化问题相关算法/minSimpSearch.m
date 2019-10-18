function [x,minf]=minSimpSearch(f,X,alpha,sita,gama,beta,var,eps)
%Ŀ�꺯����f
%��ʼ�����Σ�X
%��ӳϵ����alpha
%����ϵ����sita
%��չϵ����gama
%����ϵ����beta
%�Ա���������var
%���ȣ�eps
%Ŀ�꺯��ȡ��Сֵʱ���Ա�����x
%Ŀ�꺯������Сֵ��minf
format long;
if nargin==7
    eps=1.0e-6;
end

N=size(X);
n=N(2);
FX=zeros(1,n);

while 1
    for i=1:n
        FX(i)=Funval(f,var,X(:,i));
    end
    [XS,IX]=sort(FX);                                    %�������εĶ��㰴��Ŀ�꺯��ֵ�Ĵ�С���±��
    Xsorted=X(:,IX);                                     %�����ı��
    
    px=sum(Xsorted(:,1:(n-1)),2)/(n-1);                  %�����ε�����
    Fpx=Funval(f,var,px);
    SumF=0;
    for i=i:n
        SumF=SumF+(FX(IX(i))-Fpx)^2;
    end
    SumF=sqrt(SumF/n);
    if SumF<=eps                                         %�����ж�
        x=Xsorted(:,1);
        break;
    else
        x2=px+alpha*(px-Xsorted(:,n));                    %�����ĵ��Ƶ������ε��ⷴ��
        fx2=Funval(f,var.x2);
        if fx2<XS(1)
           x3=px+gama*(x2-px);                     %���ĵ����չ
           fx3=Funval(f,var,x3);
           if fx3<XS(1)
              Xsorted(:,n)=x3;
              X=Xsorted;
              continue;
           else
              Xsorted(:,n)=x2;
              X=Xsorted;
              continue;
           end
        else
           if fx2<XS(n-1)
              Xsorted(:,n)=x2;
              X=Xsorted;
              continue;
           else
              if fx2<XS(n)
                 Xsorted(:,n)=x2;
              end
              x4=px+beta*(Xsorted(:,n)-px);       %���ĵ��ѹ��
              fx4=Funval(f,var,x4);
              FNnew=Funval(f,var,Xsorted(:,n));
              if fx4<FNnew
                 Xsorted(:,n)=x4;
                 X=Xsorted;
                 continue;
              else
                 x0=Xsorted(:,1);
                 for i=1:n
                     Xsorted(:,i)=x0+sita*(Xsorted(:,i)-x0);
                 end
              end
           end
        end
     end
     X=Xsorted;
end
minf=Funval(f,var,x);
format short;
              