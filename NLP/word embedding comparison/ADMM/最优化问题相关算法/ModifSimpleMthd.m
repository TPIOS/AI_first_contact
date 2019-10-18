function [x,minf]=ModifSimpleMthd(A,c,b,baseVector)
%Լ������A
%Ŀ�꺯��ϵ��������c
%Լ���Ҷ�������b
%��ʼ��������baseVector
%Ŀ�꺯��ȡ��Сֵʱ���Ա���ֵ��x
%Ŀ�꺯������Сֵ��minf


sz=size(A);
nVia=sz(2);
n=sz(1);
xx=1:nVia;
nobase=zeros(1,1);
m=1;

if c>=0
    vr=find(c~=0,1,'last');
    rgv=inv(A(:,(nVia-n+1):nVia))*b;
    if rgv>=0
        x=zeros(1,vr);
        minf=0;
    else
        disp('���������Ž�');
        x=NaN;
        minf=NaN;
        return;
    end
end

for i=1:nVia            %��ȡ�ǻ������±�
    if(isempty(find(baseVector==xx(i),1)))
        nobase(m)=i;
        m=m+1;
    else
        ;
    end
end

bCon=1;
M=0;
B=A(:,baseVector);
invB=inv(B);

while bCon
    nB=A(:,nobase);         %�ǻ���������
    ncb=c(nobase);          %�ǻ�����ϵ��
    B=A(:,baseVector);      %����������
    cb=c(baseVector);       %������ϵ��
    xb=invB*b;
    f=cb*xb;
    w=cb*invB;
    
    for i=1:length(nobase)  %�б�
        sigma(i)=w*nB(:,i)-ncb(i);
    end
    [maxs,ind]=max(sigma);  %indΪ���������±�
    if maxs<=0              %���ֵС���㣬���������
        minf=cb*xb;
        vr=find(c~=0,1,'last');
        for l=1:vr
            ele=find(baseVector==l,1);
            if(isempty(ele))
                x(l)=0;
            else
                x(l)=xb(ele);
            end
        end
        bCon=0;
    else
        y=inv(B)*A(:,nobase(ind));
        if y<=0             %���������Ž�
            disp('���������Ž⣡');
            x=NaN;
            minf=NaN;
            return;
        else
            minb=inf;
            chagB=0;
            for j=1:length(y)
                if y(j)>0
                    bz=xb(j)/y(j);
                    if bz<minb
                        minb=bz;
                        chagB=j;
                    end
                end
            end                     %chagBΪ�������±�
            tmp=baseVector(chagB);  %���»�����ͷǻ�����
            baseVector(chagB)=nobase(ind);
            nobase(ind)=tmp;
            
            for j=1:chagB-1         %����������������任
                if y(j)~=0
                    invB(j,:)=invB(j,:)-invB(chagB,:)*y(j)/y(chagB);
                end
            end
            for j=chagB+1:length(y)
                if y(j)~=0
                    invB(j,:)=invB(j,:)-invB(chagB,:)*y(j)/y(chagB);
                end
            end
            invB(chagB,:)=invB(chagB,:)/y(chagB);
            
        end
    end
    M=M+1;
    if(M==1000000)               %������������
        disp('�Ҳ������Ž⣡');
        x=NaN;
        minf=NaN;
        return;
    end
end
                 
       