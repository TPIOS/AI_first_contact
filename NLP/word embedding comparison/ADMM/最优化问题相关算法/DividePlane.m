function [intx,intf]=DividePlane(A,c,b,baseVector)
%Լ������A��
%Ŀ�꺯��ϵ��������c��
%Լ���Ҷ�������b��
%��ʼ��������baseVector
%Ŀ�꺯��ȡ��Сֵʱ���Ա���ֵ��x��
%Ŀ�꺯������Сֵ��minf

%ע�����Ŀ�꺯����ϵ����������0��������������ǻ�����


sz=size(A);
nVia=sz(2);
n=sz(1);
xx=1:nVia;

if length(baseVector)~=n
    disp('�������ĸ���Ҫ��Լ�������������ȣ�');
    mx=NaN;
    mf=NaN;
    return;
end

M=0;
sigma=-[transpose(c) zeros(1,(nVia-length(c)))];
xb=b;

while 1
    %�����õ����η��������Ž⣬���µĳ�����̿ɲο����Թ滮�ĵ����ͷ�����
    [maxs,ind]=max(sigma);
    if maxs<=0                          %�õ����η��ҵ����Ž�
        vr=find(c~=0,1,'last');
        for l=1:vr
            ele=find(baseVector==l,1);
            if(isempty(ele))
                mx(l)=0;
            else
                mx(l)=xb(ele);
            end
        end
        if max(abs(round(mx)-mx))<1.0e-7    %�ж����Ž��Ƿ�Ϊ����
            intx=mx;
            intf=mx*c;
            return;
        else
            sz=size(A);
            sr=sz(1);
            sc=sz(2);
            [max_x,index_x]=max(abs(round(mx)-mx));
            [isB,num]=find(index_x==baseVector);
            fi=xb(num)-floor(xb(num));
            for i=1:(index_x-1)
                Atmp(1,i)=A(num,i)-floor(A(num,i));
            end
            for i=(index_x+1):sc
                Atmp(1,i)=A(num,i)-floor(A(num,i));
            end
            
            
            %���³�����乹����ż�����η��ĳ�ʼ���
            Atmp(1,index_x)=0;
            A=[A zeros(sr,1);-Atmp(1,:) 1];
            xb=[xb;-fi];
            baseVector=[baseVector sc+1];
            sigma=[sigma 0];
            
            %��ż�����η��ĵ�������
            while 1
                if xb>=0
                    if max(abs(round(xb)-xb))<1.0e-7
                        %�ö�ż�����η������������
                        vr=find(c~=0,1,'last');
                        for l=1:vr
                            ele=find(baseVector==l,1);
                            if(isempty(ele))
                                mx_l(1)=0;
                            else
                                mx_l(1)=xb(ele);
                            end
                        end
                        intx=mx_1;
                        intf=mx_1*c;
                        return;
                    else            %���Žⲻ�������⣬��������и��
                        sz=size(A);
                        sr=sz(1);
                        sc=sz(2);
                        [max_x,index_x]=max(abs(round(mx_1)-mx_1));
                        [isB,num]=find(index_x==baseVector);
                        fi=xb(num)-floor(xb(num));
                        for i=1:(index_x-1)
                            Atmp(1,i)=A(num,i)-floor(A(num,i));
                        end
                        for i=(index_x+1):sc
                            Atmp(1,i)=A(num,i)-floor(A(num,i));
                        end
                        
                        %��������Ƕ���һ�ε����ε����ĳ�ʼ���
                        Atmp(1,index_x)=0;
                        A=[A zeros(sr,1);-Atmp(1,:) 1];
                        xb=[xb;-fi];
                        baseVector=[baseVector sc+1];
                        sigma=[sigma 0];
                        continue;
                    end
                else
                    minb_1=inf;
                    chagB_1=inf;
                    sA=aize(A);
                    [br,idb]=min(xb);
                    for j=1:sA(2)
                        if A(idb,j)<0
                            bm=sigma(j)/A(idb,j);
                            if bm<minb_1
                                minb_1=bm;
                                chagB_1=j;
                            end
                        end
                    end
                    sigma=sigma-A(idb,:)*minb_1;
                    xb(idb)=xb(idb)/A(idb,chagB_1);
                    A(idb,:)=A(idb,:)/A(idb,chagB_1);
                    for i=1:sA(1)
                        if i~=idb
                            xb(i)=xb(i)-A(i,chagB_1)*xb(idb);
                            A(i,:)=A(i,:)-A(i,chagB_1)*A(idb,:);
                        end
                    end
                    baseVector(idb)=chagB_1;
                end
            end
        end
    else
        %����Ϊ�����η��ĵ�������
        minb=inf;
        chagB=inf;
        for j=1:n
            if A(j,ind)>0
                bz=xb(j)/A(j,ind);
                if bz<minb
                    minb=bz;
                    chagB=j;
                end
            end
        end
        sigma=sigma-A(chagB,:)*maxs/A(chagB,ind);
        xb(chagB)=xb(chagB)/A(chagB,ind);
        A(chagB,:)=A(chagB,:)/A(chagB,ind);
        for i=1:n
            if i~=chagB
                xb(i)=xb(i)-A(i,ind)*xb(chagB);
                A(i,:)=A(i,:)-A(i,ind)*A(chagB,:);
            end
        end
        baseVector(chagB)=ind;
    end
    M=M+1;
    if(M==1000000)
        disp('�Ҳ������Ž�');
        mx=NaN;
        minf=NaN;
        return;
    end
end