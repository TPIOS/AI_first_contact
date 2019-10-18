function [x,fm]=IntProgFZ(f,A,b,Aeq,beq,lb,ub)
%Ŀ�꺯��ϵ��������f
%����ʽԼ������A
%����ʽԼ���Ҷ�������b
%��ʽԼ������Aeq
%��ʽԼ���Ҷ�������beq
%�Ա����½磺lb
%�Ա����Ͻ磺ub
%Ŀ�꺯��ȡ��Сֵʱ���Ա�����ֵ
%Ŀ�꺯������Сֵ��minf

x=NaN;
fm=NaN;
NF_lb=zeros(size(lb));
NF_ub=zeros(size(ub));
NF_lb(:,1)=lb;
NF_ub(:,1)=ub;
F=inf;

while 1
    sz=size(NF_lb);
    k=sz(2);
    opt=optimset('TolX',1e-9);
    
    %������Թ滮
    [xm,fv,exitflag]=linprog(f,A,b,Aeq,beq,NF_lb(:,1),NF_ub(:,1),[],opt);
    if exitflag==-2             %���������Ž�
        xm=NaN;
        fv=NaN;
    end
    if xm==NaN
        fv=inf;
    end
    if fv~=inf
        if fv<F
            if max(abs(round(xm)-xm))<1.0e-7    %�жϸ��������Ƿ�Ϊ����
                F=fv;
                x=xm;
                tmpNF_lb=NF_lb(:,2:k);          %ȥ����һ��
                tmpNF_ub=NF_ub(:,2:k);          %ȥ����һ��
                NF_lb=tmpNF_lb;
                NF_ub=tmpNF_ub;
                if isempty(NF_lb)==0
                    continue;
                else
                    if x~=NaN
                        fm=F;
                        return;
                    else
                        disp('���������Ž⣡');
                        x=NaN;
                        fm=NaN;
                        return;
                    end
                end
            else
                lb1=NF_lb(:,1);
                ub1=NF_ub(:,1);
                tmpNF_lb=NF_lb(:,2:k);          %ȥ����һ��
                tmpNF_ub=NF_ub(:,2:k);          %ȥ����һ��
                NF_lb=tmpNF_lb;
                NF_ub=tmpNF_ub;
                [bArr,index]=find(abs((xm-round(xm)))>1.0e-7);
                %����һ������������
                p=bArr(1);
                new_lb=lb1;
                new_ub=ub1;
                new_lb(p)=max(floor(xm(p))+1,lb1(p));   %��������
                new_ub(p)=min(floor(xm(p)),ub1(p));     %��������
                NF_lb=[NF_lb new_lb lb1];
                NF_ub=[NF_ub ub1 new_ub];
                continue;
            end
        else                                %fv����F
            tmpNF_lb=NF_lb(:,2:k);          %ȥ����һ��
            tmpNF_ub=NF_ub(:,2:k);          %ȥ����һ��
            NF_lb=tmpNF_lb;
            NF_ub=tmpNF_ub;
            if isempty(NF_lb)==0
                continue;
            else
                if x~=NaN
                    fm=F;
                    return;
                else
                    disp('���������Ž�!');
                    x=NaN;
                    fm=NaN;
                    return;
                end
            end
        end
    else                        %fvΪ�����
        tmpNF_lb=NF_lb(:,2:k);  %ȥ����һ��
        tmpNF_ub=NF_ub(:,2:k);  %ȥ����һ��
        NF_lb=tmpNF_lb;
        NF_ub=tmpNF_ub;
        if isempty(NF_lb)==0
            continue;
        else
            if x~=NaN
                fm=F;
                return;
            else
                disp('���������Ž�!');
                x=NaN;
                fm=NaN;
                return;
            end
        end
    end
end
                
            