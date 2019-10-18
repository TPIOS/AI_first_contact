function [xm,fv]=SimuAPSO(fitness,N,c1,c2,lamda,M,D)
%���Ż���Ŀ�꺯����fitness
%������Ŀ��N
%ѧϰ����1��c1
%ѧϰ����2��c2
%�˻�����lamda
%������������M
%�����ά����D
%Ŀ�꺯��ȡ��Сֵʱ���Ա���ֵ��xm
%Ŀ�꺯������Сֵ��fv

format long;
for i=1:N
    for j=1:D
        x(i,j)=randn;               %�����ʼ��λ��
        v(i,j)=randn;               %�����ʼ���ٶ�
    end
end
for i=1:N
    p(i)=fitness(x(i,:));
    y(i,:)=x(i,:);
end
pg=x(N,:);                          %pgΪȫ������
for i=1:(N-1)
    if fitness(x(i,:))<fitness(pg)
        pg=x(i,:);
    end
end
T=fitness(pg)/log(5);               %��ʼ�¶�
for t=1:M
    groupFit=fitness(pg);
    for i=1:N
        Tfit(i)=exp(-(p(i)-groupFit)/T);
    end
    SumTfit=sum(Tfit);
    Tfit=Tfit/SumTfit;
    pBet=rand();
    for i=1:N                       %�����̶Ĳ���ȷ��ȫ�����ŵ�ĳ�����ֵ
        ComFit(i)=sum(Tfit(1:i));
        if pBet<=ComFit(i)
            pg_plus=x(i,:);
            break;
        end
    end
    C=c1+c2;
    ksi=2/abs(2-C-sqrt(C^2-4*C));   %�ٶ�ѹ������
    for i=1:N
        v(i,:)=ksi*(v(i,:)+c1*rand*(y(i,:)-x(i,:))+c2*rand*(pg_plus-x(i,:)));
        x(i,:)=x(i,:)+v(i,:);
        if fitness(x(i,:))<p(i)
            p(i)=fitness(x(i,:));
            y(i,:)=x(i,:);
        end
        if p(i)<fitness(pg)
            pg=y(i,:);
        end
    end
    T=T*lamda;                      %���²���
end
xm=pg';
fv=fitness(pg);