function [xm,fv]=RandWPSO(fitness,N,c1,c2,mean_max,mean_min,sigma,M,D)
%���Ż�Ŀ�꺯����fitness
%������Ŀ��N
%ѧϰ����1��c1
%ѧϰ����2��c2
%���Ȩ��ƽ��ֵ�����ֵ��mean_max
%���Ȩ��ƽ��ֵ����Сֵ��mean_min
%���Ȩ�صķ��sigma
%������������M
%�����ά����D
%Ŀ�꺯��ȡ��Сֵʱ���Ա���ֵ��xm
%Ŀ�꺯������Сֵ��fv

format long;
for i=1:N
    for j=1:D;
        x(i,j)=randn;      %�����ʼ��λ��
        v(i,j)=randn;      %�����ʼ���ٶ�
    end
end
for i=1:N
    p(i)=fitness(x(i,:));
    y(i,:)=x(i,:);
end
pg=x(N,:);                  %pgΪȫ������
for i=1:(N-1)
    if fitness(x(i,:))<fitness(pg)
        pg=x(i,:);
    end
end
for t=1:M
    for i=1:N
        miu=mean_min+(mean_max-mean_min)*rand();    %���Ȩ�ص�ƽ��ֵ
        w=miu+sigma*randn();                        %���Ȩ��
        v(i,:)=w*v(i,:)+c1*rand*(y(i,:)-x(i,:))+c2*rand*(pg-x(i,:));
        x(i,:)=x(i,:)+v(i,:);
        if fitness(x(i,:))<p(i)
            p(i)=fitness(x(i,:));
            y(i,:)=x(i,:);
        end
        if p(i)<fitness(pg)
            pg=y(i,:);
        end
    end
end
xm=pg';
fv=fitness(pg);