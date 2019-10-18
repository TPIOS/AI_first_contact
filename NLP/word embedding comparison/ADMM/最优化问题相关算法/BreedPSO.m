function [xm,fv]=BreedPSO(fitness,N,c1,c2,w,Pc,Sp,M,D)
%���Ż���Ŀ�꺯����fitness
%������Ŀ��N
%ѧϰ����1��c1
%ѧϰ����2��c2
%����Ȩ�أ�w
%�ӽ����ʣ�Pc
%�ӽ��ش�С������Sp
%������������M
%�����ά����D
%Ŀ�꺯��ȡ��Сֵʱ���Ա���ֵ��xm
%Ŀ�꺯������Сֵ��fv

format long;
for i=1:N
    for j=1:D
        x(i,j)=randn;           %��ʼ�����λ��
        v(i,j)=randn;           %��ʼ������ٶ�
    end
end
for i=1:N
    p(i)=fitness(x(i,:));
    y(i,:)=x(i,:);
end
pg=x(N,:);                      %pgΪȫ������
for i=1:(N-1)
    if fitness(x(i,:))<fitness(pg)
        pg=x(i,:);
    end
end
for t=1:M
    for i=1:N
        v(i,:)=w*v(i,:)+c1*rand*(y(i,:)-x(i,:))+c2*rand*(pg-x(i,:));
        x(i,:)=x(i,:)+v(i,:);
        if fitness(x(i,:))<p(i)
            p(i)=fitness(x(i,:));
            y(i,:)=x(i,:);
        end
        if p(i)<fitness(pg)
            pg=y(i,:);
        end
        r1=rand();
        if r1<Pc                        %�ӽ�����
            numPool=round(Sp*N);            %�ӽ��صĴ�С
            PoolX=x(1:numPool,:);           %�ӽ����е����ӵ�λ��
            PoolVX=v(1:numPool,:);          %�ӽ����е����ӵ��ٶ�
            for i=1:numPool
                seed1=floor(rand()*(numPool-1))+1;
                seed2=floor(rand()*(numPool-1))+1;
                pb=rand();
                %�Ӵ���λ�ü���
                childx1(i,:)=pb*PoolX(seed1,:)+(1-pb)*PoolX(seed2,:);
                %�Ӵ����ٶȼ���
                childv1(i,:)=(PoolVX(seed1,:)+PoolVX(seed2,:))*norm(PoolVX(seed1,:))/norm(PoolVX(seed1,:)+PoolVX(seed2,:));
            end
            x(1:numPool,:)=childx1;         %�Ӵ���λ���滻����λ��
            v(1:numPool,:)=childv1;         %�Դ����ٶ��滻�����ٶ�
        end
    end
end
xm=pg';
fv=fitness(pg);