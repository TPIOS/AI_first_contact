function [xm,fv]=LnCPSO(fitness,N,cmax,cmin,w,M,D)
%���Ż���Ŀ�꺯����fitness
%������Ŀ��N
%ѧϰ���ӵ����ֵ��cmax
%ѧϰ���ӵ���Сֵ��cmin
%����Ȩ�أ�w
%������������M
%�����ά����D
%Ŀ�꺯��ȡ��Сֵʱ���Ա���ֵ��xm
%Ŀ�꺯������Сֵ��fv

format long;
for i=1:N
    for j=1:D
        x(i,j)=randn;           %�����ʼ��λ��
        v(i,j)=randn;           %�����ʼ���ٶ�
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
    c=cmax-(cmax-cmin)*t/M;     %����ͬ���仯��ѧϰ����
    for i=1:N
        v(i,:)=w*v(i,:)+c*rand*(y(i,:)-x(i,:))+c*rand*(pg-x(i,:));
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
