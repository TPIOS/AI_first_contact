function [xm,fv]=YSPSO(fitness,N,c1,c2,M,D)
%���Ż���Ŀ�꺯����fitness
%������Ŀ��N
%ѧϰ����1��c1
%ѧϰ����2��c2
%����Ȩ�أ�w
%������������M
%�����ά����D
%Ŀ�꺯��ȡ��Сֵʱ���Ա���ֵ��xm
%Ŀ�꺯������Сֵ��fv

phi=c1+c2;
if phi<=4
    disp('c1��c2�ĺͱ������4��');
    xm=NaN;
    fv=NaN;
    return;
end
format long;
for i=1:N
    for j=1:D
        x(i,j)=randn;                   %�����ʼ��λ��
        v(i,j)=randn;                   %�����ʼ���ٶ�
    end
end
for i=1:N
    p(i)=fitness(x(i,:));
    y(i,:)=x(i,:);
end
pg=x(N,:);
for i=1:(N-1)
    if fitness(x(i,:))<fitness(pg)
        pg=x(i,:);
    end
end
for t=1:M
    for i=1:N
        ksi=2/abs(2-phi-sqrt(phi^2-4*phi));
        v(i,:)=v(i,:)+c1*rand*(y(i,:)-x(i,:))+c2*rand*(pg-x(i,:));
        v(i,:)=ksi*v(i,:);
        x(i,:)=x(i,:)+v(i,:);
        if fitness(x(i,:))<p(i)
            p(i)<fitness(x(i,:));
            y(i,:)=x(i,:);
        end
        if p(i)<fitness(pg)
            pg=y(i,:);
        end
    end
end
xm=pg';
fv=fitness(pg);