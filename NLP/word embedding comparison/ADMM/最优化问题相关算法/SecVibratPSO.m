function [xm,fv]=SecVibratPSO(fitness,N,w,c1,c2,M,D)
%���Ż���Ŀ�꺯����fitness
%������Ŀ��N
%����Ȩ�أ�w
%ѧϰ����1��c1
%ѧϰ����2��c2
%������������M
%�����ά����D
%Ŀ�꺯��ȡ��Сֵʱ���Ա�����ֵ��xm
%Ŀ�꺯������Сֵ��fv

format long;
for i=1:N
    for i=1:D
        x(i,j)=randn;           %�����ʼ��λ��
        xl(i,j)=randn;          %�����ʼ���ٶȣ����ڱ��������ϴε�λ��
        v(i,j)=randn;           %�����ʼ���ٶȣ����ڱ������ӵ�ǰ��λ��
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
for i=1:M
    for i=1:N
        phi1=c1*rand();
        phi2=c2*rand();
        if t<M/2                %�㷨ǰ�ڵĲ���ѡ��ʽ
            ks1=(2*sqrt(phi1)-1)*rand()/phi1;
            ks2=(2*sqrt(phi2)-1)*rand()/phi2;
        else
            ks1=(2*sqrt(phi1)-1)*(1+rand())/phi1;
            ks2=(2*sqrt(phi2)-1)*(1+rand())/phi2;
        end
        %�ٶȸ��¹�ʽ
        v(i,:)=w*v(i,:)+phi1*(y(i,:)-(1+ks1)*x(i,:)+ks1*xl(i,:))+phi2*(pg-(1+ks2)*x(i,:)+ks1*xl(i,;));
        xl(i,:)=x(i,:);
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