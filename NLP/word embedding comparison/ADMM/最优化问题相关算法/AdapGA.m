function [xv,fv]=AdapGA(fitness,a,b,NP,NG,Pc1,Pc2,Pm1,Pm2,eps)
%���Ż�Ŀ�꺯����fitness
%�Ա����½磺a
%�Ա����½磺b
%��Ⱥ��������NP
%������������NG
%�ӽ�����1��Pc1
%�ӽ�����2��Pc2
%���쳣��1��Pm1
%���쳣��2��Pm2
%�Ա�����ɢ���ȣ�eps
%Ŀ�꺯��ȡ��Сֵʱ���Ա���ֵ��xm
%Ŀ�꺯������Сֵ��fv

L=ceil(log2((b-a)/eps+1));          %������ɢ���ȣ�ȷ�������Ʊ���������볤
x=zeros(NP,L);
for i=1:NP
    x(i,:)=Initial(L);              %��Ⱥ��ʼ��
    fx(i)=fitness(Dec(a,b,x(i,:),L));   %������Ӧֵ
end
for k=1:NG
    sumfx=sum(fx);                  %���и�����Ӧֵ֮��
    Px=fx/sumfx;                    %���и�����Ӧֵ��ƽ��ֵ
    PPx=0;
    PPx(1)=Px(1);
    for i=2:NP                      %�������̶Ĳ��Եĸ�������
        PPx(i)=PPx(i-1)+Px(i);
    end
    for i=1:NP
        sita=rand();
        for n=1:NP
            if sita<=PPx(n)
                SelFather=n;        %�������̶Ĳ���ȷ���ĸ���
                break;
            end
        end
        Selmother=floor(rand()*(NP-1))+1;   %���ѡ��ĸ��
        posCut=floor(rand()*(L-2))+1;       %���ȷ�������
        favg=sumfx/NP;                      %Ⱥ��ƽ����Ӧֵ
        fmax=max(fx);                       %Ⱥ�������Ӧֵ
        Fitness_f=fx(SelFather);            %����ĸ�����Ӧֵ
        Fitness_m=fx(Selmother);            %�����ĸ����Ӧֵ
        Fm=max(Fitness_f,Fitness_m);        %����˫���ϴ����Ӧֵ
        if Fm>=favg
            Pc=Pc1*(fmax-Fm)/(fmax-favg);
        else
            Pc=Pc2;
        end
        r1=rand();
        if r1<=Pc                           %����
            nx(i,1:posCut)=x(SelFather,1:posCut);
            nx(i,(posCut+1):L)=x(Selmother,(posCut+1):L);
            fmu=fitness(Dec(a,b,nx(i,:),L));
            if fmu>=favg
                Pm=Pm1*(fmax-fmu)/(fmax-favg);
            else
                Pm=Pm2;
            end
            r2=rand();
            if r2<=Pm                       %����
                posMut=round(rand()*(L-1)+1);
                nx(i,posMut)=~nx(i,posMut);
            end
        else
            nx(i,:)=x(SelFather,:);
        end
    end
    x=nx;
    for i=1:NP
        fx(i)=fitness(Dec(a,b,x(i,:),L));
    end
end
fv=-inf;
for i=1:NP
    fitx=fitness(Dec(a,b,x(i,:),L));
    if fitx>fv
        fv=fitx;                            %ȥ�����е����ֵ��Ϊ���ս��
        xv=Dec(a,b,x(i,:),L);
    end
end
function result=Initial(length)             %��ʼ������
for i=1:length
    r=rand();
    result(i)=round(r);
end
function y=Dec(a,b,x,L)                     %�����Ʊ���ת��Ϊʮ���Ʊ���
base=2.^((L-1):-1:0);
y=dot(base,x);
y=a+y*(b-a)/(2^L-1);