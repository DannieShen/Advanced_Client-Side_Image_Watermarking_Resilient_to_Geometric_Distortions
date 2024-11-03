function [S_l,Sym_l,SA_l,SAl_index] = S_Gen(flag,M_D,len_skm,PQ)
    M = length(flag);
    So_l = rand(sum(flag),M_D+1);
    So_l = orth(So_l);
    So_l(:,1) = [];

    % 取M_D个低频矩个数大小的标准正交向量
    S_l = zeros(M,M_D);
    Sym_l = zeros(M,M_D);
    j = 1;
    for i = 1:M
        if flag(i) == 1
            S_l(i,:) = So_l(j,:);
            n = PQ(i,1);
            m = PQ(i,2);
            index = PQ(:,1)==n & PQ(:,2)==-m;
            flag(index) = 1;
            Sym_l(index,:) = So_l(j,:);
            j = j + 1;
        end
    end
    % clear So_l
    
    % 随机从M_D中取len_skm个标准正交向量的下标
    SAl_index = randperm(M_D);
    SAl_index = SAl_index(1:len_skm);        
    SAl_index = sort(SAl_index);

    % 随机从M_D中取len_skm个标准正交向量
    SA_l = zeros(M,len_skm);
    for i = 1:len_skm
        SA_l(:,i) = S_l(:,SAl_index(i));
    end
end