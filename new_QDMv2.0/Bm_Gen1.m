function [Bm] = Bm_Gen1(L,T,R,sk_a, flag)
    Bm = zeros(L,T);
    M = length(flag);
    k = 1;
    for i = 1:M
        if flag(i)==1
            for j = 1:R
                Bm(k,sk_a((i-1)*R+j)) = 1;
            end
            k = k + 1;
        end
    end
    if k == L+1
        fprintf("correct!\n");
    else
        fprintf("incorrect!k=%d\n",k);
    end
    Bm = sparse(Bm);
 end