function [Bm] = Bm_Gen(L,T,R,sk_w)
    Bm = zeros(L,T);
    for i = 1:L
        for j = 1:R
            Bm(i,sk_w((i-1)*R+j)) = 1;
        end
    end
    Bm = sparse(Bm);
 end

