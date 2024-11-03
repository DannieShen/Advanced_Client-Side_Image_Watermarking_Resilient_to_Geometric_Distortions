function [Bm] = Bm_Gen1(T,R,sk_a,SAl_index)
    len_skm = length(SAl_index);
    Bm = zeros(len_skm,T);
    for i = 1:length(SAl_index)
        for j = 1:R
            Bm(i,sk_a((SAl_index(i)-1)*R+j)) = 1;
        end
    end
    Bm = sparse(Bm);
end

