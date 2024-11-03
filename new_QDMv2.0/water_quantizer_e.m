function [embed_q] = water_quantizer_e(A_enc_abs,flag,delta,Delta)
    M = length(flag);
    embed_q = zeros(M,1);
    L = sum(flag);
    k = 1;
    for i = 1:M
        embed_q(i) = A_enc_abs(i)*flag(i);
        if flag(i) == 1
            embed_q(i) = floor((embed_q(i)+delta(k))/Delta)*Delta+Delta/2-delta(k);
            k = k+1;
        end
    end
    if k ~= L+1
        fprintf("WRONG");
    end
end

