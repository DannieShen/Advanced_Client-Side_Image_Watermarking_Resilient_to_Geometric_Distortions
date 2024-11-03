function [arbitration] = detection_my(Bm,G,w_vec,b_k,Delta,delta,flag)
    G = sparse(G);
    GG = full(Bm * G);
    M = length(flag);
    L = sum(flag);
    wp = zeros(L,1);
    k = 1;
    for i = 1:M
        if flag(i) == 1
            wp(k) = w_vec(i);
            k = k + 1;
        end
    end
    wp_q = floor(floor(wp+delta)/Delta)*Delta-delta;
       e = wp - wp_q;
    es = e - Delta/2;
    ee = GG' * es;
    b_arb = sign(ee);

    b_arb = floor((b_arb+1)/2);

    fprintf("b_arb:");
    fprintf("%d",b_arb);
    fprintf("\n");
    fprintf("b_k  :");
    fprintf("%d",b_k);
    fprintf("\n");

    s = 0;
    for i = 1:length(b_k)
        if b_arb(i) == b_k(i)
            s = s + 1;
        end
    end

    fprintf("BER: %.1f%%\n",(1-s/length(b_k))*100);

    if s == length(b_k)
        arbitration = 1;
    else
        arbitration = 0;
    end
end


