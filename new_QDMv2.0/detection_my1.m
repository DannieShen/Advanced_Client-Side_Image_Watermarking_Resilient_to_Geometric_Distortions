function [b_arb] = detection_my1(Bm,G,w_vec,b_k,Delta,delta,flag)
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
    wp_q = floor((wp+delta)/Delta)*Delta-delta;
       e = wp - wp_q;
    b_arb = zeros(length(b_k),1);
    for i = 1:length(b_k)
        if e(i)>=Delta/2
            b_arb(i) = 1;
        elseif e(i)<Delta/2
            b_arb(i) = 0;
        end
    end
    fprintf("b_arb:");
    fprintf("%d",b_arb);
    fprintf("\n");
    fprintf("b_k  :");
    fprintf("%d",b_k);
    fprintf("\n");
end


