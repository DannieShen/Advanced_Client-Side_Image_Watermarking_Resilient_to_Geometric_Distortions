function [arbitration,BER] = detection_my(Bm,G,SA_l,w_vec,b_k, Delta, delta)
    G = sparse(G);
    GG = full(Bm * G);
    wp = SA_l' * w_vec;
    wp_q = (Delta * round((wp + delta)/Delta)-delta);
    e = wp - wp_q;
    ee = GG' * e;
    b_arb = sign(ee);

    b_arb = floor((b_arb+1)/2);

%     fprintf("b_arb:");
%     fprintf("%d",b_arb);
%     fprintf("\n");
%     fprintf("b_k  :");
%     fprintf("%d",b_k);
%     fprintf("\n");

    s = 0;
    for i = 1:length(b_k)
        if b_arb(i) == b_k(i)
            s = s + 1;
        end
    end
    
    BER = (1-s/length(b_k))*100;
    if s == length(b_k)
%         fprintf("BER: %.1f%%\n",(1-s/length(b_k))*100);
        arbitration = 1;
    else
%         fprintf("BER: %.1f%%\n",(1-s/length(b_k))*100);
        arbitration = 0;
    end
end


