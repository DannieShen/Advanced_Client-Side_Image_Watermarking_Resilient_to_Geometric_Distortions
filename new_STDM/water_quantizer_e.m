function [em_q] = water_quantizer_e(A,SA_l,delta,Delta)
    ro = SA_l' * A;
    ro_q = Delta * round((ro+delta)/Delta) - delta;

    [~,L] = size(SA_l);
    [M,N] = size(A);
    em_q = zeros(M,N);
    for i = 1:L
        em_q = em_q + ro_q(i)*SA_l(:,i) - ro(i)*SA_l(:,i);
    end
end