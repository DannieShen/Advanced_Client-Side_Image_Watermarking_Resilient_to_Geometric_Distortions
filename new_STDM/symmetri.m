function [w_v] = symmetri(w_v,PQ,flag)
    N = length(flag);
    for i = 1:N
        if flag(i) == 1
            n = PQ(i,1);
            m = PQ(i,2);
            index = PQ(:,1)==n & PQ(:,2)==-m;
            w_v(index) = conj(w_v(i));
        end
    end
end
