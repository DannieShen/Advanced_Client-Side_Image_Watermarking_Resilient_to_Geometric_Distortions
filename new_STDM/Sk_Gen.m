function [sk_a,sk_w] = Sk_Gen(G,M_D,SAl_index,T1,T,L,len_skm,R)
%     sk_a = zeros(2*M_D*R,1);
    sk_a = zeros(M_D*R,1);
    sk_w = zeros(len_skm*R,1);
    k = 1;
    h = 1;
    for i = 1:M_D
        if i == SAl_index(k)
            source = find(G(:,h)==1);
            ind = randperm(length(source));
            for j = 1:R
                sk_a((i-1)*R+j) = source(ind(j));
                sk_w((k-1)*R+j) = source(ind(j));
            end
            if k < len_skm
                k = k + 1;
                if h < L
                    h = h + 1;
                else
                    h = ceil(rand()*L);
                end
            end
        else
            sk_a((i-1)*R+1:i*R) = randperm(T-T1, R) + T1;
        end
    end
%     for i = M_D+1:2*M_D
%         sk_a((i-1)*R+1:i*R) = randperm(T1,R);
%     end
end

