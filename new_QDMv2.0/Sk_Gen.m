function [sk_a,sk_w] = Sk_Gen(G,PQ,flag,T,R)
    M = length(flag);
    L = sum(flag);
    sk_a = zeros(M*R,1);
    sk_w = zeros(L*R,1);
    T1 = floor(T/2);
    h = 1;
    k = 1;
    for i = 1:M
        
        if flag(i) == 1
            source = find(G(:,h)==1);
            ind = randperm(length(source));
            for j = 1:R
                sk_a((i-1)*R+j) = source(ind(j));  
%                 sk_a((Ain-1)*R+j) = source(ind(j));
                sk_w((k-1)*R+j) = source(ind(j));
            end
            if k < L
                k = k + 1;
                if h < L
                    h = h + 1;
                else
                    h = ceil(rand()*L);
                    fprintf("jkjfskjfskg\n");
                end
            end
        else
            sk_a((i-1)*R+1:i*R) = randperm(T1,R);

        end
        Ain = find(PQ(:,1)==PQ(i,1)&PQ(:,2)==(-PQ(i,2)));
        for j = 1:R
            sk_a((Ain-1)*R+j) = sk_a((i-1)*R+j);
        end
%         fprintf("Ain:%d n:%d m:%d -m:%d sk_a:%d\n",Ain,PQ(i,1),PQ(i,2),-PQ(i,2),sk_a((i-1)*R+1)+sk_a((i-1)*R+2)+sk_a((i-1)*R+3));
    end
end

