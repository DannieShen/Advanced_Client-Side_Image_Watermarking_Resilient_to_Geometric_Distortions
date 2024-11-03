% function [Bm] = Bm_Gen(len_skm,T,R,sk_w)
%     Bm = zeros(len_skm,T-floor(T/2));
%     for i = 1:len_skm
%         for j = 1:R
%             Bm(i,sk_w((i-1)*R+j)-floor(T/2)) = 1;
%         end
%     end
%     Bm = sparse(Bm);
%     
% end

function [Bm] = Bm_Gen(len_skm,T,R,sk_w)
    Bm = zeros(len_skm,T);
    for i = 1:len_skm
        for j = 1:R
            Bm(i,sk_w((i-1)*R+j)) = 1;
        end
    end
    Bm = sparse(Bm);
end

