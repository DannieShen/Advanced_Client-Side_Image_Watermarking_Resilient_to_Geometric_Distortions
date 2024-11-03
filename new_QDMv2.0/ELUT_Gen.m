function [ ELUT ] = ELUT_Gen( sigma_E, T )


%mu_E = round(sigma_E * rand(T, 1) + sigma_E);
ELUT = round(sigma_E .* randn(T, 1));

end

