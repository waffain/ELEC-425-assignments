function gradient = activation_tanh_gradient(y)
    gradient = 1 - y.^2;
end