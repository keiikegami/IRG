# set model type
mutable struct IRG
    alpha::Array{Float64,1}
    beta::Array{Float64,1}
    gamma::Array{Float64,1}
    sigma::Array{Float64, 1}
    psi::Float64
    delta::Float64
end

# functions in this model
# vector valued functions
function u(model::IRG, s::Array{Float64, 2}, x::Array{Float64, 2}, p::Int64)
    return ((s[:, p] - x[:, p]).^(1 - model.alpha[p]))./(1 - model.alpha[p])
end

function h(model::IRG, x::Array{Float64, 2}, epsilon::Float64, p::Int64)
    return model.gamma[p] * x[:, p] + epsilon * (x[:, p].^(model.beta[p]))
end

function s(model::IRG, x::Array{Float64, 2}, epsilon::Array{Float64, 1})
    box = zeros((size(x)[1], 2))
    box[:, 1] = (1 - model.psi)*h(model, x, epsilon[1], 1) + model.psi*h(model, x, epsilon[2], 2)
    box[:, 2] = (1 - model.psi)*h(model, x, epsilon[2], 2) + model.psi*h(model, x, epsilon[1], 1)
    return box
end

# derivatives of  payoff, transition functions
# utility function
function ux(model::IRG, s::Array{Float64, 2}, x::Array{Float64, 2}, p::Int64)
    return -(s[:, p] - x[:, p]).^(-model.alpha[p])
end

function uxx(model::IRG, s::Array{Float64, 2}, x::Array{Float64, 2}, p::Int64)
    return -model.alpha[p]*(s[:, p] - x[:, p]).^(-model.alpha[p]-1)
end

# transition function
function sx(model::IRG, x::Array{Float64, 2}, epsilon::Array{Float64, 1},  p::Int64)
    return (1 - model.psi)*(model.gamma[p] + epsilon[p]*model.beta[p]*x[:, p].^(model.beta[p] - 1))
end

function sxx(model::IRG, x::Array{Float64, 2}, epsilon::Array{Float64, 1},  p::Int64)
    return (1 - model.psi)*epsilon[p]*model.beta[p]*(model.beta[p] - 1)*x[:, p].^(model.beta[p] - 2)
end

# Gaussian Quadrature
function gq(model::IRG, num_nodes::Int64)
    return qnwlogn([num_nodes, num_nodes], [0,0], diagm([sigma^2, sigma^2]))
end

# collocation function
function vmax(model::IRG, colnodes::Array{Float64, 2}, action::Array{Float64, 2}, coef::Array{Float64, 2}, epss::Array{Float64, 2}, weights::Array{Float64, 1})
    xnew = action
    v = zeros((size(colnodes)[1], 2))
    for p in 1:2
        xl, xu = 0.0, colnodes[:, p]
        order1 = [0 0]
        order1[1, p] = 1
        order2 = [0 0]
        order2[1, p] = 2
        for it in 1:maxit
            util, util_der1, util_der2 = u(model, colnodes, action, p), ux(model, colnodes, action, p), uxx(model, colnodes, action, p) 
            # println(util[1])
            # println(util_der1[1])
            # println(util_der2[1])
            println(action)
            Ev, Evx, Evxx = 0.0, 0.0, 0.0
            for k in 1:num_nodes^2
                println(k)
                eps, weight= epss[k, :], weights[k]
                transition, transition_der1, transition_der2 = s(model, action, eps), sx(model, action, eps, p), sxx(model, action, eps, p)
                #println(transition)
                #println(transition_der1)
                #println(transition_der2)
                # coefによって結果が変わらないんだけど、係数はここにしか出てないからこいつがやっぱり怪しいよな
                # 係数は別に良いっぽいダメなのはfuneval？
                vn = funeval(coef[:, p], basis, transition)
                vnder1 =  funeval(coef[:, p], basis, transition, order1)
                vnder2 = funeval(coef[:, p], basis, transition, order2)
                # vnが係数と一致してるんだけどなにこれ
                #println(vn)
                #println(vnder1)
                #println(vnder2)
                Ev += weight * vn
                Evx += weight* vnder1.* transition_der1
                Evxx += weight * (vnder1.*transition_der2 + vnder2 .* (transition_der1.^2))
            end
            # println(Ev[1])
            # println(Evx[1])
            # println(Evxx[1])
            v[:, p] = util + Ev
            delx = -(util_der1 + model.delta * Evx) ./ (util_der2 + model.delta*Evxx)
            # delxがでかすぎる
            #println(delx)
            # println(xl-action[:, p])
            # println(xu-action[:, p])
            delx = min.(max.(delx, xl-action[:, p]), xu-action[:, p])
            action[:, p] = action[:, p] + delx
            if norm(delx) < tol
                break
            end
        end
        xnew[:, p] = action[:, p]
    end
    return v, xnew
end