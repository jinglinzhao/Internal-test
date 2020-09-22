# In[1]:


using Pkg
Pkg.activate(".")
# Pkg.add(["DataFrames","CSV","FITSIO","Distributions","MultivariateStats","PDMats",Plots","Optim","Interpolations"])
# Pkg.add(["SpecialFunctions", "NLsolve", "QuadGK"])
# Pkg.instantiate()
using CSV
#using FITSIO
using Statistics
using MultivariateStats
using PDMats
using Plots
using Optim
using Interpolations
using LinearAlgebra


# In[105]:


function window_function(x::Real)
    exp(-0.5*x^2)/sqrt(2π)
end

function basis_n(x::Real, n::Integer; sinfactor::Real = one(x))
    if n==0         return window_function(x) end
    if mod(n,2)==1  return sin(((n+1)/2)*sinfactor*x) * window_function(x) end
    if mod(n,2)==0  return cos((n/2)*sinfactor*x)     * window_function(x) end
end


# In[106]:


x_limit = 3
num_samples = 100
x = range(-x_limit,stop=x_limit,length=num_samples)


# In[107]:


plt1 = plot(x,basis_n.(x,0), label="0")
plt2 = plot(x,basis_n.(x,1), label="1")
plot!(plt1,x,basis_n.(x,2), label="2")
plot!(plt2,x,basis_n.(x,3), label="3")
plot!(plt1,x,basis_n.(x,4), label="4")
plot!(plt2,x,basis_n.(x,5), label="5")
plot!(plt1,x,basis_n.(x,6), label="6")
plot!(plt2,x,basis_n.(x,7), label="7")
plot!(plt1,x,basis_n.(x,8), label="8")
#plot!(plt2,x,basis_n.(x,9), label="9")
#plot!(plt1,x,basis_n.(x,10), label="10")
#plot!(plt2,x,basis_n.(x,0), label="0")
plot(plt1,plt2, layout=2)


# In[110]:


sf = 2
plt1 = plot(x,basis_n.(x,0,sinfactor=sf), label="0")
plt2 = plot(x,basis_n.(x,1,sinfactor=sf), label="1")
plot!(plt1,x,basis_n.(x,2,sinfactor=sf), label="2")
plot!(plt2,x,basis_n.(x,3,sinfactor=sf), label="3")
plot!(plt1,x,basis_n.(x,4,sinfactor=sf), label="4")
plot!(plt2,x,basis_n.(x,5,sinfactor=sf), label="5")
plot!(plt1,x,basis_n.(x,6,sinfactor=sf), label="6")
plot!(plt2,x,basis_n.(x,7,sinfactor=sf), label="7")
plot!(plt1,x,basis_n.(x,8,sinfactor=sf), label="8")
#plot!(plt2,x,basis_n.(x,9), label="9")
#plot!(plt1,x,basis_n.(x,10), label="10")
#plot!(plt2,x,basis_n.(x,0), label="0")
plot(plt1,plt2, layout=2)


# In[243]:


function make_basis(x::AbstractArray{T}; x_center = zero(T), x_scale = one(T), degree_poly::Integer = 0, degree_line::Integer = 6) where T<:Real
    num_bases = degree_line
    num_samples = length(x)
    @assert num_bases >= 1
    @assert num_samples > num_bases

    x = (x.- x_center)./x_scale
    b = zeros(num_samples,num_bases)
    #balt = zeros(num_samples,num_bases)
    norm = zeros(num_bases)
    proj = zeros(num_samples)
    b[:,1] .= 1
    #balt[:,1] .= 1
    norm[1] = sum(b[:,1].^2)
    for i in 2:num_bases
        #balt[:,i] .= basis_n.(x,i-2)
        b[:,i] .= basis_n.(x,i-2,sinfactor=2)
        norm[i] = sum(b[:,i].^2)
        if i>1
            for j = 1:(i-1)
                proj .= ((b[:,i].*b[:,j])/norm[j]) .* b[:,j]
                b[:,i] .-= proj
            end
        end
    end
    return b # (b,balt)
end


# In[292]:


x_limit = 4
x = range(-x_limit,stop=x_limit,length=100)
b = make_basis(x, degree_line=10)


# In[293]:


plot(b)


# In[294]:


function eval_lsf(x::Real; a::AbstractArray, x_center = zero(T), x_scale = one(T), degree_poly::Integer = 0, degree_line::Integer = 0) where T<:Real
    @assert -10 <= x_center <= 10
    @assert 0 < x_scale < 100
    @assert 0 <= degree_poly <= 2
    @assert 0 <= degree_line <= 8
    x = (x - x_center) / x_scale
    basis_comb = zeros(Float64,2+degree_poly+degree_line)
    basis_comb[1] = one(x)
    if degree_poly >= 1
        for i in 2:degree_poly+1
            basis_comb[i] = x.^(i-1)
        end
    end
    for i in (2+degree_poly):(2+degree_poly+degree_line)
        basis_comb[i] = basis_n(x,i-2-degree_poly)
    end
    return basis_comb' * a
end


# In[295]:


function fit_and_predict(θ; x::AbstractArray, yobs::AbstractArray, sigmaobs::AbstractArray, degree_poly::Integer = 0, degree_line::Integer = 1)
    x_center = θ[1]
    x_scale = abs(θ[2])
    #=
    @assert -3 <= x_center <= 3
    @assert 0 < x_scale < 100
    @assert 0 <= degree_poly <= 2
    @assert 0 <= degree_line <= 8
    x = (x .- x_center) ./ x_scale
    basis = zeros(Float64,length(x),2+degree_poly+degree_line)
    basis[:,1] .= ones(length(x))
    if degree_poly >= 1
        for i in 2:degree_poly+1
            basis[:,i] .= ((x.-x[1])./(x[end]-x[1]).-0.5).^(i-1)
        end
    end
    for i in (2+degree_poly):(2+degree_poly+degree_line)
        basis[:,i] .= hermite_n.(x,i-2-degree_poly)
    end
    =#
    basis = make_basis(x, x_center=x_center, x_scale=x_scale, degree_poly=degree_poly, degree_line=degree_line)
    #a = llsq(basis, yobs; bias=false)
    X = basis
    #a = (X' * X) \ (X' * yobs)  # unweighted

    Winv = PDiagMat(sigmaobs.^2)
    #println("eigs = ", eigvals(Xt_invA_X(Winv, X)))
    a = Xt_invA_X(Winv, X)  \ (X' * (inv(Winv) * yobs) )
    pred = ( basis * a )
    return pred,a
end


# In[296]:


function predict(θ; a::AbstractArray, x::AbstractArray, degree_poly::Integer = 0, degree_line::Integer = 0)
    x_center = θ[1]
    x_scale = abs(θ[2])
    basis = make_basis(x, x_center=x_center, x_scale=x_scale, degree_poly=degree_poly, degree_line=degree_line)
    pred = (basis * a )
    return pred
end


# In[297]:


function calc_rms_error(ypred,yobs)
    rmse = sqrt(Statistics.mean(abs2.(ypred .- yobs)))
end
function calc_chi_sq(ypred,yobs,sigmaobs)
    sum(abs.( (ypred .- yobs).^2 ./ sigmaobs.^2 ))
end


# In[298]:


function find_cols_to_fit(wavelengths::AbstractArray{T,1}, line_center::Real; Δ::Real = 0.3) where T<:Real
    findfirst(x->x>=line_center-Δ,wavelengths):findlast(x->x<=line_center+Δ,wavelengths)
end

function find_cols_to_fit(wavelengths::AbstractArray{T,1}, line_lo::Real, line_hi::Real) where T<:Real
    @assert line_lo < line_hi
    findfirst(x->x>=line_lo,wavelengths):findlast(x->x<=line_hi,wavelengths)
end


# In[299]:


function fit_line_profile(xobs,yobs,sigmaobs; θ_init, width_default::Real = zero(eltype(θ_init)), max_degree_line::Integer=4, degree_poly::Integer=1, make_plot::Bool = true)
    @assert 0 <= max_degree_line <= 8
    @assert 0 <= degree_poly <= 2
    @assert size(xobs) == size(yobs) == size(sigmaobs)
    @assert 1 <= length(θ_init) <= 2
    initial_θ = θ_init
    local ypred, coeff, res
    if make_plot plt = scatter(xobs,yobs,label="Obs",legend=:bottomright) end
    #plt = scatter(xobs,yobs,label="Obs",legend=:topright)
    width_init = width_default == zero(width_default) ? θ_init[2] : width_default
    deg_line = 0
    function helper_fixed_width(θ)
        θtmp = [ θ[1], width_init ]
        ypred, coeff = fit_and_predict(θtmp, x=xobs, yobs=yobs,sigmaobs=sigmaobs,degree_line=deg_line,degree_poly=degree_poly)
        calc_chi_sq(ypred,yobs,sigmaobs)
    end
    res = optimize(helper_fixed_width, [initial_θ[1] ], Newton())
    #res = optimize(helper, initial_θ)
    println("Degree = 0 (fixed width): χ^2 = ", res.minimum, "  RMS = ",calc_rms_error(ypred,yobs) )
    initial_θ = vcat(res.minimizer, [width_init])
    (ypred, coeff) = fit_and_predict(initial_θ, x=xobs, yobs=yobs,sigmaobs=sigmaobs, degree_line=deg_line,degree_poly=degree_poly)
    println("   arg = ", res.minimizer, " a = ", coeff)
    if make_plot
        plot!(plt,xobs,ypred,label="Fixed width")
    end
    for deg_line in 0:max_degree_line
        function helper(θ)
            ypred, coeff = fit_and_predict(θ, x=xobs, yobs=yobs,sigmaobs=sigmaobs,degree_line=deg_line,degree_poly=degree_poly)
            calc_chi_sq(ypred,yobs,sigmaobs)
        end

        res = optimize(helper, initial_θ)
        println("Degree = ", deg_line, ": χ^2 = ", res.minimum, "  RMS = ",calc_rms_error(ypred,yobs) )
        initial_θ = res.minimizer
        (ypred, coeff) = fit_and_predict(initial_θ, x=xobs, yobs=yobs,sigmaobs=sigmaobs, degree_line=deg_line,degree_poly=degree_poly)
        println("   arg = ", res.minimizer, " a = ", coeff)
        if make_plot
            plot!(plt,xobs,ypred,label=string(deg_line))
        end
    end
    if make_plot
        flush(stdout)
        display(plt)
    end
    return (res.minimizer, coeff, ypred)
end


# In[300]:


using Interpolations

AF = Real # AbstractFloat
AA = AbstractArray

function measure_bisector(xs::AA{T,1}, ys::AA{T,1}; interpolate::Bool=true,
                          top::T=0.99, len::Integer=100) where T<:AF
    if interpolate
        return measure_bisector_interpolate(xs, ys, top=top, len=len)
    else
        return measure_bisector_loop(xs, ys, top=top, len=len)
    end
end

function measure_bisector_interpolate(xs::AA{T,1}, ys::AA{T,1}; top::T=0.99,
                                      len::Integer=100) where T<:AF
    # check lengths and normalization
    @assert length(xs) == length(ys)

    # normalize the spec, find bottom of line
    ys ./= maximum(ys)
    botind = argmin(ys)
    depths = range(ys[botind], top, length=len)

    # find left and right halves
    lind = findfirst(ys .< top)
    rind = findlast(ys .< top)
    lspec = reverse(ys[lind:botind])
    rspec = ys[botind:rind]
    lwav = reverse(xs[lind:botind])
    rwav = xs[botind:rind]

    # interpolate wavelengths onto intensity grid
    lspline = LinearInterpolation(lspec, lwav, extrapolation_bc=Flat())
    rspline = LinearInterpolation(rspec, rwav, extrapolation_bc=Flat())
    w1 = lspline(depths)
    w2 = rspline(depths)
    wavs = (lspline(depths) .+ rspline(depths)) ./ 2.0
    return wavs, depths
end

function measure_bisector_loop(xs::AA{T,1}, ys::AA{T,1}; top::T=0.99,
                               len::Integer=100) where T<:AF
    # normalize the spec, find bottom of line
    ys ./= maximum(ys)

    # assign depths to measure bisector at
    dep = range(one(T)-minimum(ys)-0.01, one(T) - top, length=len)

    # set iterators
    nccf = Int(length(xs) ÷ 2)
    L = nccf
    R = nccf

    # allocate memory
    xL = zeros(len)
    xR = zeros(len)
    wav = zeros(len)

    # loop over depths
    for d in eachindex(dep)
        y = 1 - dep[d]
        while((ys[L] < y) & (L > 0))
            L -= 1
        end

        while ((ys[R] < y) & (R < length(xs)))
            R += 1
        end

        if ((y > maximum(ys[1:nccf])) | (y > maximum(ys[nccf+1:end])))
            L = 0
            R = length(xs)
        end

        if L == 0
            xL[d] = xL[d-1]
        else
            mL = (xs[L+1] - xs[L]) / (ys[L+1] - ys[L])
            xL[d] = xs[L] + mL * (y - ys[L])
        end

        if R == length(xs)
            xR[d] = xR[d-1]
        else
            mR = (xs[R-1] - xs[R]) / (ys[R-1] - ys[R])
            xR[d] = xs[R] + mR * (y - ys[R])
        end
        wav[d] = (xL[d] + xR[d]) / 2.0
    end
    return wav, one(T) .- dep
end


# In[51]:


using CSV
df = CSV.read("../feI5434_spectra_bisectors/spectra_for_eric.csv")
sol_flux, sol_var, sol_wave = df[:,2], 0.001.*df[:,2], df[:,1]


# In[373]:


line_center = 5434.5
delta_lambda_fit = 0.35
idx_cols = find_cols_to_fit(sol_wave,line_center,Δ=delta_lambda_fit)


# In[374]:


using Interpolations

AF = Real # AbstractArray
AA = AbstractArray

function measure_bisector(xs::AA{T,1}, ys::AA{T,1}; interpolate::Bool=true,
                          top::T=0.99, len::Integer=100) where T<:AF
    if interpolate
        return measure_bisector_interpolate(xs, ys, top=top, len=len)
    else
        return measure_bisector_loop(xs, ys, top=top, len=len)
    end
end

function measure_bisector_interpolate(xs::AA{T,1}, ys::AA{T,1}; top::T=0.99,
                                      len::Integer=100) where T<:AF
    # check lengths and normalization
    @assert length(xs) == length(ys)

    # normalize the spec, find bottom of line
    ys ./= maximum(ys)
    botind = argmin(ys)
    depths = range(ys[botind], top, length=len)

    # find left and right halves
    lind = findfirst(ys .< top)
    rind = findlast(ys .< top)
    lspec = reverse(ys[lind:botind])
    rspec = ys[botind:rind]
    lwav = reverse(xs[lind:botind])
    rwav = xs[botind:rind]

    # interpolate wavelengths onto intensity grid
    lspline = LinearInterpolation(lspec, lwav, extrapolation_bc=Flat())
    rspline = LinearInterpolation(rspec, rwav, extrapolation_bc=Flat())
    w1 = lspline(depths)
    w2 = rspline(depths)
    wavs = (lspline(depths) .+ rspline(depths)) ./ 2.0
    return wavs, depths
end

function measure_bisector_loop(xs::AA{T,1}, ys::AA{T,1}; top::T=0.99,
                               len::Integer=100) where T<:AF
    # normalize the spec, find bottom of line
    ys ./= maximum(ys)

    # assign depths to measure bisector at
    dep = range(one(T)-minimum(ys)-0.01, one(T) - top, length=len)

    # set iterators
    nccf = Int(length(xs) ÷ 2)
    L = nccf
    R = nccf

    # allocate memory
    xL = zeros(len)
    xR = zeros(len)
    wav = zeros(len)

    # loop over depths
    for d in eachindex(dep)
        y = 1 - dep[d]
        while((ys[L] < y) & (L > 0))
            L -= 1
        end

        while ((ys[R] < y) & (R < length(xs)))
            R += 1
        end

        if ((y > maximum(ys[1:nccf])) | (y > maximum(ys[nccf+1:end])))
            L = 0
            R = length(xs)
        end

        if L == 0
            xL[d] = xL[d-1]
        else
            mL = (xs[L+1] - xs[L]) / (ys[L+1] - ys[L])
            xL[d] = xs[L] + mL * (y - ys[L])
        end

        if R == length(xs)
            xR[d] = xR[d-1]
        else
            mR = (xs[R-1] - xs[R]) / (ys[R-1] - ys[R])
            xR[d] = xs[R] + mR * (y - ys[R])
        end
        wav[d] = (xL[d] + xR[d]) / 2.0
    end
    return wav, one(T) .- dep
end


# In[375]:


θ_init = [-0.02, 0.1]
xobs = sol_wave[idx_cols] .- line_center
yobs = convert(Array{Float64,1},sol_flux[idx_cols] )
sigmaobs = convert(Array{Float64,1},sol_var[idx_cols] )
(ypred, coeff) = fit_and_predict(θ_init, x=xobs, yobs=yobs, sigmaobs=sigmaobs, degree_poly=0, degree_line=14)
println("RMS = ",calc_rms_error(ypred,yobs), " χ^2 = ", calc_chi_sq(ypred,yobs,sigmaobs), " dof = ", length(xobs))
println("coeff = ", coeff)
scatter(xobs,yobs,legend=:none)
plot!(xobs,ypred,legend=:none)


# In[452]:


function make_basis(x::AbstractArray{T}; x_center = zero(T), x_scale = one(T), degree_poly::Integer = 0, degree_line::Integer = 6) where T<:Real
    num_bases = degree_line
    num_samples = length(x)
    @assert num_bases >= 1
    @assert num_samples > num_bases

    x = (x.- x_center)./x_scale
    b = zeros(num_samples,num_bases)
    #balt = zeros(num_samples,num_bases)
    norm = zeros(num_bases)
    proj = zeros(num_samples)
    b[:,1] .= 1
    #balt[:,1] .= 1
    norm[1] = b[:,1]'*b[:,1]
    b[:,1] ./= sqrt(norm[1])
    for i in 2:num_bases
        #balt[:,i] .= basis_n.(x,i-2)
        b[:,i] .= basis_n.(x,i-2,sinfactor=1.32)
        if i>1
            for j = 1:(i-1)
                proj .= ((b[:,i]'*b[:,j])/norm[j]) .* b[:,j]
                b[:,i] .-= proj
                norm[i] = b[:,i]'*b[:,i]
            end
        end
        norm[i] = b[:,i]'*b[:,i]
        b[:,i] ./= sqrt(norm[i]) # b[:,i]'*b[:,i])
    end
    return b # (b,balt)
end


# In[457]:


(ypred, coeff) = fit_and_predict(θ_init, x=xobs, yobs=yobs, sigmaobs=sigmaobs, degree_poly=0, degree_line=6)
println("RMS = ",calc_rms_error(ypred,yobs), " χ^2 = ", calc_chi_sq(ypred,yobs,sigmaobs), " dof = ", length(xobs))
println("coeff = ", coeff)
(ypred, coeff) = fit_and_predict(θ_init, x=xobs, yobs=yobs, sigmaobs=sigmaobs, degree_poly=0, degree_line=8)
println("RMS = ",calc_rms_error(ypred,yobs), " χ^2 = ", calc_chi_sq(ypred,yobs,sigmaobs), " dof = ", length(xobs))
println("coeff = ", coeff)
(ypred, coeff) = fit_and_predict(θ_init, x=xobs, yobs=yobs, sigmaobs=sigmaobs, degree_poly=0, degree_line=10)
println("RMS = ",calc_rms_error(ypred,yobs), " χ^2 = ", calc_chi_sq(ypred,yobs,sigmaobs), " dof = ", length(xobs))
println("coeff = ", coeff)
(ypred, coeff) = fit_and_predict(θ_init, x=xobs, yobs=yobs, sigmaobs=sigmaobs, degree_poly=0, degree_line=12)
println("RMS = ",calc_rms_error(ypred,yobs), " χ^2 = ", calc_chi_sq(ypred,yobs,sigmaobs), " dof = ", length(xobs))
println("coeff = ", coeff)
#=
(ypred, coeff) = fit_and_predict(θ_init, x=xobs, yobs=yobs, sigmaobs=sigmaobs, degree_poly=0, degree_line=14)
println("RMS = ",calc_rms_error(ypred,yobs), " χ^2 = ", calc_chi_sq(ypred,yobs,sigmaobs), " dof = ", length(xobs))
println("coeff = ", coeff)
(ypred, coeff) = fit_and_predict(θ_init, x=xobs, yobs=yobs, sigmaobs=sigmaobs, degree_poly=0, degree_line=10)
println("RMS = ",calc_rms_error(ypred,yobs), " χ^2 = ", calc_chi_sq(ypred,yobs,sigmaobs), " dof = ", length(xobs))
println("coeff = ", coeff)

(ypred, coeff) = fit_and_predict(θ_init, x=xobs, yobs=yobs, sigmaobs=sigmaobs, degree_poly=0, degree_line=18)
println("RMS = ",calc_rms_error(ypred,yobs), " χ^2 = ", calc_chi_sq(ypred,yobs,sigmaobs), " dof = ", length(xobs))
println("coeff = ", coeff)
(ypred, coeff) = fit_and_predict(θ_init, x=xobs, yobs=yobs, sigmaobs=sigmaobs, degree_poly=0, degree_line=20)
println("RMS = ",calc_rms_error(ypred,yobs), " χ^2 = ", calc_chi_sq(ypred,yobs,sigmaobs), " dof = ", length(xobs))
println("coeff = ", coeff)
=#
plt1 = scatter(xobs,yobs,legend=:none)
plot!(plt1, xobs,ypred,legend=:none)
plt2 = scatter(xobs, yobs.-ypred,legend=:none)
plot(plt1,plt2,layout=(2,1))


# In[458]:


i = 2
bis_obs = measure_bisector(xobs,df[idx_cols,i] , interpolate=false, top=0.98)
(ypred, coeff) = fit_and_predict(θ_init, x=xobs, yobs=yobs, sigmaobs=sigmaobs, degree_poly=1, degree_line=length(coeff)-2)
#yobs_recon = predict(coeff[1:2], a=coeff[3:end], x=xobs, degree_line=length(coeff)-2)
bis_fit = measure_bisector(xobs,ypred , interpolate=false, top=0.98)
scatter(bis_obs[1],bis_obs[2], ms=1, label="Observed" * string(i), legend=:none)
plot!(bis_fit[1],bis_fit[2], label="Reconstructed")


# In[451]:


#=
i = 20
bis_obs = measure_bisector(xobs,df[idx_cols,i] , interpolate=false, top=0.98)
yobs_recon = predict(Xr[1:2,i], a=Xr[3:end,i], x=xobs, degree_line=max_deg)
bis_fit = measure_bisector(xobs,yobs_recon , interpolate=false, top=0.96)
scatter!(bis_obs[1],bis_obs[2], ms=1, label="Observed " * string(i))
plot!(bis_fit[1],bis_fit[2], label="Reconstructed "* string(i))
i = 40
bis_obs = measure_bisector(xobs,df[idx_cols,i] , interpolate=false, top=0.98)
yobs_recon = predict(Xr[1:2,i], a=Xr[3:end,i], x=xobs, degree_line=max_deg)
bis_fit = measure_bisector(xobs,yobs_recon , interpolate=false, top=0.96)
scatter!(bis_obs[1],bis_obs[2], ms=1, label="Observed " * string(i))
plot!(bis_fit[1],bis_fit[2], label="Reconstructed "* string(i))
i = 60
bis_obs = measure_bisector(xobs,df[idx_cols,i] , interpolate=false, top=0.98)
yobs_recon = predict(Xr[1:2,i], a=Xr[3:end,i], x=xobs, degree_line=max_deg)
#bis_fit4 = measure_bisector(xobs,yobs_recon , interpolate=false, top=0.98)
scatter!(bis_obs[1],bis_obs[2], ms=1, label="Observed " * string(i))
plot!(bis_fit4[1],bis_fit4[2], label="Reconstructed "* string(i))
i = 80
bis_obs = measure_bisector(xobs,df[idx_cols,i] , interpolate=false, top=0.98)
yobs_recon = predict(Xr[1:2,i], a=Xr[3:end,i], x=xobs, degree_line=max_deg)
#bis_fit5 = measure_bisector(xobs,yobs_recon , interpolate=false, top=0.98)
scatter!(bis_obs[1],bis_obs[2], ms=1, label="Observed " * string(i))
plot!(bis_fit5[1],bis_fit5[2], label="Reconstructed "* string(i))
i = 100
bis_obs = measure_bisector(xobs,df[idx_cols,i] , interpolate=false, top=0.98)
yobs_recon = predict(Xr[1:2,i], a=Xr[3:end,i], x=xobs, degree_line=max_deg)
#bis_fit6 = measure_bisector(xobs,yobs_recon , interpolate=false, top=0.98)
scatter!(bis_obs[1],bis_obs[2], ms=1, label="Observed " * string(i))
plot!(bis_fit6[1],bis_fit6[2], label="Reconstructed "* string(i))
=#


# In[ ]:
