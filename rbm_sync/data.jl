in_size = 28 * 28


std_transform = true


##


using Knet: dir ; include(dir("data","mnist.jl"))

m_nist = mnist()


samples1 = []
for i in 1:size(m_nist[1])[end]
    sample = reshape(m_nist[1][:,:,:,i], 1, in_size)
    push!(samples1, sample)
end ;
samples1_ = vcat(samples1...)

samples2 = []
for i in 1:size(m_nist[3])[end]
    sample = reshape(m_nist[3][:,:,:,i], 1, in_size)
    push!(samples2, sample)
end ;
samples2_ = vcat(samples2...)

nist = [
        reshape(samples1_, size(m_nist[1])[end], in_size),
        m_nist[2],
        reshape(samples2_, size(m_nist[3])[end], in_size),
        m_nist[4],
       ]


using StatsBase: standardize, fit, transform, reconstruct!
using StatsBase: ZScoreTransform, UnitRangeTransform

# nist[1] = standardize(ZScoreTransform,nist[1],dims=1)
# nist[3] = standardize(ZScoreTransform,nist[3],dims=1)
#
# nist[1] = standardize(UnitRangeTransform,nist[1],dims=1)
# nist[3] = standardize(UnitRangeTransform,nist[3],dims=1)

std_transform ? (begin

    std1 = fit(ZScoreTransform, vcat(nist[1],nist[3]),dims=1)
    nist[1] = transform(std1,nist[1])
    nist[3] = transform(std1,nist[3])

    @show nist[1]

    std2 = fit(UnitRangeTransform, vcat(nist[1],nist[3]),dims=1)
    nist[1] = transform(std2,nist[1])
    nist[3] = transform(std2,nist[3])

end) : ()


to_label(int_val) =
begin
    lbl = zeros(1, 10)
    lbl[int_val] += 1
lbl
end

fn_01_to_minus1plus1 = x->(x.*2).-1


data_train = []
data_dev  = []

data_train2 = [[] for _ in 1:10]
data_dev2 = [[] for _ in 1:10]


for i in 1:size(nist[1])[1]
    sample = reshape(nist[1][i,:], 1, in_size)
    sample = fn_01_to_minus1plus1(sample)
    binary ? sample = binarize_data(sample) : ()
    push!(data_train, sample)
    push!(data_train2[nist[2][i]], sample)
end ;

for i in 1:size(nist[3])[1]
    sample = reshape(nist[3][i,:], 1, in_size)
    sample = fn_01_to_minus1plus1(sample)
    binary ? sample = binarize_data(sample) : ()
    push!(data_dev, sample)
    push!(data_dev2[nist[4][i]], sample)
end ;


##


# function scale_features(X)
#     μ = mean(X, dims=1)
#     σ = std(X, dims=1)
#
#     X_norm = (X .- μ) ./ σ
#
#     return (X_norm, μ, σ)
# end
#
#
# # Scale the testing features using the learned parameters
# function transform_features(X, μ, σ)
#     X_norm = (X .- μ) ./ σ
#     return X_norm
# end
