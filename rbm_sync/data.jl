using Knet: dir ; include(dir("data","mnist.jl"))

nist = mnist() ; binary = false


data_train = []
data_dev  = []

data_train2 = [[] for _ in 1:10]
data_dev2 = [[] for _ in 1:10]


in_size = 28*28


to_label(int_val) =
begin
    lbl = zeros(1, 10)
    lbl[int_val] += 1
lbl
end

fn_01_to_minus1plus1 = x->(x.*2).-1


for i in 1:size(nist[1])[end]
    sample = reshape(nist[1][:,:,:,i], 1, in_size)
    sample = fn_01_to_minus1plus1(sample)
    binary ? sample = binarize_data(sample) : ()
    push!(data_train, sample)
    push!(data_train2[nist[2][i]], sample)
    # push!(data_train, (reshape(nist[1][:,:,:,i], 1, in_size),to_label(nist[2][i])))
end ;

for i in 1:size(nist[3])[end]
    sample = reshape(nist[3][:,:,:,i], 1, in_size)
    sample = fn_01_to_minus1plus1(sample)
    binary ? sample = binarize_data(sample) : ()
    push!(data_dev, sample)
    push!(data_dev2[nist[4][i]], sample)
    # push!(data_dev, (reshape(nist[3][:,:,:,i], 1, in_size),to_label(nist[4][i])))
end ;


##
