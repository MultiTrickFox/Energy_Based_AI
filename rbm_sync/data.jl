using Knet: dir ; include(dir("data","mnist.jl"))

nist = mnist()

data_train = []
data_dev  = []

in_size = 28*28


to_label(int_val) =
begin
    lbl = zeros(1, 10)
    lbl[int_val] += 1
lbl
end

for i in 1:size(nist[1])[end]
    push!(data_train, (reshape(nist[1][:,:,:,i], 1, in_size),to_label(nist[2][i])))
end ;

for i in 1:size(nist[3])[end]
    push!(data_dev, (reshape(nist[3][:,:,:,i], 1, in_size),to_label(nist[4][i])))
end ;


data_train = [e[1] for e in data_train]
data_dev = [e[1] for e in data_dev]


fn_01_to_minus1plus1 = x->(x.*2).-1

data_train = fn_01_to_minus1plus1.(data_train)
data_dev = fn_01_to_minus1plus1.(data_dev)


# try

    if binary
        data_train = [binarize_data.(e) for e in data_train]
        data_dev = [binarize_data.(e) for e in data_dev]
    end

# catch
#
#     println("Warning: data.jl imported before rbm.jl")
#
# end
