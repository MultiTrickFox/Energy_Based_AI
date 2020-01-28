using Knet: dir ; include(dir("data","mnist.jl"))

nist = mnist()

data_train = []
data_dev  = []

out_size = 784


to_label(int_val) =
begin
    lbl = zeros(1, out_size)
    lbl[int_val] += 1
lbl
end

for i in 1:size(nist[1])[end]
    push!(data_train, (reshape(nist[1][:,:,:,i], 1, out_size),to_label(nist[2][i])))
end ;

for i in 1:size(nist[3])[end]
    push!(data_dev, (reshape(nist[3][:,:,:,i], 1, out_size),to_label(nist[4][i])))
end ;

data_train = [e[1] for e in data_train]
data_dev = [e[1] for e in data_dev]


step_fn(x) =
    if x > 0
        1
    else
        0
    end

data_train = [step_fn.(e) for e in data_train]
data_dev = [step_fn.(e) for e in data_dev]



# data_train = collect(zip(xtrn, ytrn))
# data_dev = collect(zip(xtst, ytst))

# data_train = xtrn
# data_test = xtst


# using Knet,GZip,Compat
#
#
# function loaddata()
#     xtrn = gzload("train-images-idx3-ubyte.gz")[17:end]
#     xtst = gzload("t10k-images-idx3-ubyte.gz")[17:end]
#     ytrn = gzload("train-labels-idx1-ubyte.gz")[9:end]
#     ytst = gzload("t10k-labels-idx1-ubyte.gz")[9:end]
#     xtrn,ytrn,xtst,ytst
# end
#
# function gzload(file; path=Knet.dir("data",file), url="http://yann.lecun.com/exdb/mnist/$file")
#     isfile(path) || download(url, path)
#     f = gzopen(path)
#     a = @compat read(f)
#     close(f)
#     return(a)
# end
#
#
# data = loaddata()
#
# data_train = data[1]
# data_dev = data[3]
#
#
# @show typeof(data_train[1])
