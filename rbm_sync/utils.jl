# export JULIA_NUM_THREADS=‘4’

using Base.Threads: @threads, nthreads, threadid


# threadpool (generic function with 2 methods)


threadpool(fn_map, array) =
  begin
    array_return = Array[[] for _ in 1:nthreads()]
    @threads for e in array
        push!(array_return[threadid()], fn_map(e))
    end ; array_return = vcat(array_return...)

  array_return
  end


threadpool(fn_aggregate, fn_map, array) =
  begin
      array_return = Array[[] for _ in 1:nthreads()]
      @threads for e in array
          push!(array_return[threadid()], fn_map(e))
      end ; array_return = vcat(array_return...)
      value_return = fn_aggregate(array_return)

  value_return
  end


### ###


# length(procs()) < 2 ? addprocs(3) : ()

using Distributed: @everywhere, @distributed

using Distributed: procs, addprocs


# procpool (generic function with 2 methods)


procpool(fn_map, array) =

    @distributed (vcat) for e in array
        fn_map(e)
    end


procpool(fn_aggregate, fn_map, array) =

    fn_aggregate(
        begin @distributed (vcat) for e in array
                fn_map(e)
            end
        end
    )


### ###


batchify(list, batch_size) =

    [list[((i-1)*batch_size)+1:i*batch_size] for i in 1:convert(Int16,floor(length(list)/batch_size))]


##


indexof(e, list) =

    for (i,e2) in enumerate(list)

        e2 == e ? (return i) : ()

    end

indicesof(e, list) =

    findall(x->x==e,list)


##


len(arr) = length(arr)

resize(arr, size) = reshape(arr, size)


int(nr::AbstractFloat) = convert(Int32,floor(nr))
int(nr::AbstractString) = parse(Int32,nr)

float(nr::Integer) = convert(Float32,nr)
float(nr::AbstractString) = parse(Float32,nr)

str(nr) = "$(nr)"


##


using Random: shuffle, shuffle!

choice(arr) = arr[randn(1:length(arr))]

using Random: randperm

choices(arr,n;detailed=true) = detailed ? arr[randperm(length(arr))[1:n]] : shuffle(arr)[1:n]


enum(arr) = enumerate(arr)


type(e) = typeof(e)


##


input(prompt="") =
begin

    print(prompt)

chomp(readline())
end


##


@info "Utils read. \nthreads  : $(nthreads()) of $(length(Sys.cpu_info())) \nprocesses: $(length(procs())) of $(length(Sys.cpu_info()))"
