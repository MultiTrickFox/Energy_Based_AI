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

    findall(x->x==e,list)[1]


indicesof(e, list) =

    findall(x->x==e,list)


##


len(arr) = length(arr)


int(nr) = convert(Int32,floor(nr))

float(nr) = convert(Float32,nr)

str(nr) = "$(nr)"


print(str) = println(str)


##


using Random: shuffle, shuffle!


choice(arr) = arr[randn(1:length(arr))]

using Random: randperm

choices(arr,n) = arr[randperm(length(arr))[1:n]]


##
