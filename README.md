# PaddlePaddle CUDAGraph简介与问题

## API的签名
注意：下面的所有API都不是线程安全的。在调用以下任何一个API的时候，请保证没有其他线程同时在执行以下操作：

- `executor.run`操作。
- 任何可能会申请显存的操作。
- 没有同时在capture多个CUDA Graph。

```Python
class CUDAGraph:
    def __init__(self, place=None, mode="thread_local"):
        # place需要传入paddle.CUDAPlace对象
        # mode支持"global", "thread_local"和"relaxed"三种
        pass

    def capture_begin(self):
        # 开始capture，底层调用cudaStreamBeginCapture，并准备对应的memory pool
        pass

    def capture_end(self):
        # 结束capture，底层调用cudaStreamEndCapture
        pass

    def replay(self):
        # 执行之前capture的CUDA Graph，底层调用cudaGraphLaunch
        pass

    def reset(self):
        # 析构之前capture的CUDA Graph，并释放memory pool
        pass
```


## 动态图的使用方法
```Python
import numpy as np
import paddle
from paddle.device.cuda.graphs import CUDAGraph

input = paddle.to_tensor(np.array([1], dtype='float32')) # 这句话必须在capture_start之前

graph = CUDAGraph()
graph.capture_start()
output = input + 10 # 在capture_start和capture_end之间执行一些GPU操作
graph.capture_end()

print(output.numpy()) # 此时可能会输出随机值，因为graph没有执行

graph.replay()
print(output.numpy()) # 此时会输出11

# 实际运行过程
for input_tensor in dataloader():
    input.copy_(input_tensor, False)
    graph.replay()
    print(output.numpy())

graph.reset() # 非必须调用，但最好调用以提前释放不需要的显存
```

## 静态图的使用方法
```Python
import numpy as np
import paddle
from paddle.device.cuda.graphs import CUDAGraph

input = paddle.static.data(shape=[1], dtype='float32', name='input')
input.persistable = True # 必须设置，否则无法输入
output = input + 1
output.persistable = True # 必须设置，否则无法拿到输出结果

place = paddle.CUDAPlace(0)
exe = paddle.static.Executor(place)

exe.run(paddle.static.default_startup_program())

scope = paddle.static.global_scope()
input_tensor_var = scope.var(input.name).get_tensor()
output_tensor_var = scope.var(output.name).get_tensor()
graph = None
capture_batch_id = 1 # 从第几个batch开始capture
for batch_id, input_tensors in dataloader():
    if graph is not None:
        input_tensor_var._copy_from(input_tensors[0][input.name])
        graph.replay()
    else:
        if batch_id == capture_batch_id:
            input_tensor_var._copy_from(input_tensors[0][input.name]) # copy输入
            input_tensors = None # capture的时候feed必须是None
            graph = CUDAGraph()
            graph.capture_begin()
        exe.run(paddle.static.default_main_program(), feed=input_tensors)
        if batch_id == capture_batch_id:
            graph.capture_end()
            graph.replay() # capture完后，必须调用graph.replay()，否则这轮相当于没跑
    print(np.array(output_tensor_var)) # 打印输出结果

if graph is not None:
    graph.reset() # 非必须调用，但最好调用以提前释放不需要的显存
```

## API使用和实现上的限制

- CUDA Graph只能用于全定长网络。例如，有些预测场景，除了最后一个batch的batch size小点以外，其他batch的batch size都一样，这也不能使用CUDA Graph。
- 网络中不能包含一些变化因素在里面。因为CUDA Graph是一次性capture，多次replay的。如果网络中存在一些变化因素（比如if-else op，while op，AMP dynamic loss scaling），这些变化因素不会在replay过程中体现。
- 静态图不要在第一个batch去capture CUDA Graph。因为第一个batch可能会涉及一些初始化操作，可能会在CUDA Graph中引入额外的GPU操作。
- 多卡场景只能使用fleet的多进程、每个进程一个GPU卡的方式运行。注：多卡多线程未测试是否可行。
- 使用`ParallelExecutor`时必须设置`build_strategy.allow_cuda_graph_capture = True`。这是因为：
	- `ParallelExecutor`会每隔若干个iteration删除一次scope里面的所有var，然后在下一个iteration重新创建、初始化这些var。假设这个间隔iteration数量是100，我们在第101轮做CUDA Graph capture，那么就会使得CUDA Graph额外capture一些GPU操作。
	- `ParallelExecutor`内部有一个H2D的copy，而且Host指针会析构。这个操作必须提前在CUDA Graph capture做。
- 静态图不能通过`Executor.run`的feed参数传入输入数据，即feed参数必须为空。必须保证输入变量persistable = True，然后通过`tensor = paddle.static.global_scope().var(var_name).get_tensor; tensor._copy_from(...)`的方式来输入数据。这是因为CUDA Graph必须保证所有tensor的指针不变。目前Paddle `Executor.run`的feed方式均可能会改变输入tensor的指针。
- 静态图不能通过`Executor.run`的fetch_list参数取出输出数据，即fetch_list参数必须为空。因为fetch_list不为空时，会有`cudaStreamSynchronize`操作，而`cudaStreamSynchronize`不可以在CUDA Graph capture的过程中调用。
- LRScheduler在CUDA Graph capture过程中直接不生效，必须像输入变量那样手动给LearningRate设置值。这是因为LRScheduler在`Executor.run`的过程中存在H2D copy，且Host指针会析构，无法保证CUDA Graph replay过程的正确性。
- 现在Paddle存在很多的H2D copy都存在上述问题，理论上都无法保证这些场景下CUDA Graph replay的正确性。
- 使用`ParallelExecutor`时必须设置`export FLAGS_sync_nccl_allreduce=0`，因为这个环境变量会在调用`ncclAllReduce`后做一次`cudaStreamSynchronize`，而`cudaStreamSynchronize`不可以在CUDA Graph capture的过程中调用。
- 使用多线程DataLoader时，必须保证`mode = "thread_local"`，否则DataLoader里面有些GPU操作也会被capture进去（主要是DoubleBufferReader会有H2D的async copy）。此外，由于`mode = "thread_local"`，ParallelExecutor不能使用多线程来跑，需要设置`exec_strategy.num_threads = 1`或者`build_strategy.fix_op_run_order = True`。
- 在做CUDA Graph capture的时候，框架会给CUDA Graph申请一片单独的GPU memory pool，而且不会调用`cudaFree`，保证tensor的数据不被析构且不被CUDA Graph外的其他操作污染。因此CUDA Graph可能会使得显存占用量升高。若爆显存，建议在CUDA Graph capture前可以先调用一次`paddle.device.cuda.empty_cache()`先把框架cache的显存清理掉，腾出空间给CUDA Graph capture。此外，CUDA Graph不使用后，请手动调用`graph.reset()`方法，该方法会释放CUDA Graph占用的GPU memory pool。
