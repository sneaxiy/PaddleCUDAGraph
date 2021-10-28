import paddle
from paddle.device.cuda.graphs import CUDAGraph
import numpy as np

concat_num = 20 # 当concat_num足够大（大于10）的时候，Paddle的concat op实现里会有H2D copy
use_cuda_graph = True # 当use_cuda_graph = False时，以下代码没问题。当use_cuda_graph = True时，以下代码有bug

xs = []

for i in range(concat_num):
    x_np = np.array([i], dtype=np.int64)
    xs.append(paddle.to_tensor(x_np))

graph = None
if use_cuda_graph: 
    graph = CUDAGraph()
    graph.capture_begin()

y = paddle.concat(xs)

if use_cuda_graph:
    graph.capture_end()
    graph.replay()

paddle.device.cuda.current_stream().synchronize()
print(y.numpy()) 

if use_cuda_graph:
    graph.reset()
