## Benchmark

### 1.0

Benchmarking version: 1.0

Date: 13/12/2022

|      **Device**       | **Server Type** | **Req/Sec** | **Latency** | **Batch Size** |
|:---------------------:|-----------------|:-----------:|:-----------:|:--------------:|
| gpu-rtx (g5.2xlarge)  | PythonServe     |    ~0.2     |     7s      |       1        |
| gpu-rtx (g5.2xlarge)  | TritonServe     |    ~0.1     |    7.3s     |       1        |
| gpu-fast (p3.2xlarge) | PythonServe     |    ~0.2     |     6s      |       1        |
| gpu-fast (p3.2xlarge) | TritonServe     |    ~0.1     |    7.5s     |       1        |
