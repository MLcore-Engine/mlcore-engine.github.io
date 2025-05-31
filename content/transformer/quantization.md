+++
date = '2025-05-27T15:58:39+08:00'
draft = true
title = '模型量化基础代码版'
+++



# 深度学习模型量化完整教程

## 目录
1. [量化基础理论](#1-量化基础理论)
2. [量化的数学原理](#2-量化的数学原理)
3. [量化方法详解](#3-量化方法详解)
4. [PyTorch量化实战](#4-pytorch量化实战)
5. [量化感知训练(QAT)](#6-量化感知训练qat)
6. [高级量化技术](#7-高级量化技术)
7. [量化调试与优化](#8-量化调试与优化)
8. [实际案例分析](#9-实际案例分析)
9. [常见问题与解决方案](#10-常见问题与解决方案)

---

## 1. 量化基础理论

### 1.1 什么是模型量化？

模型量化是将神经网络中的浮点数参数和计算转换为低精度表示的技术。这个过程类似于音频的数字化采样：将连续的模拟信号转换为离散的数字信号。

### 1.2 为什么需要量化？

#### 存储优势
```
原始模型 (FP32): 1个参数 = 32位 = 4字节
INT8量化后: 1个参数 = 8位 = 1字节
压缩比: 4:1

示例：
- BERT-Base (110M参数)
  - FP32: 440MB
  - INT8: 110MB
  - INT4: 55MB
```

#### 计算优势
```
FP32乘法: 需要专门的浮点运算单元(FPU)
INT8乘法: 可使用更简单的整数运算单元
          支持SIMD指令集加速
          
性能提升: 通常2-4倍
能耗降低: 约10-20倍
```

### 1.3 量化的挑战

1. **精度损失**: 离散化过程不可避免地引入误差
2. **动态范围**: 不同层的数值范围差异巨大
3. **异常值**: 极端值会影响量化效果
4. **量化噪声**: 累积误差可能导致性能下降

---

## 2. 量化的数学原理

### 2.1 线性量化的数学表示

#### 基本映射关系
将浮点数区间 [α, β] 映射到整数区间 [a, b]：

```
量化函数 Q: ℝ → ℤ
Q(x) = round(1/s · (x - z))

反量化函数 Q⁻¹: ℤ → ℝ  
Q⁻¹(x_q) = s · x_q + z

其中：
s (scale): 缩放因子
z (zero-point): 零点偏移
```

#### 参数计算

**缩放因子计算：**
```
s = (β - α) / (b - a)

对于INT8量化：
- 对称量化: s = max(|α|, |β|) / 127
- 非对称量化: s = (β - α) / 255
```

**零点计算：**
```
z = round(a - α/s)

约束条件: a ≤ z ≤ b
```

### 2.2 量化误差分析

#### 量化误差定义
```
ε(x) = Q⁻¹(Q(x)) - x

最大量化误差:
|ε_max| = s/2
```

#### 信噪比(SNR)分析
```
SNR = 10 · log₁₀(P_signal / P_noise)

其中：
P_signal = E[x²]
P_noise = E[ε²] ≈ s²/12 (均匀量化)

理论SNR ≈ 6.02 · b + 1.76 (dB)
b为量化位数
```

### 2.3 详细计算示例

假设要量化权重矩阵：
```python
W = [[2.1, -1.3, 0.5],
     [1.8, -2.7, 0.9],
     [-0.6, 3.2, -1.9]]
```

**步骤1: 统计分析**
```
最小值: α = -2.7
最大值: β = 3.2
范围: R = 5.9
```

**步骤2: INT8对称量化**
```
scale = max(|−2.7|, |3.2|) / 127
      = 3.2 / 127
      = 0.0252

量化过程：
W[0,0]: Q(2.1) = round(2.1 / 0.0252) = round(83.33) = 83
W[0,1]: Q(-1.3) = round(-1.3 / 0.0252) = round(-51.59) = -52
...

量化矩阵：
W_q = [[83, -52, 20],
       [71, -107, 36],
       [-24, 127, -75]]
```

**步骤3: 反量化验证**
```
W'[0,0] = 83 × 0.0252 = 2.092 (误差: 0.008)
W'[0,1] = -52 × 0.0252 = -1.310 (误差: 0.010)
...

均方误差(MSE) = 0.000067
```

---

## 3. 量化方法详解

### 3.1 均匀量化 vs 非均匀量化

#### 均匀量化
量化级别均匀分布：
```python
def uniform_quantize(x, num_bits=8):
    qmin = -(2**(num_bits-1))
    qmax = 2**(num_bits-1) - 1
    
    scale = (x.max() - x.min()) / (qmax - qmin)
    zero_point = qmin - round(x.min() / scale)
    
    x_q = torch.round(x / scale + zero_point)
    x_q = torch.clamp(x_q, qmin, qmax)
    
    return x_q, scale, zero_point
```

#### 非均匀量化
量化级别根据数据分布调整：
```python
def kmeans_quantize(x, num_clusters=256):
    # 使用K-means聚类找到最优量化级别
    x_flat = x.flatten()
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(x_flat.reshape(-1, 1))
    
    # 量化：将每个值映射到最近的聚类中心
    labels = kmeans.predict(x_flat.reshape(-1, 1))
    centers = kmeans.cluster_centers_
    
    x_q = centers[labels].reshape(x.shape)
    return x_q, centers
```

### 3.2 对称量化 vs 非对称量化

#### 对称量化实现
```python
class SymmetricQuantizer:
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.qmax = 2**(num_bits-1) - 1
        
    def calculate_scale(self, x):
        max_val = torch.max(torch.abs(x))
        self.scale = max_val / self.qmax
        return self.scale
    
    def quantize(self, x):
        self.calculate_scale(x)
        x_q = torch.round(x / self.scale)
        x_q = torch.clamp(x_q, -self.qmax, self.qmax)
        return x_q.to(torch.int8)
    
    def dequantize(self, x_q):
        return x_q.float() * self.scale
```

#### 非对称量化实现
```python
class AsymmetricQuantizer:
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.qmin = 0
        self.qmax = 2**num_bits - 1
        
    def calculate_params(self, x):
        min_val = torch.min(x)
        max_val = torch.max(x)
        
        self.scale = (max_val - min_val) / (self.qmax - self.qmin)
        self.zero_point = torch.round(self.qmin - min_val / self.scale)
        self.zero_point = torch.clamp(self.zero_point, self.qmin, self.qmax)
        
    def quantize(self, x):
        self.calculate_params(x)
        x_q = torch.round(x / self.scale + self.zero_point)
        x_q = torch.clamp(x_q, self.qmin, self.qmax)
        return x_q.to(torch.uint8)
    
    def dequantize(self, x_q):
        return self.scale * (x_q.float() - self.zero_point)
```

### 3.3 动态量化详解

动态量化在运行时计算量化参数：

```python
class DynamicQuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 存储INT8权重和量化参数
        self.register_buffer('weight_int8', None)
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_zero_point', None)
        
        # 原始权重用于量化
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def quantize_weights(self):
        # 量化权重（离线进行）
        quantizer = AsymmetricQuantizer(num_bits=8)
        self.weight_int8 = quantizer.quantize(self.weight)
        self.weight_scale = quantizer.scale
        self.weight_zero_point = quantizer.zero_point
        
    def forward(self, x):
        # 动态量化输入
        x_quantizer = AsymmetricQuantizer(num_bits=8)
        x_int8 = x_quantizer.quantize(x)
        
        # INT8矩阵乘法
        # 注意：实际实现需要特殊的INT8 GEMM库
        output_int32 = torch.matmul(x_int8.int(), self.weight_int8.int().t())
        
        # 反量化输出
        output_scale = x_quantizer.scale * self.weight_scale
        output_zero_point = 0  # 简化处理
        
        output = output_scale * output_int32.float() + self.bias
        
        return output
```

### 3.4 静态量化详解

静态量化需要校准过程：

```python
class StaticQuantizer:
    def __init__(self, num_bits=8, num_bins=2048):
        self.num_bits = num_bits
        self.num_bins = num_bins
        self.histogram = torch.zeros(num_bins)
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
    def update_stats(self, x):
        """收集激活值统计信息"""
        self.min_val = min(self.min_val, x.min().item())
        self.max_val = max(self.max_val, x.max().item())
        
        # 更新直方图
        x_flat = x.flatten()
        hist, bin_edges = torch.histogram(x_flat, bins=self.num_bins)
        self.histogram += hist
        
    def calculate_optimal_params(self, method='entropy'):
        """基于收集的统计信息计算最优量化参数"""
        if method == 'minmax':
            # 简单的最小-最大方法
            scale = (self.max_val - self.min_val) / (2**self.num_bits - 1)
            zero_point = round(-self.min_val / scale)
            
        elif method == 'entropy':
            # KL散度最小化方法
            scale, zero_point = self._minimize_kl_divergence()
            
        elif method == 'percentile':
            # 百分位数方法（去除异常值）
            scale, zero_point = self._percentile_calibration(99.9)
            
        return scale, zero_point
    
    def _minimize_kl_divergence(self):
        """使用KL散度找到最优量化参数"""
        # 实现较复杂，这里是简化版本
        best_scale = None
        best_zero_point = None
        min_kl = float('inf')
        
        # 尝试不同的量化范围
        for alpha in torch.linspace(0.5, 1.0, steps=100):
            test_max = self.max_val * alpha
            test_min = self.min_val * alpha
            
            scale = (test_max - test_min) / (2**self.num_bits - 1)
            zero_point = round(-test_min / scale)
            
            # 计算KL散度
            kl = self._compute_kl(scale, zero_point)
            
            if kl < min_kl:
                min_kl = kl
                best_scale = scale
                best_zero_point = zero_point
                
        return best_scale, best_zero_point
```

---

## 4. PyTorch量化实战

### 4.1 PyTorch量化API概览

```python
import torch
import torch.quantization as tq

# 量化配置
qconfig = tq.QConfig(
    activation=tq.MinMaxObserver.with_args(dtype=torch.quint8),
    weight=tq.MinMaxObserver.with_args(dtype=torch.qint8)
)

# 不同的观察器选项
observer_options = {
    'MinMaxObserver': tq.MinMaxObserver,
    'MovingAverageMinMaxObserver': tq.MovingAverageMinMaxObserver,
    'HistogramObserver': tq.HistogramObserver,
    'PerChannelMinMaxObserver': tq.PerChannelMinMaxObserver
}
```

### 4.2 完整的动态量化示例

```python
import torch
import torch.nn as nn
import torchvision.models as models

class DynamicQuantizationExample:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
    def quantize(self):
        """执行动态量化"""
        # 指定要量化的层类型
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            qconfig_spec={
                nn.Linear: torch.quantization.default_dynamic_qconfig,
                nn.LSTM: torch.quantization.default_dynamic_qconfig,
                nn.GRU: torch.quantization.default_dynamic_qconfig
            },
            dtype=torch.qint8
        )
        return self.quantized_model
    
    def compare_models(self, input_data):
        """比较量化前后的模型"""
        with torch.no_grad():
            # 原始模型推理
            original_output = self.model(input_data)
            
            # 量化模型推理
            quantized_output = self.quantized_model(input_data)
            
        # 计算差异
        mse = torch.mean((original_output - quantized_output) ** 2)
        print(f"MSE between outputs: {mse.item()}")
        
        # 模型大小比较
        original_size = self._get_model_size(self.model)
        quantized_size = self._get_model_size(self.quantized_model)
        
        print(f"Original model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Compression ratio: {original_size/quantized_size:.2f}x")
        
    def _get_model_size(self, model):
        """计算模型大小（MB）"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.numel() * param.element_size()
            
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
            
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def benchmark_speed(self, input_data, num_runs=100):
        """性能对比测试"""
        import time
        
        # 预热
        for _ in range(10):
            _ = self.model(input_data)
            _ = self.quantized_model(input_data)
        
        # 原始模型计时
        start = time.time()
        for _ in range(num_runs):
            _ = self.model(input_data)
        original_time = time.time() - start
        
        # 量化模型计时
        start = time.time()
        for _ in range(num_runs):
            _ = self.quantized_model(input_data)
        quantized_time = time.time() - start
        
        print(f"Original model: {original_time:.3f}s")
        print(f"Quantized model: {quantized_time:.3f}s")
        print(f"Speedup: {original_time/quantized_time:.2f}x")

# 使用示例
model = models.resnet18(pretrained=True)
quantizer = DynamicQuantizationExample(model)
quantized_model = quantizer.quantize()

# 测试
dummy_input = torch.randn(1, 3, 224, 224)
quantizer.compare_models(dummy_input)
quantizer.benchmark_speed(dummy_input)
```

### 4.3 静态量化完整流程

```python
class StaticQuantizationPipeline:
    def __init__(self, model, calibration_loader):
        self.model = model
        self.calibration_loader = calibration_loader
        
    def prepare_model(self):
        """准备模型进行量化"""
        # 1. 设置量化配置
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 2. 准备模型（插入观察器）
        self.prepared_model = torch.quantization.prepare(self.model)
        
        return self.prepared_model
    
    def calibrate(self):
        """使用代表性数据校准模型"""
        print("Calibrating model...")
        self.prepared_model.eval()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.calibration_loader):
                self.prepared_model(data)
                
                if batch_idx % 10 == 0:
                    print(f"Calibration batch {batch_idx}/{len(self.calibration_loader)}")
                    
                # 通常使用100-1000个batch进行校准
                if batch_idx > 100:
                    break
                    
        print("Calibration complete!")
        
    def convert(self):
        """转换为量化模型"""
        self.quantized_model = torch.quantization.convert(self.prepared_model)
        return self.quantized_model
    
    def evaluate_accuracy(self, test_loader):
        """评估量化模型精度"""
        correct = 0
        total = 0
        
        self.quantized_model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                output = self.quantized_model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        accuracy = 100 * correct / total
        print(f'Quantized Model Accuracy: {accuracy:.2f}%')
        return accuracy
    
    def full_pipeline(self):
        """执行完整的静态量化流程"""
        # 1. 准备
        self.prepare_model()
        
        # 2. 校准
        self.calibrate()
        
        # 3. 转换
        self.convert()
        
        # 4. 保存量化模型
        torch.save(self.quantized_model.state_dict(), 'quantized_model.pth')
        
        # 5. 也可以使用TorchScript保存
        scripted_quantized_model = torch.jit.script(self.quantized_model)
        scripted_quantized_model.save('quantized_model_scripted.pth')
        
        return self.quantized_model

# 自定义量化模块示例
class QuantizableConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        self.fc = nn.Linear(64 * 8 * 8, 10)
        
    def forward(self, x):
        # 量化输入
        x = self.quant(x)
        
        # 主要计算
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = torch.nn.functional.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        # 反量化输出
        x = self.dequant(x)
        return x
    
    def fuse_modules(self):
        """融合可以合并的层"""
        torch.quantization.fuse_modules(self, [
            ['conv1', 'bn1', 'relu1'],
            ['conv2', 'bn2', 'relu2']
        ], inplace=True)
```

### 4.4 Post-Training量化优化技巧

```python
class AdvancedQuantizationTechniques:
    def __init__(self, model):
        self.model = model
        
    def bias_correction(self, calibration_loader):
        """偏差校正技术"""
        # 收集原始模型和量化模型的输出
        original_outputs = []
        quantized_outputs = []
        
        self.model.eval()
        quantized_model = self.quantize_model()
        
        with torch.no_grad():
            for data, _ in calibration_loader:
                original_outputs.append(self.model(data))
                quantized_outputs.append(quantized_model(data))
                
        # 计算并校正偏差
        for name, module in quantized_model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # 计算平均偏差
                bias_correction = self._compute_bias_correction(
                    original_outputs, quantized_outputs, name
                )
                
                # 应用偏差校正
                if module.bias is not None:
                    module.bias.data += bias_correction
                else:
                    module.bias = nn.Parameter(bias_correction)
                    
    def equalization(self):
        """权重均衡化"""
        # 找到连续的Conv/Linear层
        consecutive_layers = self._find_consecutive_layers()
        
        for layer1, layer2 in consecutive_layers:
            # 计算均衡化因子
            scale_factor = self._compute_equalization_scale(layer1, layer2)
            
            # 应用均衡化
            layer1.weight.data *= scale_factor.view(-1, 1, 1, 1)
            layer2.weight.data /= scale_factor.view(1, -1, 1, 1)
            
            if layer1.bias is not None:
                layer1.bias.data *= scale_factor
                
    def outlier_channel_splitting(self, threshold=3.0):
        """异常通道分割"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # 计算每个通道的统计信息
                channel_max = module.weight.data.abs().view(
                    module.out_channels, -1
                ).max(dim=1)[0]
                
                mean = channel_max.mean()
                std = channel_max.std()
                
                # 识别异常通道
                outlier_mask = channel_max > (mean + threshold * std)
                outlier_indices = torch.where(outlier_mask)[0]
                
                if len(outlier_indices) > 0:
                    # 分割异常通道
                    self._split_outlier_channels(module, outlier_indices)
```

---


## 5. 量化感知训练(QAT)深入

### 5.1 QAT的理论基础

```python
import torch
import torch.nn as nn

class FakeQuantize(nn.Module):
    """假量化操作的实现"""
    def __init__(self, observer, quant_min=-128, quant_max=127, 
                 learning_rate=1.0, symmetric=False):
        super().__init__()
        self.observer = observer
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.learning_rate = learning_rate
        self.symmetric = symmetric
        
        # 可学习的量化参数
        self.register_buffer('scale', torch.tensor([1.0]))
        self.register_buffer('zero_point', torch.tensor([0]))
        self.register_buffer('fake_quant_enabled', torch.tensor([1]))
        
    def forward(self, x):
        if self.training:
            # 训练时更新统计信息
            self.observer(x)
            
            # 计算量化参数
            scale, zero_point = self.observer.calculate_qparams()
            
            # 使用指数移动平均更新
            self.scale = self.scale * (1 - self.learning_rate) + \
                        scale * self.learning_rate
            self.zero_point = self.zero_point * (1 - self.learning_rate) + \
                            zero_point * self.learning_rate
        
        if self.fake_quant_enabled[0] == 1:
            # 执行假量化
            x = self.fake_quantize(x)
            
        return x
    
    def fake_quantize(self, x):
        """假量化的前向和反向传播"""
        # 量化
        x_int = torch.round(x / self.scale + self.zero_point)
        x_int = torch.clamp(x_int, self.quant_min, self.quant_max)
        
        # 反量化
        x_quant = (x_int - self.zero_point) * self.scale
        
        # 直通估计器(STE)用于梯度
        x_quant = x + (x_quant - x).detach()
        
        return x_quant
    
    @torch.jit.export
    def extra_repr(self):
        return f'quant_min={self.quant_min}, quant_max={self.quant_max}'
```

### 5.2 自定义QAT训练循环

```python
class QATTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def prepare_qat(self):
        """准备模型进行QAT"""
        # 1. 定义量化配置
        qat_config = torch.quantization.QConfig(
            activation=FakeQuantize.with_args(
                observer=torch.quantization.MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine
            ),
            weight=FakeQuantize.with_args(
                observer=torch.quantization.MovingAveragePerChannelMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric
            )
        )
        
        # 2. 应用配置
        self.model.qconfig = qat_config
        
        # 3. 准备模型
        torch.quantization.prepare_qat(self.model, inplace=True)
        
    def train_epoch(self, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            optimizer.zero_grad()
            
            output = self.model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
                
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self, criterion):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def train_qat(self, epochs, lr=0.001):
        """完整的QAT训练流程"""
        # 准备QAT
        self.prepare_qat()
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        
        # 训练循环
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            # 逐步启用量化
            if epoch < epochs // 4:
                # 前25%只训练，不量化
                self.disable_fake_quant()
            elif epoch < epochs // 2:
                # 25%-50%只量化权重
                self.enable_weight_fake_quant()
            else:
                # 50%-100%完全量化
                self.enable_all_fake_quant()
            
            # 训练
            train_loss, train_acc = self.train_epoch(optimizer, criterion)
            
            # 验证
            val_loss, val_acc = self.validate(criterion)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            scheduler.step()
            
        # 转换为量化模型
        self.model.eval()
        self.quantized_model = torch.quantization.convert(self.model)
        
        return self.quantized_model
    
    def disable_fake_quant(self):
        """禁用假量化"""
        for module in self.model.modules():
            if isinstance(module, FakeQuantize):
                module.fake_quant_enabled[0] = 0
                
    def enable_weight_fake_quant(self):
        """只启用权重假量化"""
        for name, module in self.model.named_modules():
            if isinstance(module, FakeQuantize):
                if 'weight' in name:
                    module.fake_quant_enabled[0] = 1
                else:
                    module.fake_quant_enabled[0] = 0
                    
    def enable_all_fake_quant(self):
        """启用所有假量化"""
        for module in self.model.modules():
            if isinstance(module, FakeQuantize):
                module.fake_quant_enabled[0] = 1
```

### 5.3 QAT高级技巧

```python
class AdvancedQATTechniques:
    def __init__(self, model):
        self.model = model
        
    def distillation_qat(self, teacher_model, temperature=4.0):
        """知识蒸馏辅助的QAT"""
        class DistillationLoss(nn.Module):
            def __init__(self, alpha=0.7, temperature=4.0):
                super().__init__()
                self.alpha = alpha
                self.temperature = temperature
                self.criterion = nn.CrossEntropyLoss()
                self.kl_div = nn.KLDivLoss(reduction='batchmean')
                
            def forward(self, student_output, labels, teacher_output):
                # 标准分类损失
                ce_loss = self.criterion(student_output, labels)
                
                # 蒸馏损失
                student_logits = F.log_softmax(
                    student_output / self.temperature, dim=1
                )
                teacher_logits = F.softmax(
                    teacher_output / self.temperature, dim=1
                )
                distillation_loss = self.kl_div(student_logits, teacher_logits)
                
                # 组合损失
                loss = self.alpha * ce_loss + \
                       (1 - self.alpha) * distillation_loss * self.temperature ** 2
                
                return loss, ce_loss, distillation_loss
            
        return DistillationLoss(temperature=temperature)
    
    def progressive_quantization(self, bit_schedule):
        """渐进式量化：从高位到低位"""
        for epoch, num_bits in bit_schedule.items():
            print(f"Epoch {epoch}: Switching to {num_bits}-bit quantization")
            
            # 更新所有量化器的位数
            for module in self.model.modules():
                if hasattr(module, 'weight_fake_quant'):
                    # 更新权重量化器
                    module.weight_fake_quant.quant_min = -(2**(num_bits-1))
                    module.weight_fake_quant.quant_max = 2**(num_bits-1) - 1
                    
                if hasattr(module, 'activation_fake_quant'):
                    # 更新激活量化器
                    module.activation_fake_quant.quant_min = 0
                    module.activation_fake_quant.quant_max = 2**num_bits - 1
                    
    def mixed_precision_qat(self, bit_config):
        """混合精度QAT：不同层使用不同位数"""
        for name, module in self.model.named_modules():
            if name in bit_config:
                num_bits = bit_config[name]
                
                # 为该层设置特定的量化配置
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    module.qconfig = self._get_qconfig_for_bits(num_bits)
                    
    def _get_qconfig_for_bits(self, num_bits):
        """根据位数返回相应的量化配置"""
        if num_bits == 8:
            return torch.quantization.default_qat_qconfig
        elif num_bits == 4:
            # 4位量化配置
            return torch.quantization.QConfig(
                activation=FakeQuantize.with_args(
                    observer=torch.quantization.MinMaxObserver,
                    quant_min=0,
                    quant_max=15,
                    dtype=torch.quint8
                ),
                weight=FakeQuantize.with_args(
                    observer=torch.quantization.MinMaxObserver,
                    quant_min=-8,
                    quant_max=7,
                    dtype=torch.qint8
                )
            )
        else:
            raise ValueError(f"Unsupported bit width: {num_bits}")
```

---

## 6. 高级量化技术

### 6.1 向量量化(Vector Quantization)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """向量量化器实现"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # 码本（codebook）
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs):
        # 输入形状: [B, C, H, W]
        input_shape = inputs.shape
        
        # 展平
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # 计算到每个码本向量的距离
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
        
        # 找到最近的码本索引
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 量化
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        # 计算VQ损失
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # 直通估计器
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encodings
    
    def get_codebook_usage(self, encodings_list):
        """分析码本使用情况"""
        usage = torch.zeros(self.num_embeddings)
        for encodings in encodings_list:
            usage += encodings.sum(0)
        return usage / len(encodings_list)
```

### 6.2 Product Quantization

```python
class ProductQuantizer:
    """乘积量化实现"""
    def __init__(self, dim, num_subvectors, num_centroids):
        self.dim = dim
        self.num_subvectors = num_subvectors
        self.num_centroids = num_centroids
        self.subvector_dim = dim // num_subvectors
        
        # 每个子向量的码本
        self.codebooks = []
        
    def fit(self, data, n_iter=10):
        """训练量化器"""
        n_samples = data.shape[0]
        
        # 将数据分割成子向量
        data_subvectors = data.reshape(n_samples, self.num_subvectors, 
                                      self.subvector_dim)
        
        # 为每个子向量训练码本
        for i in range(self.num_subvectors):
            subvector_data = data_subvectors[:, i, :]
            
            # 使用K-means聚类
            codebook = self._kmeans(subvector_data, self.num_centroids, n_iter)
            self.codebooks.append(codebook)
            
    def encode(self, data):
        """编码数据"""
        n_samples = data.shape[0]
        codes = np.zeros((n_samples, self.num_subvectors), dtype=np.int32)
        
        data_subvectors = data.reshape(n_samples, self.num_subvectors, 
                                      self.subvector_dim)
        
        for i in range(self.num_subvectors):
            subvector_data = data_subvectors[:, i, :]
            
            # 找到最近的码字
            distances = self._compute_distances(subvector_data, self.codebooks[i])
            codes[:, i] = np.argmin(distances, axis=1)
            
        return codes
    
    def decode(self, codes):
        """解码数据"""
        n_samples = codes.shape[0]
        reconstructed = np.zeros((n_samples, self.dim))
        
        for i in range(self.num_subvectors):
            start_idx = i * self.subvector_dim
            end_idx = (i + 1) * self.subvector_dim
            
            # 从码本中检索
            reconstructed[:, start_idx:end_idx] = self.codebooks[i][codes[:, i]]
            
        return reconstructed
    
    def _kmeans(self, data, k, n_iter):
        """简化的K-means实现"""
        n_samples = data.shape[0]
        
        # 随机初始化中心点
        indices = np.random.choice(n_samples, k, replace=False)
        centers = data[indices].copy()
        
        for _ in range(n_iter):
            # 分配到最近的中心
            distances = self._compute_distances(data, centers)
            labels = np.argmin(distances, axis=1)
            
            # 更新中心点
            for i in range(k):
                mask = labels == i
                if mask.any():
                    centers[i] = data[mask].mean(axis=0)
                    
        return centers
    
    def _compute_distances(self, X, Y):
        """计算欧氏距离"""
        return np.sum((X[:, np.newaxis] - Y[np.newaxis]) ** 2, axis=2)
```

### 6.3 学习型量化

```python
class LearnedStepSizeQuantization(nn.Module):
    """LSQ: Learned Step Size Quantization"""
    def __init__(self, num_bits=8, symmetric=True, per_channel=False):
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        
        if symmetric:
            self.Qn = -(2**(num_bits-1))
            self.Qp = 2**(num_bits-1) - 1
        else:
            self.Qn = 0
            self.Qp = 2**num_bits - 1
            
    def init_scale(self, x, num_channels=None):
        """初始化可学习的scale参数"""
        if self.per_channel:
            # 每通道scale
            x_flat = x.view(num_channels, -1)
            mean = x_flat.mean(dim=1)
            std = x_flat.std(dim=1)
            
            # 初始scale设置为 2*std / sqrt(Qp)
            init_scale = 2 * std / math.sqrt(self.Qp)
            
            self.scale = nn.Parameter(init_scale.view(-1, 1, 1, 1))
        else:
            # 全局scale
            mean = x.mean()
            std = x.std()
            init_scale = 2 * std / math.sqrt(self.Qp)
            
            self.scale = nn.Parameter(torch.tensor(init_scale))
            
    def forward(self, x):
        if not hasattr(self, 'scale'):
            self.init_scale(x, x.shape[0] if self.per_channel else None)
            
        # 梯度缩放因子
        grad_scale = 1.0 / math.sqrt(x.numel() * self.Qp)
        
        # 缩放梯度
        scale = grad_scale * self.scale
        
        # 量化
        x_scaled = x / scale
        x_quant = torch.round(torch.clamp(x_scaled, self.Qn, self.Qp))
        
        # 反量化
        x_dequant = x_quant * scale
        
        # STE
        return x + (x_dequant - x).detach()
```

### 6.4 二值化和三值化

```python
class BinaryQuantization(nn.Module):
    """二值量化：+1/-1"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # 使用符号函数
        x_binary = torch.sign(x)
        
        # STE用于反向传播
        x_binary = x + (x_binary - x).detach()
        
        return x_binary
    
class TernaryQuantization(nn.Module):
    """三值量化：+1/0/-1"""
    def __init__(self, threshold=0.7):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, x):
        # 计算阈值
        x_abs = torch.abs(x)
        threshold = self.threshold * x_abs.mean()
        
        # 三值量化
        x_ternary = torch.sign(x)
        x_ternary[x_abs < threshold] = 0
        
        # 缩放因子（保持方差）
        scale = x_abs[x_abs >= threshold].mean()
        x_ternary = x_ternary * scale
        
        # STE
        x_ternary = x + (x_ternary - x).detach()
        
        return x_ternary

class XNORNet(nn.Module):
    """XNOR-Net: 二值神经网络"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            BinaryConv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            BinaryActivation(),
            nn.MaxPool2d(2),
            
            BinaryConv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            BinaryActivation(),
            nn.MaxPool2d(2),
        )
        
    def forward(self, x):
        return self.features(x)

class BinaryConv2d(nn.Module):
    """二值卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        
    def forward(self, x):
        # 二值化权重
        w = self.conv.weight
        w_binary = torch.sign(w)
        
        # 计算缩放因子
        alpha = w.abs().mean(dim=(1,2,3), keepdim=True)
        
        # 二值卷积
        self.conv.weight = w + (w_binary - w).detach()
        output = self.conv(x)
        self.conv.weight = w
        
        return output * alpha.view(-1, 1, 1)
```

---

## 7. 量化调试与优化

### 7.1 量化敏感度分析

```python
class QuantizationSensitivityAnalyzer:
    def __init__(self, model, calibration_loader, validation_loader):
        self.model = model
        self.calibration_loader = calibration_loader
        self.validation_loader = validation_loader
        
    def analyze_layer_sensitivity(self):
        """分析每层对量化的敏感度"""
        # 获取原始模型精度
        baseline_accuracy = self.evaluate_model(self.model)
        print(f"Baseline accuracy: {baseline_accuracy:.2f}%")
        
        sensitivity_results = {}
        
        # 遍历每一层
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                print(f"\nAnalyzing layer: {name}")
                
                # 保存原始权重
                original_weight = module.weight.data.clone()
                
                # 测试不同量化位数
                for num_bits in [8, 4, 2]:
                    # 量化该层
                    quantized_weight = self.quantize_tensor(
                        original_weight, num_bits
                    )
                    module.weight.data = quantized_weight
                    
                    # 评估精度
                    accuracy = self.evaluate_model(self.model)
                    drop = baseline_accuracy - accuracy
                    
                    if name not in sensitivity_results:
                        sensitivity_results[name] = {}
                    sensitivity_results[name][f'{num_bits}bit'] = {
                        'accuracy': accuracy,
                        'drop': drop
                    }
                    
                    print(f"  {num_bits}-bit: {accuracy:.2f}% (drop: {drop:.2f}%)")
                    
                # 恢复原始权重
                module.weight.data = original_weight
                
        return sensitivity_results
    
    def quantize_tensor(self, tensor, num_bits):
        """量化张量"""
        if num_bits == 32:
            return tensor
            
        # 计算量化参数
        min_val = tensor.min()
        max_val = tensor.max()
        
        qmin = -(2**(num_bits-1))
        qmax = 2**(num_bits-1) - 1
        
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - torch.round(min_val / scale)
        
        # 量化和反量化
        tensor_q = torch.round(tensor / scale + zero_point)
        tensor_q = torch.clamp(tensor_q, qmin, qmax)
        tensor_deq = scale * (tensor_q - zero_point)
        
        return tensor_deq
    
    def find_optimal_bit_allocation(self, target_model_bits):
        """找到最优的位分配方案"""
        sensitivity = self.analyze_layer_sensitivity()
        
        # 根据敏感度排序层
        layer_importance = []
        for layer_name, results in sensitivity.items():
            # 计算敏感度分数（8bit到4bit的精度下降）
            sensitivity_score = results['4bit']['drop'] - results['8bit']['drop']
            layer_importance.append((layer_name, sensitivity_score))
            
        layer_importance.sort(key=lambda x: x[1], reverse=True)
        
        # 分配位数：重要的层用更多位
        bit_allocation = {}
        for i, (layer_name, _) in enumerate(layer_importance):
            if i < len(layer_importance) // 3:
                bit_allocation[layer_name] = 8  # 最重要的1/3用8位
            elif i < 2 * len(layer_importance) // 3:
                bit_allocation[layer_name] = 6  # 中间1/3用6位
            else:
                bit_allocation[layer_name] = 4  # 最不重要的1/3用4位
                
        return bit_allocation
```

### 7.2 量化错误分析

```python
class QuantizationErrorAnalyzer:
    def __init__(self, float_model, quantized_model):
        self.float_model = float_model
        self.quantized_model = quantized_model
        
    def analyze_activation_errors(self, input_data):
        """分析激活值的量化误差"""
        self.float_model.eval()
        self.quantized_model.eval()
        
        float_activations = {}
        quantized_activations = {}
        
        # 注册钩子收集激活值
        def get_activation(name, activation_dict):
            def hook(model, input, output):
                activation_dict[name] = output.detach()
            return hook
        
        # 为每层注册钩子
        for name, module in self.float_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
                module.register_forward_hook(
                    get_activation(name, float_activations)
                )
                
        for name, module in self.quantized_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
                module.register_forward_hook(
                    get_activation(name, quantized_activations)
                )
        
        # 前向传播
        with torch.no_grad():
            _ = self.float_model(input_data)
            _ = self.quantized_model(input_data)
        
        # 分析误差
        error_stats = {}
        for name in float_activations:
            if name in quantized_activations:
                float_act = float_activations[name]
                quant_act = quantized_activations[name]
                
                # 计算各种误差指标
                mse = torch.mean((float_act - quant_act) ** 2).item()
                mae = torch.mean(torch.abs(float_act - quant_act)).item()
                
                # 相对误差
                rel_error = torch.mean(
                    torch.abs(float_act - quant_act) / 
                    (torch.abs(float_act) + 1e-8)
                ).item()
                
                # 余弦相似度
                cos_sim = F.cosine_similarity(
                    float_act.flatten(), 
                    quant_act.flatten(), 
                    dim=0
                ).item()
                
                error_stats[name] = {
                    'mse': mse,
                    'mae': mae,
                    'relative_error': rel_error,
                    'cosine_similarity': cos_sim
                }
                
        return error_stats
    
    def visualize_weight_distributions(self):
        """可视化权重分布的变化"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        layer_idx = 0
        for name, module in self.float_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and layer_idx < 4:
                # 获取浮点权重
                float_weights = module.weight.data.cpu().numpy().flatten()
                
                # 获取量化权重
                quant_module = dict(self.quantized_model.named_modules())[name]
                quant_weights = quant_module.weight().dequantize().cpu().numpy().flatten()
                
                # 绘制分布
                ax = axes[layer_idx]
                ax.hist(float_weights, bins=50, alpha=0.5, label='Float32', density=True)
                ax.hist(quant_weights, bins=50, alpha=0.5, label='INT8', density=True)
                ax.set_title(f'Layer: {name}')
                ax.set_xlabel('Weight Value')
                ax.set_ylabel('Density')
                ax.legend()
                
                layer_idx += 1
                
        plt.tight_layout()
        plt.savefig('weight_distribution_comparison.png')
        plt.close()
```

### 7.3 量化模型优化

```python
class QuantizationOptimizer:
    def __init__(self, model, calibration_data):
        self.model = model
        self.calibration_data = calibration_data
        
    def optimize_quantization_parameters(self):
        """优化量化参数以最小化误差"""
        optimized_params = {}
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'observer'):
                print(f"Optimizing {name}")
                
                # 收集激活值
                activations = []
                def hook(module, input, output):
                    activations.append(output.detach())
                    
                handle = module.register_forward_hook(hook)
                
                # 运行校准数据
                with torch.no_grad():
                    for data in self.calibration_data:
                        _ = self.model(data)
                        
                handle.remove()
                
                # 优化量化参数
                all_activations = torch.cat(activations)
                optimal_scale, optimal_zero_point = self.optimize_scale_zp(
                    all_activations
                )
                
                optimized_params[name] = {
                    'scale': optimal_scale,
                    'zero_point': optimal_zero_point
                }
                
        return optimized_params
    
    def optimize_scale_zp(self, tensor, num_bits=8):
        """使用网格搜索优化scale和zero_point"""
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        best_scale = None
        best_zero_point = None
        min_error = float('inf')
        
        # 网格搜索
        for scale_factor in np.linspace(0.8, 1.2, 20):
            test_scale = (max_val - min_val) / (2**num_bits - 1) * scale_factor
            
            for zp_offset in range(-10, 11):
                test_zp = round(-min_val / test_scale) + zp_offset
                test_zp = np.clip(test_zp, 0, 2**num_bits - 1)
                
                # 量化和反量化
                tensor_q = torch.round(tensor / test_scale + test_zp)
                tensor_q = torch.clamp(tensor_q, 0, 2**num_bits - 1)
                tensor_dq = (tensor_q - test_zp) * test_scale
                
                # 计算误差
                error = torch.mean((tensor - tensor_dq) ** 2).item()
                
                if error < min_error:
                    min_error = error
                    best_scale = test_scale
                    best_zero_point = test_zp
                    
        return best_scale, best_zero_point
    
    def apply_graph_optimization(self):
        """应用计算图优化"""
        # 1. 算子融合
        self.model = torch.quantization.fuse_modules(self.model, [
            ['conv1', 'bn1', 'relu1'],
            ['conv2', 'bn2', 'relu2'],
        ])
        
        # 2. 常量折叠
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.optimize_for_inference(self.model)
        
        return self.model
```

---

## 9. 实际案例分析

### 9.1 BERT模型量化案例

```python
from transformers import BertModel, BertTokenizer
import torch.quantization as tq

class BERTQuantizationCase:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        
    def prepare_bert_for_quantization(self):
        """准备BERT模型进行量化"""
        # BERT特殊处理：某些层不适合量化
        quantization_blacklist = [
            'embeddings',  # 嵌入层保持高精度
            'pooler',      # 池化层
        ]
        
        # 设置量化配置
        qconfig = tq.QConfig(
            activation=tq.HistogramObserver.with_args(
                quant_min=0, 
                quant_max=127,
                dtype=torch.quint8,
                reduce_range=True  # 对激活值使用reduced range
            ),
            weight=tq.PerChannelMinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric
            )
        )
        
        # 应用量化配置
        for name, module in self.model.named_modules():
            # 检查是否在黑名单中
            if any(blacklist_item in name for blacklist_item in quantization_blacklist):
                module.qconfig = None
            else:
                module.qconfig = qconfig
                
    def quantize_bert_dynamic(self):
        """BERT动态量化"""
        # 动态量化对BERT效果较好
        quantized_model = tq.quantize_dynamic(
            self.model,
            {
                torch.nn.Linear: tq.default_dynamic_qconfig,
                # BERT使用的是nn.Linear而不是单独的LSTM/GRU
            },
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def quantize_bert_static(self, calibration_data):
        """BERT静态量化"""
        # 准备模型
        self.prepare_bert_for_quantization()
        self.model.eval()
        
        # 准备量化
        prepared_model = tq.prepare(self.model)
        
        # 校准
        print("Calibrating BERT...")
        with torch.no_grad():
            for batch in calibration_data:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                _ = prepared_model(input_ids, attention_mask=attention_mask)
                
        # 转换
        quantized_model = tq.convert(prepared_model)
        
        return quantized_model
    
    def compare_inference_speed(self, test_sentences, num_runs=100):
        """比较推理速度"""
        import time
        
        # 准备输入
        inputs = self.tokenizer(
            test_sentences, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        
        # 原始模型
        self.model.eval()
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(**inputs)
        original_time = time.time() - start
        
        # 量化模型
        quantized_model = self.quantize_bert_dynamic()
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = quantized_model(**inputs)
        quantized_time = time.time() - start
        
        print(f"Original model: {original_time:.2f}s")
        print(f"Quantized model: {quantized_time:.2f}s")
        print(f"Speedup: {original_time/quantized_time:.2f}x")
        
    def measure_model_size(self):
        """测量模型大小"""
        import os
        
        # 保存原始模型
        torch.save(self.model.state_dict(), 'bert_original.pth')
        original_size = os.path.getsize('bert_original.pth') / 1024 / 1024
        
        # 保存量化模型
        quantized_model = self.quantize_bert_dynamic()
        torch.save(quantized_model.state_dict(), 'bert_quantized.pth')
        quantized_size = os.path.getsize('bert_quantized.pth') / 1024 / 1024
        
        print(f"Original model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Compression ratio: {original_size/quantized_size:.2f}x")
        
        # 清理
        os.remove('bert_original.pth')
        os.remove('bert_quantized.pth')
```

### 9.2 ResNet量化案例

```python
class ResNetQuantizationCase:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        
    def prepare_resnet_for_mobile(self):
        """为移动端部署准备ResNet"""
        # 1. 模型修改：去除不必要的层
        class MobileResNet(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                # 复制除了最后的FC层
                self.features = nn.Sequential(
                    *list(original_model.children())[:-1]
                )
                # 添加量化/反量化节点
                self.quant = tq.QuantStub()
                self.dequant = tq.DeQuantStub()
                
                # 新的轻量级分类头
                self.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(2048, 1000)
                )
                
            def forward(self, x):
                x = self.quant(x)
                x = self.features(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                x = self.dequant(x)
                return x
                
        mobile_model = MobileResNet(self.model)
        
        # 2. 融合BatchNorm
        mobile_model.eval()
        mobile_model = self._fuse_resnet_modules(mobile_model)
        
        return mobile_model
    
    def _fuse_resnet_modules(self, model):
        """融合ResNet中的模块"""
        # ResNet的特殊结构需要仔细处理
        for module_name, module in model.named_children():
            if module_name == 'features':
                for idx, layer in enumerate(module):
                    if hasattr(layer, 'conv1'):
                        # 基本块
                        torch.quantization.fuse_modules(layer, [
                            ['conv1', 'bn1'],
                            ['conv2', 'bn2']
                        ], inplace=True)
                        
                        # 如果有conv3（瓶颈块）
                        if hasattr(layer, 'conv3'):
                            torch.quantization.fuse_modules(layer, [
                                ['conv3', 'bn3']
                            ], inplace=True)
                            
                        # 下采样
                        if hasattr(layer, 'downsample'):
                            torch.quantization.fuse_modules(
                                layer.downsample, 
                                ['0', '1'],  # conv, bn
                                inplace=True
                            )
                            
        return model
    
    def quantize_with_qat(self, train_loader, val_loader, epochs=10):
        """使用QAT量化ResNet"""
        # 准备模型
        model = self.prepare_resnet_for_mobile()
        
        # QAT配置
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # 准备QAT
        model.train()
        model = torch.quantization.prepare_qat(model)
        
        # 训练
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                    
            # 验证阶段
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    
            accuracy = 100. * correct / len(val_loader.dataset)
            print(f'Epoch {epoch}, Validation Accuracy: {accuracy:.2f}%')
            
        # 转换为量化模型
        model.eval()
        quantized_model = torch.quantization.convert(model)
        
        return quantized_model
```

### 9.3 YOLO目标检测量化案例

```python
class YOLOQuantizationCase:
    def __init__(self, model_path):
        self.model = self.load_yolo_model(model_path)
        
    def quantize_yolo_preserving_accuracy(self):
        """保持精度的YOLO量化策略"""
        # YOLO特殊考虑：
        # 1. 检测头需要高精度
        # 2. 特征提取可以更激进地量化
        
        # 混合精度配置
        mixed_precision_config = {
            # Backbone - 可以使用更低精度
            'backbone.conv1': 4,  # 4-bit
            'backbone.conv2': 4,
            'backbone.conv3': 6,  # 6-bit
            'backbone.conv4': 6,
            
            # Neck - 中等精度
            'neck.conv1': 8,  # 8-bit
            'neck.conv2': 8,
            
            # Head - 保持高精度
            'head.detect1': 16,  # FP16
            'head.detect2': 16,
            'head.detect3': 16,
        }
        
        # 应用混合精度量化
        for name, module in self.model.named_modules():
            if name in mixed_precision_config:
                bits = mixed_precision_config[name]
                module.qconfig = self._get_qconfig_for_bits(bits)
            elif 'backbone' in name:
                # 默认backbone使用INT4
                module.qconfig = self._get_qconfig_for_bits(4)
            else:
                # 其他部分使用INT8
                module.qconfig = torch.quantization.default_qconfig
                
    def post_process_quantized_outputs(self, outputs):
        """后处理量化输出"""
        # YOLO输出包含：位置、置信度、类别概率
        # 需要特殊处理以保持检测精度
        
        boxes = outputs[..., :4]  # 边界框
        confidence = outputs[..., 4:5]  # 置信度
        class_probs = outputs[..., 5:]  # 类别概率
        
        # 对边界框进行额外的校准
        # 量化可能导致边界框偏移
        boxes = self.calibrate_boxes(boxes)
        
        # 对置信度使用更严格的阈值
        # 量化可能导致更多的假阳性
        confidence_threshold = 0.5  # 比正常阈值略高
        
        return boxes, confidence, class_probs
```

---

## 10. 常见问题与解决方案

### 10.1 量化后精度严重下降

**问题分析：**
1. 某些层对量化特别敏感
2. 激活值分布有长尾
3. 量化参数选择不当

**解决方案：**

```python
class AccuracyRecoveryTechniques:
    def __init__(self, float_model, quantized_model):
        self.float_model = float_model
        self.quantized_model = quantized_model
        
    def selective_quantization(self, sensitive_layers):
        """选择性量化：跳过敏感层"""
        for name, module in self.quantized_model.named_modules():
            if name in sensitive_layers:
                # 恢复为浮点
                float_module = dict(self.float_model.named_modules())[name]
                setattr(self.quantized_model, name.split('.')[-1], float_module)
                
    def outlier_suppression(self, activation_data, percentile=99.9):
        """异常值抑制"""
        # 计算激活值的百分位数
        lower = torch.quantile(activation_data, (100 - percentile) / 100)
        upper = torch.quantile(activation_data, percentile / 100)
        
        # 裁剪异常值
        clipped_data = torch.clamp(activation_data, lower, upper)
        
        return clipped_data
    
    def knowledge_distillation_finetune(self, train_loader, epochs=5):
        """使用知识蒸馏微调量化模型"""
        optimizer = torch.optim.Adam(self.quantized_model.parameters(), lr=0.0001)
        kl_div = nn.KLDivLoss(reduction='batchmean')
        ce_loss = nn.CrossEntropyLoss()
        
        self.float_model.eval()
        
        for epoch in range(epochs):
            for data, target in train_loader:
                self.quantized_model.train()
                optimizer.zero_grad()
                
                # 获取教师和学生输出
                with torch.no_grad():
                    teacher_output = self.float_model(data)
                student_output = self.quantized_model(data)
                
                # 计算损失
                T = 4.0  # 温度参数
                distillation_loss = kl_div(
                    F.log_softmax(student_output / T, dim=1),
                    F.softmax(teacher_output / T, dim=1)
                ) * T * T
                
                student_loss = ce_loss(student_output, target)
                
                loss = 0.9 * distillation_loss + 0.1 * student_loss
                
                loss.backward()
                optimizer.step()
```

### 10.2 量化模型推理速度没有提升

**可能原因：**
1. 硬件不支持INT8加速
2. 模型太小，量化开销大于收益
3. 没有使用优化的推理引擎

**解决方案：**

```python
class InferenceOptimization:
    def __init__(self, quantized_model):
        self.model = quantized_model
        
    def optimize_for_hardware(self, target_hardware='cpu'):
        """针对特定硬件优化"""
        if target_hardware == 'cpu':
            # 使用FBGEMM后端（Intel CPU优化）
            torch.backends.quantized.engine = 'fbgemm'
            
        elif target_hardware == 'arm':
            # 使用QNNPACK后端（ARM优化）
            torch.backends.quantized.engine = 'qnnpack'
            
        # JIT编译优化
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.optimize_for_inference(self.model)
        
    def batch_inference_optimization(self):
        """批处理推理优化"""
        # 使用TorchScript批处理
        @torch.jit.script
        def optimized_forward(model, x):
            return model(x)
            
        return optimized_forward
    
    def export_to_optimized_format(self):
        """导出到优化格式"""
        # ONNX导出
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            self.model,
            dummy_input,
            "quantized_model.onnx",
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}}
        )
        
        # TensorRT优化（NVIDIA GPU）
        # 需要安装torch2trt
        try:
            from torch2trt import torch2trt
            model_trt = torch2trt(
                self.model, 
                [dummy_input], 
                fp16_mode=False, 
                int8_mode=True
            )
            return model_trt
        except ImportError:
            print("TensorRT not available")
```

### 10.3 量化参数选择困难

**自动化量化参数搜索：**

```python
class AutoQuantizationSearch:
    def __init__(self, model, calibration_loader, validation_loader):
        self.model = model
        self.calibration_loader = calibration_loader
        self.validation_loader = validation_loader
        
    def grid_search_quantization_params(self):
        """网格搜索最优量化参数"""
        param_grid = {
            'observer': [
                torch.quantization.MinMaxObserver,
                torch.quantization.MovingAverageMinMaxObserver,
                torch.quantization.HistogramObserver
            ],
            'reduce_range': [True, False],
            'symmetric': [True, False],
            'per_channel': [True, False]
        }
        
        best_accuracy = 0
        best_params = None
        
        from itertools import product
        
        # 生成所有参数组合
        keys = param_grid.keys()
        values = param_grid.values()
        
        for params in product(*values):
            param_dict = dict(zip(keys, params))
            
            print(f"Testing: {param_dict}")
            
            # 创建量化配置
            if param_dict['per_channel'] and param_dict['symmetric']:
                qscheme = torch.per_channel_symmetric
            elif param_dict['per_channel']:
                qscheme = torch.per_channel_affine
            elif param_dict['symmetric']:
                qscheme = torch.per_tensor_symmetric
            else:
                qscheme = torch.per_tensor_affine
                
            qconfig = torch.quantization.QConfig(
                activation=param_dict['observer'].with_args(
                    dtype=torch.quint8,
                    reduce_range=param_dict['reduce_range']
                ),
                weight=param_dict['observer'].with_args(
                    dtype=torch.qint8,
                    qscheme=qscheme
                )
            )
            
            # 量化模型
            quantized_model = self.quantize_with_config(qconfig)
            
            # 评估
            accuracy = self.evaluate_model(quantized_model)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = param_dict
                
        print(f"Best parameters: {best_params}")
        print(f"Best accuracy: {best_accuracy:.2f}%")
        
        return best_params
```

### 10.4 部署相关问题

```python
class DeploymentSolutions:
    def __init__(self, quantized_model):
        self.model = quantized_model
        
    def mobile_deployment_optimization(self):
        """移动端部署优化"""
        # 1. 模型大小优化
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        # 2. 转换为TorchScript
        example_input = torch.rand(1, 3, 224, 224)
        traced_model = torch.jit.trace(self.model, example_input)
        
        # 3. 优化图
        optimized_model = torch.jit.optimize_for_mobile(traced_model)
        
        # 4. 保存
        optimized_model._save_for_lite_interpreter("model_mobile.ptl")
        
        return optimized_model
    
    def edge_device_deployment(self):
        """边缘设备部署"""
        # 转换为ONNX
        dummy_input = torch.randn(1, 3, 224, 224)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            "model_edge.onnx",
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}},
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        )
        
        # 使用ONNX Runtime进行INT8推理
        import onnxruntime as ort
        
        # 创建INT8推理会话
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession("model_edge.onnx", sess_options)
        
        return session
    
    def handle_unsupported_ops(self):
        """处理不支持的操作"""
        # 自定义量化操作
        class CustomQuantizedOp(nn.Module):
            def __init__(self, float_op):
                super().__init__()
                self.scale = 1.0
                self.zero_point = 0
                self.float_op = float_op
                
            def forward(self, x):
                # 反量化
                x_float = (x.float() - self.zero_point) * self.scale
                
                # 执行浮点操作
                y_float = self.float_op(x_float)
                
                # 重新量化
                y_int = torch.round(y_float / self.scale + self.zero_point)
                y_int = y_int.clamp(0, 255).to(torch.uint8)
                
                return y_int
                
        # 替换不支持的操作
        for name, module in self.model.named_modules():
            if isinstance(module, UnsupportedOp):
                setattr(self.model, name.split('.')[-1], 
                       CustomQuantizedOp(module))
```