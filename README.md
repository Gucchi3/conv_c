# PureC-DL-Kernels
![Language](https://img.shields.io/badge/language-C99-blue)
![Platform](https://img.shields.io/badge/platform-Any%20%28x86%2FARM%2FRISC--V%29-green)
![Status](https://img.shields.io/badge/status-Active%20Development-orange)

## 📖 Overview
A lightweight, zero-dependency C implementation of deep learning operators, targeting Convolutional Neural Networks (CNN), Vision Transformers (ViT), and State Space Models (Mamba/VMamba).

The goal is to achieve **maximum portability** across any processor architecture (x86, ARM, RISC-V, DSPs, MCUs) by using standard C99 without external libraries.

## 🚀 Key Features
- **Pure C implementation**: No C++, no heavy frameworks (PyTorch/TensorFlow).
- **Hardware Agnostic**: Compiles on any platform with a standard C compiler.
- **Embedded Optimization**: Efficient pointer arithmetic and memory management designed for resource-constrained devices.

## 🎯 Target Workflow
The ultimate goal is to automate the deployment pipeline:
1. **PyTorch $\to$ ONNX**: Export the trained model to ONNX format.
2. **Weight Extraction**: Use a Python script to convert parameters into C arrays defined in `weight.h`.
3. **Structure Mapping**: Define `W_Tensor` and `B_Tensor` structs in `weight.c` pointing to the data arrays.
4. **Inference**: `main.c` utilizes kernels from `src/` to automatically load weights and execute the model.

## 🗺️ Development Roadmap
The project currently focuses on implementing a standard CNN. Advanced architectures like Mamba will be addressed in later phases.

1. **Phase 1: Basic CNN Implementation** (Current Focus)
   - Implement essential layers (Conv2d, Activation, Pooling, Linear).
2. **Phase 2: MCU Verification**
   - Verify operation on actual Microcontrollers (GAP8/STM32/RISC-V).
3. **Phase 3: Advanced Architectures**
   - Implement Vision Transformer (ViT) and Mamba (SSM) blocks.

## 📊 Implementation Status
🚧 **Work in Progress**

| Category | Operator / Module | Status | Note |
| :--- | :--- | :---: | :--- |
| **Convolution** | Conv2d (HWC) | ✅ Done | Supports Stride, Padding, Bias |
||Conv2d_BN_ACT|✅ Done| Conv2d(fused BN) + ACT|
| | **PConv2d** (Pointwise) | 🚫 Suspended | Use `PConv2d_BN_ACT` instead. |
| | PConv2d_BN_ACT | ⏳ Todo | Header defined in `Conv2d.h` |
| | **DConv2d** (Depthwise) | 🚫 Suspended | Use `DConv2d_BN_ACT` instead. |
| | DConv2d_BN_ACT | ⏳ Todo | Header defined in `Conv2d.h` |
| **Pooling** | **Max Pooling** | ⏳ Todo | Standard downsampling |
| | Average Pooling | ⏳ Todo | Less frequent usage |
| **Linear** | **Linear** | ✅ Done | Done! |
| **Normalization** | Batch Norm | 🚫 Suspended | **Cancelled**: Fused into Conv via ONNX . |
| | Layer Norm | ⏳ Todo | Required for ViT / Mamba |
| **Activation** | **ReLU** | ✅ Done | |
| | RELU6 |✅ Done||
| | SiLU | ✅ Done | Required for Mamba blocks |
| | Sigmoid | ⏳ Todo | |
| | Softmax | ⏳ Todo | |
| **Attention** | **Self-Attention (QKV)** | ⏳ Todo | Postponed until CNN is complete. |
| **Mamba**| **Efficient VMamba S6** | ⏳ Todo | Postponed until CNN is complete. |

## 🛠 Utilities (Python)

Tools to bridge the gap between PyTorch training and C inference.

| Tool | Function | Status | Note |
| :--- | :--- | :---: | :--- |
| **Weight Exporter** | `.pth` (PyTorch) $\to$ `.h` (C Header) | 🚧 **Now** | Auto-generates `W_Tensor` / `B_Tensor` arrays |
|**Permute**|HWC -> CHW|⏳ Todo|Auto permute HWC -> CHW|
||CHW -> HWC|⏳ Todo|Auto permute CHW -> HWC|


## 🛠 Usage Example

🚧 **Under Construction** 🚧

Release a "From PyTorch to C" tutorial once verification is complete.

## 📄 License

MiT License
