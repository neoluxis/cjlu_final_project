# segformer_app PySide6 ONNX GUI 计划

## Summary

在 `segformer_app` 中创建 PySide6 桌面程序，使用 ONNX Runtime 对单张图片、图片文件夹或视频逐帧执行预处理与语义分割推理。GUI 使用 Qt Designer `.ui` 文件定义，不在 Python 代码里硬编码布局；预测区域支持内部 hover 命中，并可在设置栏切换边界内部半透明填充。

## Key Changes

- `main.py` 启动 QApplication，打开主窗口。
- `main_window.ui` 定义三栏 GUI 布局，并在设置栏提供“开启区域半透明填充”开关。
- `app_window.py` 加载 `.ui`，绑定按钮、下拉框、快捷键、状态栏和图像交互事件。
- `inference.py` 负责 ONNX Runtime session 加载、输入张量生成、softmax、阈值过滤。
- `preprocess.py` 提供 `none / clahe / gaussian / gamma` 预处理。
- `media_source.py` 支持单图、图片文件夹、视频帧读取与前后切换。
- `overlay.py` 做 mask 后处理、边界提取、可选半透明填充、类别可见性过滤、悬浮信息命中检测。
- `model_registry.py` 扫描 `.onnx` 文件并从文件名/路径识别真实网络名。
- `settings.py` 使用 `QSettings` 保存/恢复上次路径、模型、预处理、阈值、填充开关、窗口状态。

## GUI Behavior

- 左栏为设置栏，可打开模型文件夹、重新加载模型文件夹、选择模型、选择预处理、设置阈值、设置是否开启区域半透明填充、打开图片文件夹/图片/视频、开始预处理、切换原图/预处理图、开始推理。
- 左栏设置最大宽度并优先展示相对路径，完整模型目录保留在 tooltip 中，避免挤占图像预览区。
- 用户选择模型后立即在后台线程加载 ONNX session，并执行一次 dummy warmup；加载/预热完成前禁用“开始推理”，避免第一张真实图片承担模型冷启动时间。
- 第二栏为图像展示栏：初始显示原图；点击预处理后显示预处理图；点击推理后在当前预览基底上叠加预测边界，开启填充时在边界内部叠加半透明类别色。
- 第二栏不直接修改原图或预处理图数据；内部保留 base image 与 overlay，显示时组合成 pixmap，并在控件 resize 时自动按比例重新缩放。
- 鼠标进入预测连通区域内部任意位置时立即显示类别、平均置信度、面积像素、面积占比、bbox、区域编号；命中判断使用预测区域 `instance_map`，不依赖可见边界线宽，也不等待系统默认 tooltip 悬停延迟。
- 第三栏列出当前预测中出现的所有非背景类别；每个类别用 checkbox 控制是否显示，类别栏设置最小宽度，类别项拆成多行展示，避免标签信息被截断。
- 左右箭头和 `A/D` 切换上一张/下一张或上一帧/下一帧。
- 底部状态栏显示预处理时间、推理时间、后处理时间、预计 FPS、当前序号/总数、当前文件名/视频帧号、自动识别的真实网络名和实际使用设备。

## Model Discovery And Naming

- 递归扫描用户选择的模型文件夹中所有 `.onnx` 文件。
- 模型下拉框显示模型原名/相对路径。
- 文件名/路径包含 `segformer` 或 `mit-b0` 显示 `SegFormer`。
- 包含 `deeplabv3plus` 或 `deeplabv3+` 显示 `DeepLabV3+`。
- 包含 `pspnet` 显示 `PSPNet`。
- 包含 `mask2former` 显示 `Mask2Former`。
- 包含 `unet` 显示 `UNet`。
- 未匹配时显示 `Unknown Network`。
- 文件名包含 `_clahe/_gaussian/_gamma` 时自动推荐对应预处理，否则推荐 `none`。

## Inference And Data Contract

- ONNX 输入按现有导出脚本约定：输入名通常为 `input`，shape 为 `[1, 3, 512, 512]` 或动态 H/W。
- 输出按 `seg_logits` 或第一个输出处理，shape 为 `[1, C, H, W]`。
- 对 class 维做 softmax，取 `argmax` 得到类别 mask，取 `max probability` 得到置信度图。
- 类别使用 SWRD 固定 9 类和 palette，背景类 `0` 不显示、不列入第三栏。
- 后处理过滤 `confidence < threshold` 的像素，对每个非背景类别做 connected components，并计算区域信息。
- 模型后台加载完成后用全零 RGB dummy 输入做一次预热推理；预热结果丢弃，只用于提前初始化 ONNX Runtime 执行路径。

## Deployment Decision

- v1 继续只支持 ONNX Runtime，不支持 TorchScript。
- 当前 GUI、依赖、模型扫描和推理包装都是 ONNX 专用，已有 `.onnx` 模型资产；ONNX 导出脚本使用 tensor-only wrapper，输出 `seg_logits`，与 GUI 的 `[1, C, H, W]` logits 合约一致。
- 默认使用 `CPUExecutionProvider`，避免在未安装 CUDA 12/cuDNN 9 运行库时加载 `CUDAExecutionProvider` 报错；如需 GPU 推理，先补齐 ONNX Runtime GPU 依赖，再通过 `SEGFORMER_APP_USE_CUDA=1` 显式开启。
- TorchScript 导出脚本依赖 PyTorch/MMSEG 运行栈，并使用 `forward_dummy` trace；接入需要新增 `.pt` 模型发现、后端抽象、输入输出适配和额外依赖，兼容风险高于 v1 收益。
- 如后续确需 TorchScript，应作为 v2 单独设计 `SegmentorBackend` 抽象，再评估 `.pt`/`.torchscript` 模型扫描、设备选择、输出合约和依赖打包。

## State And Rendering Rules

- 每个媒体项维护 `original_image`、`preprocessed_image`、`current_base_mode`、`prediction_result`。
- “查看原图/预处理图”只改变 `current_base_mode` 并刷新显示，不影响 mask、不影响推理结果。
- 类别 checkbox 只改变可见类别集合并刷新 overlay。
- “开启区域半透明填充”只改变 overlay 渲染方式并刷新显示，不影响 mask、推理结果、类别可见性或 hover 命中。
- 阈值变化后标记当前预测结果过期，用户再次点击推理才重新展示。
- 预处理方式变化后标记预处理缓存和预测结果过期。
- 不把 overlay 写回原图或预处理图；所有显示图均临时合成。

## Test Plan

- 程序从 `main_window.ui` 加载界面，Python 代码不手写三栏布局。
- 无配置、无模型、空目录状态下能打开主窗口。
- 选择包含 `.onnx` 的文件夹后，下拉框正确列出模型原名。
- 模型文件夹标签显示相对路径，鼠标悬停可查看完整绝对路径。
- 选择模型后 UI 不阻塞，状态栏显示后台加载/预热进度；预热完成前“开始推理”不可用，完成后可用。
- SegFormer、DeepLabV3+、PSPNet、Mask2Former、UNet 文件名能识别为真实网络名。
- 状态栏显示真实网络名，不显示模型原始文件名。
- 状态栏显示 ONNX Runtime 实际使用设备，例如 `CPU` 或 `CUDA`。
- 单图、图片文件夹、视频均可打开；箭头和 `A/D` 正确切换。
- 点击“开始预处理”后第二栏替换为预处理图；切换按钮可在原图/预处理图之间来回切换。
- 当前看原图时 overlay 叠加到原图；当前看预处理图时 overlay 叠加到预处理图。
- 调整窗口或图像控件大小后，当前图像自动重新按比例缩放并保持居中。
- 第三栏只列出预测中出现的非背景类别；勾选/取消类别能立即控制第二栏显示。
- 设置栏最大宽度不会挤占图像预览；类别栏最小宽度足以显示类别名称、区域数、面积和置信度。
- 鼠标进入第二栏预测区域内部任意位置时立即显示正确类别、置信度、面积和 bbox；移动到背景、隐藏类别或图像外时立即隐藏。
- 开启半透明填充时显示类别色填充和边界；关闭时只显示边界。
- 切换半透明填充开关不清空预测、不重新推理、不改变第三栏类别列表。
- 关闭后重新打开，能恢复上次模型目录、媒体路径、阈值、预处理选择、填充开关和预览模式。

## Assumptions

- v1 只实现 ONNX 推理，不直接加载 `.pth` 或 MMSeg config。
- v1 不支持 TorchScript 部署；如需支持，作为 v2 后续能力单独设计。
- 区域半透明填充默认开启，透明度固定为 `0.30`。
- v1 只做单帧手动推理，不做连续播放推理、不做批量导出。
- v1 固定服务 SWRD 9 类语义分割模型。
