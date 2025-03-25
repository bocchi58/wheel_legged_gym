import torch
import os
import copy

def export_policy_as_onnx(actor_critic, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy.onnx")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    model.eval()

    # 创建一个示例输入张量
    dummy_input = torch.randn(1, model.input_size)  # 根据模型的输入大小调整

    # 导出为 ONNX 格式
    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Exported policy as ONNX to: {path}")