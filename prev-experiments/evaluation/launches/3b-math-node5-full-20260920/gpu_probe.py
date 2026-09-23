import json
import os

import torch


def main():
    selected = os.environ['CUDA_VISIBLE_DEVICES']
    if not selected.startswith('GPU-') or ',' in selected or torch.cuda.device_count() != 1:
        raise ValueError('Probe requires one assigned GPU UUID')
    torch.set_num_threads(1)
    device = torch.cuda.get_device_properties(0)
    actual = str(device.uuid).removeprefix('GPU-')
    if actual != selected.removeprefix('GPU-'):
        raise ValueError('CUDA probe identity differs from assigned UUID')
    if not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 is unsupported')
    matrix = torch.randn((128, 128), device='cuda', dtype=torch.bfloat16, requires_grad=True)
    loss = (matrix @ matrix.T).float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss).item() or not torch.isfinite(matrix.grad).all().item():
        raise ValueError('BF16 backward produced nonfinite values')
    print(json.dumps({'status': 'passed', 'gpu_uuid': selected, 'name': device.name, 'memory_bytes': device.total_memory, 'bf16_backward': True}))


if __name__ == '__main__':
    main()
