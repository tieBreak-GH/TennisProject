"""
Standalone GPU/backend diagnostic (docs/uygulama_plani.md Faz 1.2).

Reports which torch backend main._select_device would actually pick on this
machine, then times a TrackNet forward pass on it vs. CPU - so a user
setting up an AMD card (e.g. RX 9070 XT via torch-directml or ROCm) can tell
in seconds whether it's being picked up at all, without needing model
weights or a video.

Run: python check_gpu.py
"""
import sys
import time

import torch

from tracknet import BallTrackerNet


def _probe_directml():
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass
    return None


def describe_environment():
    lines = ['torch {}'.format(torch.__version__)]
    lines.append('  CUDA build: {}'.format(torch.version.cuda or 'yok (CPU-only derleme)'))
    lines.append('  ROCm/HIP build: {}'.format(torch.version.hip or 'yok'))
    lines.append('  torch.cuda.is_available(): {}'.format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        # NVIDIA CUDA and AMD ROCm builds both report True here - torch.version.hip
        # (checked above) is what tells them apart.
        lines.append('    -> {}'.format(torch.cuda.get_device_name(0)))
    lines.append('  torch.backends.mps.is_available(): {}'.format(torch.backends.mps.is_available()))
    dml = _probe_directml()
    lines.append('  torch_directml: {}'.format(
        'kurulu ve kullanilabilir ({})'.format(dml) if dml is not None else 'kurulu degil veya kullanilamiyor'))
    return '\n'.join(lines)


def benchmark_forward(device, n_iters=10):
    """
    Time n_iters forward passes of the ball-tracking model on `device`.
    Weights are random (untrained) - only compute/transfer cost is measured,
    so this needs no downloaded model weights.
    """
    model = BallTrackerNet(input_channels=9, out_channels=256).to(device).eval()
    inp = torch.rand(1, 9, 360, 640, device=device)

    with torch.no_grad():
        model(inp)  # warm-up: first call pays lazy kernel-compile/alloc cost

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            model(inp)
    elapsed = time.perf_counter() - start
    return elapsed / n_iters * 1000  # ms/frame


def main():
    print(describe_environment())
    print()

    from main import _select_device  # reuse the pipeline's actual selection logic
    chosen = _select_device(prefer_alt_gpu=True)
    print("main._select_device() bu makinede secer: '{}'".format(chosen))
    print()

    print('Kiyaslama (ms/kare, TrackNet forward pass, gercek agirlik gerektirmez):')
    cpu_ms = None
    try:
        cpu_ms = benchmark_forward('cpu')
        print('  CPU:            {:.1f} ms/kare'.format(cpu_ms))
    except Exception as e:
        print('  CPU:            basarisiz ({})'.format(e))

    if str(chosen) != 'cpu':
        try:
            gpu_ms = benchmark_forward(chosen)
            print('  {}: {:.1f} ms/kare'.format(chosen, gpu_ms))
            if cpu_ms is not None:
                speedup = cpu_ms / gpu_ms
                verdict = 'hizlanma' if speedup > 1 else 'yavaslama (bu backend bu is yukunde CPU dan hizli degil)'
                print('  -> {:.1f}x {}'.format(speedup, verdict))
        except Exception as e:
            print('  {}: basarisiz ({})'.format(chosen, e))
    else:
        print('  GPU bulunamadi - pipeline CPU kullanacak.')
        print('  AMD RX 9070 XT icin kurulum: README "AMD Windows GPU" bolumu.')


if __name__ == '__main__':
    sys.exit(main())
