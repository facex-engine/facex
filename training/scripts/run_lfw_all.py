"""Quick driver: run LFW eval on all 4 trained models."""
import torch
import arch as a
from model import MobileFaceNet
from lfw_eval import evaluate

results = {}
for name in ['nano', 'tiny', 'standard', 'xs']:
    arch = a.ALL_ARCHS[name]
    model = MobileFaceNet(arch).cuda().eval()
    ck = torch.load(f'D:/apps/facex/training/runs/{name}/last.pt',
                     map_location='cuda', weights_only=False)
    model.load_state_dict(ck['model'] if 'model' in ck else ck)
    print(f'--- {name} ---')
    stats = evaluate(model, 'D:/apps/facex/training/data/lfw_aligned.bin', device='cuda')
    print(f"  LFW: {stats['accuracy_mean']*100:.2f}% +/- {stats['accuracy_std']*100:.2f}%  thr={stats['threshold_mean']:.3f}")
    results[name] = stats

print('\n==== Summary ====')
header_a = 'arch'
header_b = 'LFW'
header_c = 'std'
print(f"{header_a:>10}  {header_b:>10}  {header_c:>8}")
for n, s in results.items():
    print(f"{n:>10}  {s['accuracy_mean']*100:>8.2f}%  {s['accuracy_std']*100:>6.2f}%")
