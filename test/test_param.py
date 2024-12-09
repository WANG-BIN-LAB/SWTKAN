import sys
sys.path.append("/home/cxc/code/TOSICA-2layer-2/")
import TOSICA
from TOSICA.TOSICA_model import SwinTransformer


model = SwinTransformer(num_genes=3000, token=300,embed_dim=48,window_size=10,device='cuda',num_classes=13)

model.to('cuda')
for name, para in model.named_parameters():
    print(f"Loading params {name} with shape {para.shape}")


pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
print(f"params:  {pre_freeze_param_count} ")