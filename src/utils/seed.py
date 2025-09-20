import random, os, numpy as np, torch
def set_seed(seed):
    random.seed(seed); 
    np.random.seed(seed); 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(False)  # 속도/재현성 트레이드오프에 맞게 조절