import os
import torch
from transformers import AutoTokenizer
from src.modeling import ESMForMultiLabel

# 你的 35M 小钢炮模型路径
ckpt_dir = "./runs/cytotox_student_35m/best"

print("🧠 正在加载 35M 小钢炮模型与分词器...")
tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)

# 初始化骨架并加载你训练的专属权重
model = ESMForMultiLabel("facebook/esm2_t12_35M_UR50D")
if os.path.exists(os.path.join(ckpt_dir, "model.safetensors")):
    from safetensors.torch import load_file
    model.load_state_dict(load_file(os.path.join(ckpt_dir, "model.safetensors")))
else:
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu"))

model.eval()  # 开启预测模式

# 🧬 终极面试题：换成你想测的任何序列！
sequences = [
    "GIGAVLKVLTTGLPALISWIKRKRQQ",  # 著名的蜂毒肽(Melittin)，已知具有强细胞毒性/溶血性
    "GIGKFLHSAKKFGKAFVGEIMNS",     # 爪蟾抗菌肽(Magainin 2)，通常被认为是较安全的抗菌肽
    "ACDEFGHIKLMNPQRSTVWYYW",      # 随便在键盘上乱敲的一条无意义短肽
]

print("\n🚀 开始测算...")
inputs = tokenizer(sequences, padding=True, truncation=True, max_length=256, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs)["logits"]
    # 将模型输出转化为 0~1 的概率
    probs = torch.sigmoid(logits).squeeze(-1).tolist()

print("-" * 50)
# 使用咱们刚才在验证集上自动找出的“最佳及格线” 0.59
for seq, prob in zip(sequences, probs):
    status = "❌ 高危 (Toxic)" if prob >= 0.59 else "✅ 安全 (Safe)"
    print(f"多肽: {seq[:20]:<20}... \n概率: {prob:.4f} \n判定: {status}\n")
print("-" * 50)