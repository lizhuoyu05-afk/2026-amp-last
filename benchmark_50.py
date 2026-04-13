import os
import torch
import pandas as pd
from transformers import AutoTokenizer
from src.modeling import ESMForMultiLabel
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score

# ==========================================
# 🧬 极度硬核的 50 条多肽基准测试集 (正负 1:1)
# ==========================================
data = [
    # ---- 👿 25条 强毒性/细胞毒性/溶血肽 (Cytotoxicity = 1) ----
    {"Name": "Melittin (蜂毒肽)", "Sequence": "GIGAVLKVLTTGLPALISWIKRKRQQ", "Cytotoxicity": 1},
    {"Name": "LL-37 (人类抗菌肽,具毒性)", "Sequence": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES", "Cytotoxicity": 1},
    {"Name": "BMAP-28 (牛强毒性肽)", "Sequence": "GGLRSLGRKILRAWKKYGPIIVPIIRIG", "Cytotoxicity": 1},
    {"Name": "Pardaxin (豹鳎毒肽)", "Sequence": "GFFALIPKIISSPLFKTLLSAVGSALSSSGGQE", "Cytotoxicity": 1},
    {"Name": "Mastoparan (黄蜂毒素)", "Sequence": "INLKALAALAKKIL", "Cytotoxicity": 1},
    {"Name": "Protegrin-1 (猪白细胞杀菌肽)", "Sequence": "RGGRLCYCRRRFCVCVGR", "Cytotoxicity": 1},
    {"Name": "Indolicidin (牛吲哚青霉素)", "Sequence": "ILPWKWPWWPWRR", "Cytotoxicity": 1},
    {"Name": "Aurein 1.2 (澳洲雨蛙肽)", "Sequence": "GLFDIIKKIAESF", "Cytotoxicity": 1},
    {"Name": "Citropin 1.1 (树蛙毒肽)", "Sequence": "GLFDVIKKVASVIGGL", "Cytotoxicity": 1},
    {"Name": "Dermaseptin S1 (皮抑菌肽)", "Sequence": "ALWKTMLKKLGTMALHVKKAVIDPLIQKL", "Cytotoxicity": 1},
    {"Name": "Magainin 2 (爪蟾抗菌肽)", "Sequence": "GIGKFLHSAKKFGKAFVGEIMNS", "Cytotoxicity": 1},
    {"Name": "PGLa (两栖类抗菌肽)", "Sequence": "GMASKAGAIAGKIAKVALKAL", "Cytotoxicity": 1},
    {"Name": "Temporin A (林蛙肽)", "Sequence": "FLPLIGRVLSGIL", "Cytotoxicity": 1},
    {"Name": "Brevinin-1E (短蛙肽)", "Sequence": "FLPLLAGLAANFLPKIFCKITRKC", "Cytotoxicity": 1},
    {"Name": "Esculentin-1 (水蛙肽)", "Sequence": "GIFSKLGRKKIKNLLISGLKNVGKEVGMDVVRTGIDIAGCKIKGEC", "Cytotoxicity": 1},
    {"Name": "Maculatin 1.1 (雨蛙肽)", "Sequence": "GLFGVLAKVAAHVVPAIAEHF", "Cytotoxicity": 1},
    {"Name": "Ovispirin (绵羊衍生肽)", "Sequence": "KNLRRIIRKIIHIIKKYG", "Cytotoxicity": 1},
    {"Name": "Cecropin A (天蚕素A)", "Sequence": "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK", "Cytotoxicity": 1},
    {"Name": "Buforin II (蟾蜍素II)", "Sequence": "TRSSRAGLQFPVGRVHRLLRK", "Cytotoxicity": 1},
    {"Name": "Tachyplesin I (鲎素I)", "Sequence": "KWCFRVCYRGICYRRCR", "Cytotoxicity": 1},
    {"Name": "Polyphemusin I (巨鲎素I)", "Sequence": "RRWCFRVCYRGFCYRKCR", "Cytotoxicity": 1},
    {"Name": "Defensin HNP-1 (人类防御素1)", "Sequence": "ACYCRIPACIAGERRYGTCIYQGRLWAFCC", "Cytotoxicity": 1},
    {"Name": "Melittin Variant (蜂毒突变体)", "Sequence": "GIGAVLKVLTTGLPALISWIKRKRQQG", "Cytotoxicity": 1},
    {"Name": "Bactenecin (牛环状杀菌肽)", "Sequence": "RLCRIVVIRVCR", "Cytotoxicity": 1},
    {"Name": "Ponericin (蚂蚁毒素)", "Sequence": "FLPLLAGLAANFLPKIFCKITRKC", "Cytotoxicity": 1},

    # ---- 👼 25条 人体安全肽/管家蛋白/无意义乱码 (Cytotoxicity = 0) ----
    {"Name": "Insulin A-chain (胰岛素A链)", "Sequence": "GIVEQCCTSICSLYQLENYCN", "Cytotoxicity": 0},
    {"Name": "Insulin B-chain (胰岛素B链)", "Sequence": "FVNQHLCGSHLVEALYLVCGERGFFYTPKA", "Cytotoxicity": 0},
    {"Name": "Actin fragment (肌动蛋白片段)", "Sequence": "DEDETTALVCDNGSGLVK", "Cytotoxicity": 0},
    {"Name": "Albumin signal (白蛋白信号肽)", "Sequence": "MKWVTFISLLFLFSSAYS", "Cytotoxicity": 0},
    {"Name": "Hemoglobin Beta (血红蛋白Beta片段)", "Sequence": "VHLTPEEKSAVTALWGKVNV", "Cytotoxicity": 0},
    {"Name": "Hemoglobin Alpha (血红蛋白Alpha片段)", "Sequence": "VLSPADKTNVKAAWGKVGAH", "Cytotoxicity": 0},
    {"Name": "Collagen type I (胶原蛋白片段)", "Sequence": "GPPGPPGPPGPPGPPGPPGPP", "Cytotoxicity": 0},
    {"Name": "Collagen variant (胶原蛋白变体)", "Sequence": "GAPGAPGAPGAPGAPGAPGAP", "Cytotoxicity": 0},
    {"Name": "Leu-enkephalin (亮氨酸脑啡肽)", "Sequence": "YGGFL", "Cytotoxicity": 0},
    {"Name": "Beta-endorphin (Beta-内啡肽)", "Sequence": "YGGFMTSEKSQTPLVTLFKNAIIKNAYKKGE", "Cytotoxicity": 0},
    {"Name": "Bradykinin (缓激肽)", "Sequence": "RPPGFSPFR", "Cytotoxicity": 0},
    {"Name": "Angiotensin II (血管紧张素II)", "Sequence": "DRVYIHPF", "Cytotoxicity": 0},
    {"Name": "Oxytocin (催产素)", "Sequence": "CYIQNCPLG", "Cytotoxicity": 0},
    {"Name": "Synthetic safe helix (安全人工螺旋)", "Sequence": "SAAEAAKEAAKEAAKEA", "Cytotoxicity": 0},
    {"Name": "GS Linker (常见柔性连接肽)", "Sequence": "GGGGSGGGGSGGGGS", "Cytotoxicity": 0},
    {"Name": "Poly-Glu-Pro (纯人工多聚肽)", "Sequence": "EPEPEPEPEPEPEPEPEPEP", "Cytotoxicity": 0},
    {"Name": "Alpha-helix control (螺旋对照组)", "Sequence": "KAAAAKAAAAKAAAAK", "Cytotoxicity": 0},
    {"Name": "Beta-sheet control (折叠对照组)", "Sequence": "AEAEAEAEAEAEAEAEAEAE", "Cytotoxicity": 0},
    {"Name": "Huntingtin frag (多聚谷氨酰胺)", "Sequence": "PQQQQQQQQQQQQQQQ", "Cytotoxicity": 0},
    {"Name": "Prion frag safe (普里昂安全片段)", "Sequence": "QYNPQGGSRPSQD", "Cytotoxicity": 0},
    {"Name": "Serine rich tail (富丝氨酸尾)", "Sequence": "TSSTSSSSSSSSSS", "Cytotoxicity": 0},
    {"Name": "Random Basic (随机碱性乱码)", "Sequence": "AAAAGGGGCCCCLLLL", "Cytotoxicity": 0},
    {"Name": "All 20 AAs (全20种氨基酸乱序)", "Sequence": "ACDEFGHIKLMNPQRSTVWY", "Cytotoxicity": 0},
    {"Name": "Random Scramble 1 (无意义序列1)", "Sequence": "SVSVGMKASVRKHCFT", "Cytotoxicity": 0},
    {"Name": "Random Scramble 2 (无意义序列2)", "Sequence": "YWVTRSQPNMLKIHGFEDCA", "Cytotoxicity": 0},
]

# 保存为独立测试集 CSV
df_test = pd.DataFrame(data)
test_file = "./data/benchmark_50.csv"
df_test.to_csv(test_file, index=False)
print(f"📦 成功生成 50 条外部独立基准测试集: {test_file}")

# ==========================================
# 🚀 唤醒 35M 学生模型并执行预测
# ==========================================
model_dir = "./runs/cytotox_student_35m/best"
print(f"\n🧠 正在加载 35M 小钢炮模型: {model_dir}")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = ESMForMultiLabel("facebook/esm2_t12_35M_UR50D")

safe_path = os.path.join(model_dir, "model.safetensors")
bin_path = os.path.join(model_dir, "pytorch_model.bin")
if os.path.exists(safe_path):
    from safetensors.torch import load_file
    model.load_state_dict(load_file(safe_path))
else:
    model.load_state_dict(torch.load(bin_path, map_location="cpu"))
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("⚡ 开启极速盲测...")
inputs = tokenizer(df_test["Sequence"].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    logits = model(**inputs)["logits"]
    probs = torch.sigmoid(logits).squeeze(-1).tolist()
    if isinstance(probs, float): probs = [probs]

# 使用验证集测出的巅峰阈值
THRESHOLD = 0.59
df_test["Predict_Prob"] = [round(p, 4) for p in probs]
df_test["Predict_Label"] = [1 if p >= THRESHOLD else 0 for p in probs]

# ==========================================
# 📊 打印泛化成绩单
# ==========================================
print("\n" + "="*80)
print(f"{'预测状态':<6} | {'多肽名称':<30} | {'真实标签':<6} | {'预测概率':<6}")
print("-" * 80)

error_count = 0
for _, row in df_test.iterrows():
    is_correct = row["Cytotoxicity"] == row["Predict_Label"]
    mark = "✅ 命中" if is_correct else "❌ 误判"
    if not is_correct: error_count += 1
    
    # 格式化打印
    print(f"[{mark}] | {row['Name'][:28]:<30} | {row['Cytotoxicity']:<8} | {row['Predict_Prob']:.4f}")

y_true = df_test["Cytotoxicity"].tolist()
y_pred = df_test["Predict_Label"].tolist()

mcc = matthews_corrcoef(y_true, y_pred)
roc = roc_auc_score(y_true, probs)
pr_auc = average_precision_score(y_true, probs)

print("="*80)
print("🏆 Benchmark 50 最终泛化成绩单")
print("="*80)
print(f"🔹 测试总数: 50 条 (正25 / 负25)")
print(f"🔹 预测正确: {50 - error_count} 条 (准确率: {(50-error_count)/50*100:.1f}%)")
print(f"🎯 PR-AUC (极其重要): {pr_auc:.4f}")
print(f"📈 ROC-AUC (分辨力)  : {roc:.4f}")
print(f"🥇 MCC (综合及格线)  : {mcc:.4f}")
print("="*80)