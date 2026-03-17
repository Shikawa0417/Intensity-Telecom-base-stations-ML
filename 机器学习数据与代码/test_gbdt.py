import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, make_scorer, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance

# ==========================================
# 1. 数据预处理
# ==========================================
file_path = r"D:\文件\AA干活\铁塔\data\铁塔数据.xlsx"

try:
    df = pd.read_excel(file_path)
    print("✅ 数据读取成功！")

    feature_columns = ['Ms', 'Pout', 'Poff', 'Dbs', 'Dpop']
    X_raw = df[feature_columns]
    y = df['kI']  # 目标变量


    if X_raw.isnull().values.any():
        X_clean = X_raw.interpolate(method='linear').ffill().bfill()
        print("⚠️ 缺失值已处理。")
    else:
        X_clean = X_raw

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    X_final = pd.DataFrame(X_scaled, columns=feature_columns)

    # 划分数据集 (8:2 比例)
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"📊 数据就绪：训练集 {X_train.shape[0]} 条，测试集 {X_test.shape[0]} 条。")

except Exception as e:
    print(f"❌ 运行报错：{e}")
    exit()

# ==========================================
# 2. 基础 GBDT (未调参)
# ==========================================
print("\n--- 正在运行基础 GBDT ---")
base_gbdt = GradientBoostingClassifier(random_state=42)
base_gbdt.fit(X_train, y_train)
y_pred_base = base_gbdt.predict(X_test)
print(f"基础模型准确率: {accuracy_score(y_test, y_pred_base):.4%}")

# ==========================================
# 3. Optuna 自动调参 (优化 GBDT)
# ==========================================
def objective(trial):

    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 8)
    subsample = trial.suggest_float('subsample', 0.6, 1.0) # 采样比例，防止过拟合
    
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=42
    )
    

    scorer = make_scorer(f1_score, average='macro')
    score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=5, scoring=scorer).mean()
    return score

print("\n🚀 正在启动 GBDT Optuna 寻优...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("\n🏆 GBDT 最佳参数组合：", study.best_params)

best_gbdt = GradientBoostingClassifier(**study.best_params, random_state=42)
best_gbdt.fit(X_train, y_train)


target_event = df[df['time'] == '2023/12/18 23:59'] 

print(f"📍 选定地震样本数: {len(target_event)}")

X_target_raw = target_event[feature_columns]

X_target_clean = X_target_raw.interpolate(method='linear').bfill().ffill()

X_target_scaled = scaler.transform(X_target_clean)

predictions = best_gbdt.predict(X_target_scaled)
prob = best_gbdt.predict_proba(X_target_scaled)[:, 1]


comparison_df = target_event.copy()
comparison_df['预测标签'] = predictions
comparison_df['预测概率'] = prob
comparison_df['是否准确'] = (comparison_df['kI'] == comparison_df['预测标签'])

output_path = r"D:\文件\AA干活\铁塔\data\地震预测对比结果1.xlsx"
comparison_df.to_excel(output_path, index=False)

print(f"✅ 导出成功！文件已保存在：{output_path}")