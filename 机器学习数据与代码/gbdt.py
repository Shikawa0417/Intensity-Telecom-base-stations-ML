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


# 1. 数据预处理
file_path = r"D:\文件\AA干活\铁塔\data\铁塔数据.xlsx"

try:
    df = pd.read_excel(file_path)
    print("✅ 数据读取成功！")

    feature_columns = ['Ms', 'Pout', 'Poff', 'Dbs', 'Dpop']
    X_raw = df[feature_columns]
    y = df['kI']  # 目标变量

    # 缺失值处理 
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


# 2. GBDT 

print("\n--- 正在运行基础 GBDT ---")
base_gbdt = GradientBoostingClassifier(random_state=42)
base_gbdt.fit(X_train, y_train)
y_pred_base = base_gbdt.predict(X_test)
print(f"基础模型准确率: {accuracy_score(y_test, y_pred_base):.4%}")


# 3. Optuna 调参 

def objective(trial):
    # GBDT 核心超参数
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

# 使用最佳参数训练最终模型
best_gbdt = GradientBoostingClassifier(**study.best_params, random_state=42)
best_gbdt.fit(X_train, y_train)

# 4. 评估报告与特征分析 

y_final_pred = best_gbdt.predict(X_test)
print("\n📊 优化后 GBDT 评估报告：")
print(classification_report(y_test, y_final_pred))


result = permutation_importance(best_gbdt, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)


gbdt_imp = pd.DataFrame({
    '特征': feature_columns,
    '贡献度均值': result.importances_mean,
    '标准差': result.importances_std
}).sort_values(by='贡献度均值', ascending=False)

plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")
sns.barplot(
    x='贡献度均值', 
    y='特征', 
    data=gbdt_imp, 
    palette='Blues_r',
    hue='特征', 
    legend=False
)
plt.title('GBDT Feature Importance (Permutation)')
plt.xlabel('Importance (Accuracy Drop)')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()

print("\n--- GBDT 特征排名详情 ---")
print(gbdt_imp)