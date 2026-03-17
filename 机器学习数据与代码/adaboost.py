import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, make_scorer, f1_score
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

# 2. 基础 AdaBoost
print("\n--- 正在运行基础 AdaBoost ---")
base_ada = AdaBoostClassifier(random_state=42)
base_ada.fit(X_train, y_train)
y_pred_base = base_ada.predict(X_test)
print(f"基础模型准确率: {accuracy_score(y_test, y_pred_base):.4%}")


# 3. Optuna 调参
def objective(trial):
    # 调优范围
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0, log=True)
    max_depth = trial.suggest_int('max_depth', 1, 3)
    
    # 构建弱学习器
    weak_learner = DecisionTreeClassifier(max_depth=max_depth, class_weight='balanced')
    
    model = AdaBoostClassifier(
        estimator=weak_learner,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )
    
    # 以 F1-macro 为目标
    scorer = make_scorer(f1_score, average='macro')
    score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=5, scoring=scorer).mean()
    return score

print("\n🚀 正在启动 Optuna 寻优 (请稍候)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("\n🏆 最佳参数组合：", study.best_params)


best_ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=study.best_params['max_depth'], class_weight='balanced'),
    n_estimators=study.best_params['n_estimators'],
    learning_rate=study.best_params['learning_rate'],
    random_state=42
)
best_ada.fit(X_train, y_train)


# 4. 评估报告与特征分析
y_final_pred = best_ada.predict(X_test)
print("\n📊 优化后 AdaBoost 评估报告：")
print(classification_report(y_test, y_final_pred))

# 置换检验计算重要性
result = permutation_importance(best_ada, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)


ada_imp = pd.DataFrame({
    '特征': feature_columns,
    '贡献度均值': result.importances_mean,
    '标准差': result.importances_std
}).sort_values(by='贡献度均值', ascending=False)


plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid") # 默认白背景网格
sns.barplot(
    x='贡献度均值', 
    y='特征', 
    data=ada_imp, 
    palette='Reds_r', 
    hue='特征', 
    legend=False
)
plt.title('AdaBoost Feature Importance (Permutation)')
plt.xlabel('Importance (Accuracy Drop)')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()

print("\n--- 特征排名详情 ---")
print(ada_imp)