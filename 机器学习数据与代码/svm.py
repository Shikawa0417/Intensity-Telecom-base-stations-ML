import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


file_path = r"D:\文件\AA干活\铁塔\data\铁塔数据.xlsx"

try:

    df = pd.read_excel(file_path)
    print("✅ 数据读取成功！")
    print(f"📊 当前数据集包含 {df.shape[0]} 条样本，{df.shape[1]} 个字段。")


    feature_columns = [
        'Ms', 
        'Pout', 
        'Poff', 
        'Dbs', 
        'Dpop'
    ]
    

    X_raw = df[feature_columns]
    y = df['I']


    if X_raw.isnull().values.any():
        print(f"⚠️ 检测到数字列存在缺失值，正在执行线性插值填充...")
   
        X_clean = X_raw.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    else:
        X_clean = X_raw
        print("✅ 数字特征列完整，无缺失值。")

  
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

  
    X_final = pd.DataFrame(X_scaled, columns=feature_columns)

    print("\n--- 预处理及标准化后的特征数据 (前5行) ---")
    print(X_final.head())
    print("\n🚀 预处理成功！")

except Exception as e:
    print(f"❌ 运行过程中发生错误：{e}")

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. 目标变量切换为二分类标签 ---
y = df['kI'] 

# --- 2. 划分数据集 ---

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. 初始化并训练 SVM ---
svm_model = SVC(kernel='linear', C=1.0, gamma=0.2, probability=True)
svm_model.fit(X_train, y_train)

# --- 4. 预测与评估 ---
y_pred = svm_model.predict(X_test)

print("\n" + "="*30)
print("📊 二分类 SVM 评估结果")
print("="*30)
print(f"总准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4%}")
print("\n详细分类报告:")
print(classification_report(y_test, y_pred))

# --- 5. 绘制混淆矩阵 (Confusion Matrix) ---

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low Intensity (0)', 'High Intensity (1)'],
            yticklabels=['Low Intensity (0)', 'High Intensity (1)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix (Binary Classification)')
plt.show()

#optuna优化
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

def objective(trial):
    c = trial.suggest_float('C', 0.001, 100, log=True)
    gamma = trial.suggest_float('gamma', 0.0001, 1, log=True)
    
    kernel = trial.suggest_categorical('kernel', ['sigmoid', 'rbf', 'linear'])
    
    model = SVC(
        C=c, 
        gamma=gamma, 
        kernel=kernel, 
        class_weight='balanced', 
        probability=True
    )
    

    scorer = make_scorer(f1_score, average='macro')
    
    score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=5, scoring=scorer).mean()
    return score

print("\n🚀 正在进行深度优化...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)


print("\n" + "="*30)
print("🏆 最佳参数组合已找到：")
print(study.best_params)
print(f"最高 F1-macro 分数: {study.best_value:.4%}")
print("="*30)


best_model = SVC(**study.best_params, class_weight='balanced', probability=True)
best_model.fit(X_train, y_train)


y_final_pred = best_model.predict(X_test)
print("\n📊 参数优化后的SVM报告：")
print(classification_report(y_test, y_final_pred))

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

print("\n正在计算SVM的特征贡献度...")





final_model = SVC(**study.best_params, class_weight='balanced', probability=True)
final_model.fit(X_train, y_train)
result = permutation_importance(
    final_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)


svm_imp = pd.DataFrame({
    '特征': feature_columns,
    '贡献度均值': result.importances_mean,
    '标准差': result.importances_std
}).sort_values(by='贡献度均值', ascending=False)



plt.figure(figsize=(10, 6))

sns.barplot(x='贡献度均值', y='特征', data=svm_imp, palette='Reds_r')

plt.xlabel('准确率下降程度')

plt.ylabel('监测指标')

plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()

plt.show()

print(svm_imp)
