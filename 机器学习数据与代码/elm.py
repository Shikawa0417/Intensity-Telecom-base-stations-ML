import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, make_scorer, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import cross_val_score


# 1. ELM 算法

class ELMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden=100, activation='sigmoid'):
        self.n_hidden = n_hidden
        self.activation = activation

    def _activate(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        return x

    def fit(self, X, y):
        X = np.array(X)
        # 将标签转为 One-hot 编码
        self.classes_ = np.unique(y)
        y_encoded = pd.get_dummies(y).values
        
        # 随机初始化权重和偏置
        np.random.seed(42)
        self.w = np.random.normal(size=(X.shape[1], self.n_hidden))
        self.b = np.random.normal(size=self.n_hidden)
        
        # 计算隐藏层输出
        H = self._activate(np.dot(X, self.w) + self.b)
        
        # 使用广义逆矩阵求解输出权重 beta
        self.beta = np.dot(np.linalg.pinv(H), y_encoded)
        return self

    def predict(self, X):
        X = np.array(X)
        H = self._activate(np.dot(X, self.w) + self.b)
        out = np.dot(H, self.beta)
        return self.classes_[np.argmax(out, axis=1)]


# 2. 数据预处理

file_path = r"D:\文件\AA干活\铁塔\data\铁塔数据.xlsx"

try:
    df = pd.read_excel(file_path)
    feature_columns = ['Ms', 'Pout', 'Poff', 'Dbs', 'Dpop']
    X_raw = df[feature_columns]
    y = df['kI']

    # 缺失值与标准化 (沿用你之前的逻辑)
    X_clean = X_raw.interpolate(method='linear').ffill().bfill()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    X_final = pd.DataFrame(X_scaled, columns=feature_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )
    print("✅ 数据预处理完成！")
except Exception as e:
    print(f"❌ 报错：{e}")
    exit()


# 3. Optuna 调参

def objective(trial):
    n_hidden = trial.suggest_int('n_hidden', 10, 500)
    activation = trial.suggest_categorical('activation', ['sigmoid', 'relu'])
    
    model = ELMClassifier(n_hidden=n_hidden, activation=activation)
    scorer = make_scorer(f1_score, average='macro')
    
    score = cross_val_score(model, X_train, y_train, cv=5, scoring=scorer).mean()
    return score

print("\n🚀 正在启动 ELM Optuna 寻优...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("\n🏆 ELM 最佳参数：", study.best_params)

# 使用最佳参数训练
best_elm = ELMClassifier(**study.best_params)
best_elm.fit(X_train, y_train)


# 4. 评估与特征分析

y_final_pred = best_elm.predict(X_test)
print("\n📊 优化后 ELM 评估报告：")
print(classification_report(y_test, y_final_pred))

# 执行置换检验
result = permutation_importance(best_elm, X_test, y_test, n_repeats=10, random_state=42)


elm_imp = pd.DataFrame({
    '特征': feature_columns,
    '贡献度均值': result.importances_mean,
    '标准差': result.importances_std  # <--- 增加这一行
}).sort_values(by='贡献度均值', ascending=False)


plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")


ax = sns.barplot(
    x='贡献度均值', 
    y='特征', 
    data=elm_imp, 
    palette='Purples_r', 
    hue='特征', 
    legend=False
)

plt.errorbar(
    x=elm_imp['贡献度均值'], 
    y=range(len(elm_imp)), 
    xerr=elm_imp['标准差'], 
    fmt='none', 
    c='black', 
    capsize=3
)

plt.title('ELM Feature Importance with Std Dev')
plt.xlabel('Importance (Accuracy Drop)')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()

print("\n--- ELM 特征排名 (含标准差) ---")
print(elm_imp)