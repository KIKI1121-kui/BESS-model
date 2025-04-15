import numpy as np
import pandas as pd

# 加载电价数据
price_data = pd.read_csv("电价数据.csv")
electricity_price = price_data['price'].values[:90] # 取前90天

price_diff = np.diff(electricity_price)
max_daily_diff = np.max(np.abs(price_diff))
avg_price = np.mean(electricity_price)
std_dev_price = np.std(electricity_price)
sell_price_ratio = 0.8 # 和 optimizer.py 中一致
max_potential_roundtrip_profit_ratio = (max(electricity_price) * sell_price_ratio - min(electricity_price)) / avg_price

print(f"电价分析 (前90天):")
print(f"  平均电价: {avg_price:.4f}")
print(f"  电价标准差: {std_dev_price:.4f}")
print(f"  最大单日价格变化: {max_daily_diff:.4f}")
print(f"  最高价: {np.max(electricity_price):.4f}")
print(f"  最低价: {np.min(electricity_price):.4f}")
print(f"  最大理论单次往返获利空间 (比例): {max_potential_roundtrip_profit_ratio:.4f}")

# 考虑基本的往返效率损失 (即使没在目标函数里扣除，物理上存在)
nominal_efficiency = 0.95 # BESSModel中的标称效率
round_trip_efficiency = nominal_efficiency * nominal_efficiency # 简化估算
print(f"  简化往返效率: {round_trip_efficiency:.4f}")
print(f"  收支平衡所需最低售/购价比: {1 / round_trip_efficiency:.4f}")
print(f"  当前最高/最低价比 * 售电折扣: {(np.max(electricity_price) / np.min(electricity_price)) * sell_price_ratio:.4f}")