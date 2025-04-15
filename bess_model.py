# bess_model.py

import numpy as np

class BESSModel:
    """
    电池储能系统(BESS)基本模型
    """
    def __init__(self, time_steps, initial_soc=0.5, capacity_kwh=200, max_power_kw=50,
                 min_soc=0.1, max_soc=0.9, efficiency=0.95):
        self.time_steps = time_steps
        self.initial_soc = initial_soc
        self.capacity_kwh = capacity_kwh
        self.max_power_kw = max_power_kw
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.efficiency = efficiency
        self.charge_efficiency = self.efficiency
        self.discharge_efficiency = self.efficiency
        self.time_interval = 1.0  # 1小时步长
        self.soc = np.zeros(time_steps+1)
        self.soc[0] = initial_soc
        self.power = np.zeros(time_steps)

        print(f"初始化BESS模型 - 容量:{capacity_kwh}kWh, 最大功率:{max_power_kw}kW, "
              f"SOC范围:[{min_soc},{max_soc}], 效率:{efficiency*100}%")

    def update_soc(self, power_vector):
        """
        根据给定的功率向量更新SOC
        正值=放电, 负值=充电
        """
        soc_vector = np.zeros(self.time_steps+1)
        soc_vector[0] = self.initial_soc

        for t in range(self.time_steps):
            power = power_vector[t]
            if power >= 0:
                # 放电
                energy_change = -power * self.time_interval / (self.discharge_efficiency * self.capacity_kwh)
            else:
                # 充电
                energy_change = -power * self.charge_efficiency * self.time_interval / self.capacity_kwh
            soc_vector[t+1] = soc_vector[t] + energy_change
        return soc_vector

    def check_constraints(self, power_vector):
        """
        检查功率和SOC约束
        """
        soc_vector = self.update_soc(power_vector)
        if np.any(power_vector > self.max_power_kw) or np.any(power_vector < -self.max_power_kw):
            return True, "功率超出限制"
        if np.any(soc_vector < self.min_soc) or np.any(soc_vector > self.max_soc):
            return True, f"SOC超出范围[{self.min_soc},{self.max_soc}]"
        return False, "满足所有约束"

    def revenue_function(self, power_vector, electricity_price,
                         pv_power=None, wind_power=None, load=None, losses=None):
        """
        计算BESS的收益函数 - 修正版
        基于电池充放电行为和电价差计算收益，而非整个系统的能量平衡
        """
        soc_vector = self.update_soc(power_vector)
        total_revenue = 0

        for t in range(self.time_steps):
            bess_power = power_vector[t]

            # 充电时(负功率)的成本
            if bess_power < 0:  # 充电
                charging_power = -bess_power  # 转为正值
                charging_cost = charging_power * electricity_price[t] * self.time_interval
                total_revenue -= charging_cost

            # 放电时(正功率)的收益
            else:  # 放电
                discharging_power = bess_power
                # 放电售电收益 - 假设放电电价等于购电电价
                discharging_revenue = discharging_power * electricity_price[t] * self.time_interval
                total_revenue += discharging_revenue

            # 计算电池循环退化成本 - 基于更合理的模型
            # 假设: 电池成本$250/kWh, 4000次循环寿命
            cycle_depth_equivalent = abs(bess_power) * self.time_interval / self.capacity_kwh
            cycle_cost_per_kwh = 250 / 4000  # 每kWh每完整循环的成本
            degradation_cost = cycle_depth_equivalent * self.capacity_kwh * cycle_cost_per_kwh

            # 考虑效率损失但不重复计算退化成本
            # 充放电过程中的能量损失已经在充放电计算中体现

            total_revenue -= degradation_cost

        return total_revenue

    def constraint_violations(self, power_vector):
        """
        若需要在优化器中对违反程度加惩罚
        """
        soc_vector = self.update_soc(power_vector)
        violations = []

        # 功率上下限
        violations.append(np.maximum(0, power_vector - self.max_power_kw))
        violations.append(np.maximum(0, -self.max_power_kw - power_vector))

        # SOC上下限
        violations.append(np.maximum(0, soc_vector - self.max_soc))
        violations.append(np.maximum(0, self.min_soc - soc_vector))

        return np.concatenate(violations)
