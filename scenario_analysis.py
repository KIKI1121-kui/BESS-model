# scenario_analysis.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_loader import load_and_preprocess_data
from forecasting import PVPowerPrediction, WindPowerPrediction
from bess_model import BESSModel
from system_model import ACPowerNetworkModel
from optimizer import BESSOptimizer
from economic_analysis import EconomicAnalysis, print_economic_results, plot_economic_comparison
from wind_data_generator import generate_wind_speed_data

# 🌟 用于增加峰谷电价差的函数
def adjust_electricity_price(price_data):
    """增加峰谷电价差"""
    price_data_copy = price_data.copy()

    if 'price' in price_data_copy.columns:
        # 创建时间索引
        if 'timestamp' in price_data_copy.columns:
            timestamps = pd.to_datetime(price_data_copy['timestamp'])
            hours = timestamps.dt.hour
        else:
            hours = np.arange(len(price_data_copy)) % 24

        # 增加峰谷电价差
        for i, hour in enumerate(hours):
            if 8 <= hour <= 11 or 18 <= hour <= 21:  # 峰时段
                price_data_copy.loc[i, 'price'] *= 1.5  # 提高50%
            elif 0 <= hour <= 5:  # 谷时段
                price_data_copy.loc[i, 'price'] *= 0.6  # 降低40%

    return price_data_copy

class ScenarioAnalysis:
    """
    场景分析类 - 基于不同天气参数构建场景并执行经济分析
    """

    def __init__(self, weather_file, price_file, wind_file=None, load_file=None):
        """
        初始化场景分析
        参数:
            weather_file: 天气数据文件路径
            price_file: 电价数据文件路径
            wind_file: 风速数据文件路径(若无将自动生成)
            load_file: 负荷数据文件路径
        """
        self.weather_file = weather_file
        self.price_file = price_file
        self.load_file = load_file

        # 如果未提供风速数据文件，则自动生成
        if wind_file is None or not os.path.exists(wind_file):
            print("未找到风速数据文件，自动生成...")
            self.wind_data = generate_wind_speed_data(weather_file, "风速数据.csv")
            self.wind_file = "风速数据.csv"
        else:
            self.wind_file = wind_file
            self.wind_data = pd.read_csv(wind_file)

        # 加载基准情景数据
        self.base_weather_data, self.base_price_data, self.base_load_data, _ = load_and_preprocess_data(
            self.weather_file, self.price_file, self.load_file, None
        )

        # 🌟 初始化经济分析对象 - 使用优化后的参数
        self.economic_analyzer = EconomicAnalysis(
            capital_cost=180000,  # 降低初始投资成本
            capacity_kwh=200,
            project_lifetime=15,
            discount_rate=0.08,
            o_m_cost_percent=0.015,  # 降低运维成本
            replacement_cost_percent=0.5,  # 降低更换成本
            replacement_year=10,
            degradation_rate=0.03
        )

        # 保存优化结果
        self.optimization_results = {}
        self.economic_metrics = []

        # 时间步数
        self.time_steps = min(90, len(self.base_price_data))

        print(f"初始化场景分析 - 使用 {weather_file}, {price_file}, {self.wind_file}")

    def create_scenario_data(self, scenario_name, modifications):
        """
        基于基准数据创建修改后的场景数据
        参数:
            scenario_name: 场景名称
            modifications: 修改参数的字典，可包含以下键:
                - 'solar_radiation': 调整系数或(min_adjust, max_adjust)元组
                - 'temperature': 调整系数或(min_adjust, max_adjust)元组
                - 'precipitation': 调整系数或(min_adjust, max_adjust)元组
                - 'wind_speed': 调整系数或(min_adjust, max_adjust)元组
        返回:
            modified_weather_data: 修改后的天气数据
            wind_data: 修改后的风速数据
        """
        print(f"\n创建场景 '{scenario_name}'...")

        # 复制基准天气数据
        modified_weather_data = self.base_weather_data.copy()
        wind_data = pd.read_csv(self.wind_file)

        # 应用修改
        for param, adjust in modifications.items():
            if param == 'solar_radiation' and param in modified_weather_data.columns:
                if isinstance(adjust, tuple):
                    min_adj, max_adj = adjust
                    # 对高值和低值应用不同的调整
                    threshold = modified_weather_data[param].mean()
                    high_mask = modified_weather_data[param] > threshold
                    low_mask = ~high_mask
                    modified_weather_data.loc[high_mask, param] *= max_adj
                    modified_weather_data.loc[low_mask, param] *= min_adj
                    print(f"  调整 {param}: 高值 x{max_adj}, 低值 x{min_adj}")
                else:
                    modified_weather_data[param] *= adjust
                    print(f"  调整 {param}: x{adjust}")

            elif param == 'temperature' and param in modified_weather_data.columns:
                if isinstance(adjust, tuple):
                    min_adj, max_adj = adjust
                    # 对温度应用绝对调整
                    threshold = modified_weather_data[param].mean()
                    high_mask = modified_weather_data[param] > threshold
                    low_mask = ~high_mask
                    modified_weather_data.loc[high_mask, param] += max_adj
                    modified_weather_data.loc[low_mask, param] += min_adj
                    print(f"  调整 {param}: 高值 +{max_adj}°C, 低值 +{min_adj}°C")
                else:
                    modified_weather_data[param] += adjust
                    print(f"  调整 {param}: +{adjust}°C")

            elif param == 'precipitation' and param in modified_weather_data.columns:
                if isinstance(adjust, tuple):
                    min_adj, max_adj = adjust
                    # 对降水应用系数调整，保持零值
                    precip_mask = modified_weather_data[param] > 0
                    threshold = modified_weather_data.loc[precip_mask, param].mean() if precip_mask.any() else 0
                    high_mask = (modified_weather_data[param] > threshold) & precip_mask
                    low_mask = (modified_weather_data[param] <= threshold) & precip_mask
                    modified_weather_data.loc[high_mask, param] *= max_adj
                    modified_weather_data.loc[low_mask, param] *= min_adj
                    print(f"  调整 {param}: 高值 x{max_adj}, 低值 x{min_adj}")
                else:
                    # 只调整非零值
                    precip_mask = modified_weather_data[param] > 0
                    modified_weather_data.loc[precip_mask, param] *= adjust
                    print(f"  调整 {param}: x{adjust} (仅非零值)")

            elif param == 'wind_speed' and 'wind_speed' in wind_data.columns:
                if isinstance(adjust, tuple):
                    min_adj, max_adj = adjust
                    # 对风速应用系数调整
                    threshold = wind_data['wind_speed'].mean()
                    high_mask = wind_data['wind_speed'] > threshold
                    low_mask = ~high_mask
                    wind_data.loc[high_mask, 'wind_speed'] *= max_adj
                    wind_data.loc[low_mask, 'wind_speed'] *= min_adj
                    print(f"  调整 {param}: 高值 x{max_adj}, 低值 x{min_adj}")
                else:
                    wind_data['wind_speed'] *= adjust
                    print(f"  调整 {param}: x{adjust}")

        # 确保数据有效性
        if 'solar_radiation' in modified_weather_data.columns:
            modified_weather_data['solar_radiation'] = np.clip(modified_weather_data['solar_radiation'], 0, 1200)
        if 'temperature' in modified_weather_data.columns:
            modified_weather_data['temperature'] = np.clip(modified_weather_data['temperature'], -20, 50)
        if 'precipitation' in modified_weather_data.columns:
            modified_weather_data['precipitation'] = np.clip(modified_weather_data['precipitation'], 0, 100)
        if 'wind_speed' in wind_data.columns:
            wind_data['wind_speed'] = np.clip(wind_data['wind_speed'], 0, 30)

        # 保存修改后的场景数据
        scenario_weather_file = f"scenario_{scenario_name}_weather.csv"
        scenario_wind_file = f"scenario_{scenario_name}_wind.csv"
        modified_weather_data.to_csv(scenario_weather_file, index=False)
        wind_data.to_csv(scenario_wind_file, index=False)

        print(f"  场景数据已保存: {scenario_weather_file}, {scenario_wind_file}")

        return modified_weather_data, wind_data, scenario_weather_file, scenario_wind_file

    def run_scenario(self, scenario_name, weather_data, price_data, load_data, wind_data):
        """
        运行特定场景的BESS优化和经济分析
        参数:
            scenario_name: 场景名称
            weather_data: 天气数据DataFrame
            price_data: 电价数据DataFrame
            load_data: 负荷数据DataFrame
            wind_data: 风速数据DataFrame
        返回:
            optimization_result: 优化结果
            economic_metrics: 经济指标
        """
        print(f"\n运行场景 '{scenario_name}'...")

        # 🌟 调整电价增加峰谷差
        price_data = adjust_electricity_price(price_data)

        # 提取电价
        if 'price' in price_data.columns:
            electricity_price = price_data['price'].values[:self.time_steps]
        else:
            # 尝试找到第一个数值列
            for col in price_data.columns:
                if pd.api.types.is_numeric_dtype(price_data[col]) and col != 'timestamp':
                    electricity_price = price_data[col].values[:self.time_steps]
                    print(f"  使用 '{col}' 列作为电价数据")
                    break

        # 提取负荷
        if load_data is not None and 'load_kw' in load_data.columns:
            load_values = load_data['load_kw'].values[:self.time_steps]
        elif load_data is not None:
            # 尝试找到第一个数值列
            for col in load_data.columns:
                if col != 'timestamp' and pd.api.types.is_numeric_dtype(load_data[col]):
                    load_values = load_data[col].values[:self.time_steps]
                    print(f"  使用 '{col}' 列作为负荷数据")
                    break
        else:
            load_values = None

        # 初始化预测模型
        pv_model = PVPowerPrediction(weather_data)
        pv_simulated = pv_model.simulate_pv_power(capacity_kw=100)
        pv_model.train_prediction_model(pv_simulated)
        pv_prediction = pv_model.predict_with_uncertainty()

        # 添加风速到天气数据
        if 'wind_speed' not in weather_data.columns and 'wind_speed' in wind_data.columns:
            # 创建一个临时的合并数据框
            temp_weather = weather_data.copy()
            # 如果两个数据框长度不一致，采用最小长度
            min_len = min(len(temp_weather), len(wind_data))
            temp_weather = temp_weather.iloc[:min_len].copy()
            temp_weather['wind_speed'] = wind_data['wind_speed'].values[:min_len]
            wind_model = WindPowerPrediction(temp_weather)
        else:
            wind_model = WindPowerPrediction(weather_data, wind_data)

        wind_prediction = wind_model.predict_wind_power()

        # 初始化电网模型
        network_model = ACPowerNetworkModel(num_buses=4)

        # 初始化BESS模型
        bess_model = BESSModel(
            time_steps=self.time_steps,
            initial_soc=0.5,
            capacity_kwh=200,
            max_power_kw=50,
            min_soc=0.1,
            max_soc=0.9,
            efficiency=0.95
        )

        # 初始化优化器并执行优化
        optimizer = BESSOptimizer(
            bess_model,
            electricity_price,
            load_values,
            pv_prediction,
            wind_prediction,
            network_model
        )

        optimization_result = optimizer.optimize()

        # 执行经济分析
        economic_metrics = self.economic_analyzer.analyze_scenario(
            optimization_result, scenario_name
        )

        # 保存结果
        self.optimization_results[scenario_name] = optimization_result
        self.economic_metrics.append(economic_metrics)

        # 输出优化结果摘要
        print(f"\n==== {scenario_name} 优化结果摘要 ====")
        print(f"优化状态: {optimization_result['success']}")
        print(f"收益: ${optimization_result['revenue']:.2f}")
        print(f"初始SOC: {optimization_result['soc_profile'][0]:.2f}")
        print(f"最终SOC: {optimization_result['soc_profile'][-1]:.2f}")
        print(f"最大充电功率: {abs(min(optimization_result['optimal_power'])):.2f} kW")
        print(f"最大放电功率: {max(optimization_result['optimal_power']):.2f} kW")

        return optimization_result, economic_metrics

    def run_all_scenarios(self):
        """
        运行所有预定义的场景
        """
        # 定义场景参数
        scenarios = {
            'baseline': {},  # 基准场景
            'high_solar': {'solar_radiation': 1.3},  # 太阳辐射增强30%
            'low_solar': {'solar_radiation': 0.7},  # 太阳辐射减弱30%
            'high_temp': {'temperature': 5},  # 温度升高5°C
            'low_temp': {'temperature': -5},  # 温度降低5°C
            'high_precip': {'precipitation': 2.0},  # 降水量翻倍
            'low_precip': {'precipitation': 0.5},  # 降水量减半
            'high_wind': {'wind_speed': 1.5},  # 风速增强50%
            'low_wind': {'wind_speed': 0.6},  # 风速减弱40%
            'worst_case': {  # 不利组合情景
                'solar_radiation': 0.6,
                'temperature': 8,
                'precipitation': 2.5,
                'wind_speed': 0.5
            }
        }

        print("\n开始运行所有场景分析...")

        # 运行基准场景
        print("\n运行基准场景...")
        base_optimization, base_economics = self.run_scenario(
            'baseline',
            self.base_weather_data,
            self.base_price_data,
            self.base_load_data,
            pd.read_csv(self.wind_file)
        )

        # 运行其他场景
        for scenario_name, modifications in scenarios.items():
            if scenario_name == 'baseline':
                continue  # 已运行基准场景

            # 创建场景数据
            modified_weather, modified_wind, _, _ = self.create_scenario_data(scenario_name, modifications)

            # 运行场景
            self.run_scenario(
                scenario_name,
                modified_weather,
                self.base_price_data,
                self.base_load_data,
                modified_wind
            )

        # 打印和可视化所有场景的经济分析结果
        print_economic_results(self.economic_metrics)
        plot_economic_comparison(self.economic_metrics)

        # 返回所有结果
        return self.optimization_results, self.economic_metrics


def run_multi_scenario_analysis():
    """
    执行多场景经济性分析的主函数
    """
    print("开始执行多场景BESS经济性分析...")

    # 检查是否已有风速数据文件，如果没有则生成
    wind_file = "风速数据.csv"
    if not os.path.exists(wind_file):
        generate_wind_speed_data("处理后的天气数据.csv", wind_file)

    # 初始化场景分析
    scenario_analyzer = ScenarioAnalysis(
        weather_file="处理后的天气数据.csv",
        price_file="电价数据.csv",
        wind_file=wind_file
    )

    # 运行所有场景
    optimization_results, economic_metrics = scenario_analyzer.run_all_scenarios()

    print("\n多场景分析完成。")
    return scenario_analyzer


if __name__ == "__main__":
    run_multi_scenario_analysis()