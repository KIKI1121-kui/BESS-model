# multi_scenario_main.py (优化版)

import os
import numpy as np
import pandas as pd
import time
import concurrent.futures
from functools import partial
import matplotlib.pyplot as plt

from data_loader import load_and_preprocess_data
from forecasting import PVPowerPrediction, WindPowerPrediction
from system_model import ACPowerNetworkModel
from bess_model import BESSModel
from optimizer import BESSOptimizer
from economic_analysis import EconomicAnalysis, print_economic_results, plot_economic_comparison
from wind_data_generator import generate_wind_speed_data


def run_single_scenario(scenario, base_files, time_steps=90):
    """
    运行单个场景的BESS优化与经济性分析

    参数:
        scenario: 场景参数字典
        base_files: 基础文件路径字典
        time_steps: 时间步长

    返回:
        结果字典
    """
    scenario_name = scenario.get('name', 'unknown')
    print(f"运行场景: {scenario_name}")
    start_time = time.time()

    try:
        # 1. 加载数据
        weather_file = base_files.get('weather_file', "处理后的天气数据.csv")
        price_file = base_files.get('price_file', "电价数据.csv")
        load_file = base_files.get('load_file', None)
        wind_file = base_files.get('wind_file', None)

        weather_data, price_data, load_data, wind_data = load_and_preprocess_data(
            weather_file, price_file, load_file, wind_file
        )

        # 提取时间戳和电价
        time_indices = price_data['timestamp'] if 'timestamp' in price_data.columns else np.arange(len(price_data))
        if 'price' in price_data.columns:
            electricity_price = price_data['price'].values
        else:
            # 如没有price列,尝试拿数值列做电价
            electricity_price = None
            for col in price_data.columns:
                if pd.api.types.is_numeric_dtype(price_data[col]):
                    electricity_price = price_data[col].values
                    break
            if electricity_price is None:
                raise ValueError("电价数据不可用")

        # 如果场景中有电价系数，应用到电价数据
        if 'price_factor' in scenario:
            electricity_price = electricity_price * scenario['price_factor']

        # 提取负荷
        if load_data is not None:
            if 'load_kw' in load_data.columns:
                load_values = load_data['load_kw'].values
            else:
                load_values = None
                for col in load_data.columns:
                    if col != 'timestamp' and pd.api.types.is_numeric_dtype(load_data[col]):
                        load_values = load_data[col].values
                        break
        else:
            load_values = np.zeros(len(electricity_price))

        # 如果场景中有负荷系数，应用到负荷数据
        if 'load_factor' in scenario:
            load_values = load_values * scenario['load_factor']

        # 截取指定长度
        electricity_price = electricity_price[:time_steps]
        load_values = load_values[:time_steps] if load_values is not None else None
        actual_time_steps = len(electricity_price)
        if isinstance(time_indices, pd.Series):
            time_indices = time_indices.iloc[:actual_time_steps]
        else:
            time_indices = time_indices[:actual_time_steps]

        # 2. 光伏与风电预测
        pv_model = PVPowerPrediction(weather_data)
        pv_capacity = scenario.get('pv_capacity', 100)
        pv_simulated = pv_model.simulate_pv_power(capacity_kw=pv_capacity)
        pv_model.train_prediction_model(pv_simulated)
        pv_prediction = pv_model.predict_with_uncertainty()

        # 应用场景中的可再生能源因子
        if 'pv_factor' in scenario:
            pv_prediction['prediction'] = pv_prediction['prediction'] * scenario['pv_factor']
            pv_prediction['lower_bound'] = pv_prediction['lower_bound'] * scenario['pv_factor']
            pv_prediction['upper_bound'] = pv_prediction['upper_bound'] * scenario['pv_factor']

        wind_model = WindPowerPrediction(weather_data, wind_data)
        wind_prediction = wind_model.predict_wind_power()

        # 应用场景中的风电因子
        if 'wind_factor' in scenario:
            wind_prediction['prediction'] = wind_prediction['prediction'] * scenario['wind_factor']
            wind_prediction['lower_bound'] = wind_prediction['lower_bound'] * scenario['wind_factor']
            wind_prediction['upper_bound'] = wind_prediction['upper_bound'] * scenario['wind_factor']

        # 3. 构建电力系统模型
        network_model = ACPowerNetworkModel(num_buses=4)

        # 4. 创建BESS模型
        bess_model = BESSModel(
            time_steps=actual_time_steps,
            initial_soc=scenario.get('initial_soc', 0.5),
            capacity_kwh=scenario.get('capacity_kwh', 200),
            max_power_kw=scenario.get('max_power_kw', 50),
            min_soc=scenario.get('min_soc', 0.1),
            max_soc=scenario.get('max_soc', 0.9),
            efficiency=scenario.get('efficiency', 0.95)
        )

        # 5. 优化求解
        optimizer = BESSOptimizer(
            bess_model,
            electricity_price,
            load_values,
            pv_prediction,
            wind_prediction,
            network_model
        )

        # 使用两阶段优化方法
        optimization_result = optimizer.solve_with_two_stage_approach()

        # 6. 经济性分析
        economic_analyzer = EconomicAnalysis(
            capital_cost=scenario.get('capital_cost', 250000),
            capacity_kwh=scenario.get('capacity_kwh', 200),
            project_lifetime=scenario.get('project_lifetime', 15),
            discount_rate=scenario.get('discount_rate', 0.08),
            o_m_cost_percent=scenario.get('o_m_cost_percent', 0.02),
            replacement_cost_percent=scenario.get('replacement_cost_percent', 0.6),
            replacement_year=scenario.get('replacement_year', 10),
            degradation_rate=scenario.get('degradation_rate', 0.03),
            electricity_buy_price=np.mean(electricity_price)
        )

        economic_metrics = economic_analyzer.analyze_scenario(optimization_result, scenario_name)

        # 计算总耗时
        total_time = time.time() - start_time

        # 输出场景结果
        print(f"场景 {scenario_name} 完成! 耗时: {total_time:.2f}秒")
        print(
            f"收益: ${economic_metrics['annual_revenue']:.2f}, LCOS: ${economic_metrics['lcos']:.4f}/kWh, NPV: ${economic_metrics['npv']:.2f}")

        return {
            'scenario': scenario,
            'scenario_name': scenario_name,
            'optimization_result': optimization_result,
            'economic_metrics': economic_metrics,
            'execution_time': total_time
        }

    except Exception as e:
        print(f"场景 {scenario_name} 执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'scenario': scenario,
            'scenario_name': scenario_name,
            'error': str(e)
        }


def run_multi_scenario_analysis(parallel=True, max_workers=4):
    """
    使用并行计算运行多场景分析

    参数:
        parallel: 是否使用并行计算
        max_workers: 最大并行进程数

    返回:
        economic_metrics_list: 所有场景的经济指标列表
    """
    print("\n=== 开始多场景分析 ===")
    overall_start_time = time.time()

    # 基础文件
    base_files = {
        'weather_file': "处理后的天气数据.csv",
        'price_file': "电价数据.csv",
        'load_file': None,
        'wind_file': None
    }

    # 定义场景参数
    scenarios = [
        {"name": "baseline", "capacity_kwh": 200, "max_power_kw": 50, "efficiency": 0.95},
        {"name": "high_solar", "capacity_kwh": 200, "max_power_kw": 50, "efficiency": 0.95, "pv_factor": 1.5},
        {"name": "low_solar", "capacity_kwh": 200, "max_power_kw": 50, "efficiency": 0.95, "pv_factor": 0.5},
        {"name": "high_temp", "capacity_kwh": 200, "max_power_kw": 50, "efficiency": 0.95, "degradation_rate": 0.05},
        {"name": "low_temp", "capacity_kwh": 200, "max_power_kw": 50, "efficiency": 0.98, "degradation_rate": 0.02},
        {"name": "high_price", "capacity_kwh": 200, "max_power_kw": 50, "efficiency": 0.95, "price_factor": 1.3},
        {"name": "low_price", "capacity_kwh": 200, "max_power_kw": 50, "efficiency": 0.95, "price_factor": 0.7},
        {"name": "high_wind", "capacity_kwh": 200, "max_power_kw": 50, "efficiency": 0.95, "wind_factor": 1.5},
        {"name": "low_wind", "capacity_kwh": 200, "max_power_kw": 50, "efficiency": 0.95, "wind_factor": 0.5},
        {"name": "worst_case", "capacity_kwh": 200, "max_power_kw": 50, "efficiency": 0.90, "price_factor": 0.7,
         "pv_factor": 0.5, "wind_factor": 0.5, "degradation_rate": 0.05}
    ]

    # 运行所有场景
    results = []

    if parallel:
        print(f"使用并行计算 (最大进程数: {max_workers})")
        run_scenario_with_base_files = partial(run_single_scenario, base_files=base_files)

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_scenario_with_base_files, scenario) for scenario in scenarios]

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"场景执行异常: {str(e)}")
    else:
        print("使用顺序计算")
        for scenario in scenarios:
            result = run_single_scenario(scenario, base_files)
            results.append(result)

    # 收集所有场景的经济指标
    economic_metrics_list = []
    successful_scenarios = 0

    for result in results:
        if 'error' not in result and 'economic_metrics' in result:
            economic_metrics_list.append(result['economic_metrics'])
            successful_scenarios += 1

    # 按场景名称排序
    economic_metrics_list.sort(key=lambda x: x['scenario'])

    # 打印经济分析结果
    print_economic_results(economic_metrics_list)

    # 绘制经济指标对比图
    plot_economic_comparison(economic_metrics_list)

    # 计算总耗时
    total_time = time.time() - overall_start_time

    print(f"\n多场景分析完成! 总耗时: {total_time:.2f}秒")
    print(f"成功运行场景数: {successful_scenarios}/{len(scenarios)}")

    return economic_metrics_list


def main():
    """
    主函数：运行BESS多场景优化和经济性分析
    """
    print("==================================================")
    print("      BESS多场景仿真与经济性分析程序 (优化版)")
    print("==================================================")

    # 确认数据文件是否存在
    required_files = ["处理后的天气数据.csv", "电价数据.csv"]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"错误: 以下必要文件不存在: {', '.join(missing_files)}")
        return

    # 检查并生成风速数据文件
    wind_file = "风速数据.csv"
    if not os.path.exists(wind_file):
        print("风速数据文件不存在，自动生成中...")
        generate_wind_speed_data("处理后的天气数据.csv", wind_file)
    else:
        print(f"使用已存在的风速数据文件: {wind_file}")

    # 运行多场景分析 (开启并行)
    start_time = time.time()
    try:
        # 确定最佳进程数 (CPU核心数 - 1，最小为2)
        import multiprocessing
        num_cpus = multiprocessing.cpu_count()
        optimal_workers = max(min(num_cpus - 1, 8), 2)  # 最多8个进程，至少2个进程

        print(f"检测到 {num_cpus} 个CPU核心，将使用 {optimal_workers} 个并行进程")
        economic_metrics = run_multi_scenario_analysis(parallel=True, max_workers=optimal_workers)
    except Exception as e:
        print(f"多场景分析出错: {str(e)}")
        print("尝试使用顺序执行...")
        economic_metrics = run_multi_scenario_analysis(parallel=False)

    total_time = time.time() - start_time
    print(f"\n程序总耗时: {total_time:.2f}秒")
    print("\n分析结果已保存到'经济性指标对比优化.png'")
    print("程序执行完毕。")


if __name__ == "__main__":
    main()