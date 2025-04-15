# main.py

import os
import numpy as np
import pandas as pd
import time

from data_loader import load_and_preprocess_data
from forecasting import PVPowerPrediction, WindPowerPrediction
from system_model import ACPowerNetworkModel
from bess_model import BESSModel
from optimizer import BESSOptimizer
from visualization import visualize_results


def main():
    """
    主函数：整合各模块执行BESS优化调度
    """
    print("开始执行BESS优化调度程序...")
    start_time = time.time()  # 添加时间统计

    # ========== 1. 数据加载与预处理 ==========
    weather_file = "处理后的天气数据.csv"
    price_file = "电价数据.csv"
    load_file = None  # 如果有真实负荷数据文件，可指定路径
    wind_file = None  # 如果有真实风电数据文件，可指定路径

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
                print(f"使用 '{col}' 列作为电价数据")
                break
        if electricity_price is None:
            raise ValueError("电价数据不可用，请检查 price_file CSV 结构")

    # 提取负荷
    if load_data is not None:
        if 'load_kw' in load_data.columns:
            load_values = load_data['load_kw'].values
        else:
            # 找一个数值列来当负荷
            load_values = None
            for col in load_data.columns:
                if col != 'timestamp' and pd.api.types.is_numeric_dtype(load_data[col]):
                    load_values = load_data[col].values
                    print(f"使用 '{col}' 作为负荷数据")
                    break
    else:
        load_values = np.zeros(len(electricity_price))

    # 设置优化时间范围 - 修改为可配置的时间范围
    optimization_horizon = 90  # 可以调整为更长时间

    # 只取前optimization_horizon时段进行优化
    electricity_price = electricity_price[:optimization_horizon]
    load_values = load_values[:optimization_horizon] if load_values is not None else None
    time_steps = len(electricity_price)
    if isinstance(time_indices, pd.Series):
        time_indices = time_indices.iloc[:time_steps]
    else:
        time_indices = time_indices[:time_steps]

    # ========== 2. 光伏与风电预测 ==========
    pv_model = PVPowerPrediction(weather_data)
    pv_simulated = pv_model.simulate_pv_power(capacity_kw=100)
    pv_model.train_prediction_model(pv_simulated)
    pv_prediction = pv_model.predict_with_uncertainty()

    wind_model = WindPowerPrediction(weather_data, wind_data)
    wind_prediction = wind_model.predict_wind_power()

    # ========== 3. 构建电力系统模型 ==========
    network_model = ACPowerNetworkModel(num_buses=4)

    # 提前计算敏感度系数，用于优化
    ptdf_matrix, vsf_matrix = network_model.calculate_sensitivity_factors()

    # ========== 4. 创建BESS模型 ==========
    bess_model = BESSModel(
        time_steps=time_steps,
        initial_soc=0.5,
        capacity_kwh=200,
        max_power_kw=50,
        min_soc=0.1,
        max_soc=0.9,
        efficiency=0.95
    )

    # ========== 5. 优化求解 ==========
    optimizer = BESSOptimizer(
        bess_model,
        electricity_price,
        load_values,
        pv_prediction,
        wind_prediction,
        network_model
    )

    # 使用新的两阶段优化方法
    print("\n开始使用两阶段方法进行BESS优化...")
    optimization_result = optimizer.solve_with_two_stage_approach()

    # ========== 6. 可视化与结果输出 ==========
    if time_indices is None or len(time_indices) < time_steps:
        time_indices = np.arange(time_steps)

    visualize_results(
        time_indices,
        optimization_result,
        electricity_price,
        load_values,
        pv_prediction,
        wind_prediction,
        ac_power_flow=True
    )

    end_time = time.time()
    total_time = end_time - start_time

    print("\n==== Optimization Results Summary ====")
    print(f"Optimization Status: {optimization_result['success']}")
    print(f"Revenue: ${optimization_result['revenue']:.2f}")
    print(f"Initial SOC: {optimization_result['soc_profile'][0]:.2f}")
    print(f"Final SOC: {optimization_result['soc_profile'][-1]:.2f}")
    print(f"Max Charging Power: {abs(min(optimization_result['optimal_power'])):.2f} kW")
    print(f"Max Discharging Power: {max(optimization_result['optimal_power']):.2f} kW")
    print(f"Total Computation Time: {total_time:.2f} seconds")

    if 'losses_kw' in optimization_result:
        avg_loss = np.mean(optimization_result['losses_kw'])
        max_loss = np.max(optimization_result['losses_kw'])
        total_loss = np.sum(optimization_result['losses_kw'])
        print(f"Average Line Loss: {avg_loss:.2f} kW")
        print(f"Maximum Line Loss: {max_loss:.2f} kW")
        print(f"Total Energy Loss: {total_loss:.2f} kWh")

    # 潮流结果
    if 'ac_power_flow_results' in optimization_result:
        flow_violations = sum(1 for res in optimization_result['ac_power_flow_results'] if res['violations'])
        convergence_failures = sum(1 for res in optimization_result['ac_power_flow_results'] if not res['success'])
        print(f"\n==== AC Power Flow Results ====")
        print(f"Time steps with flow violations: {flow_violations} / {time_steps}")
        print(f"Time steps with convergence failures: {convergence_failures} / {time_steps}")

        # 输出违反约束的统计
        if flow_violations > 0:
            violation_stats = {
                'line_overload': {'high': 0, 'medium': 0, 'low': 0},
                'voltage_violation': {'high': 0, 'medium': 0, 'low': 0}
            }

            for t, res in enumerate(optimization_result['ac_power_flow_results']):
                if res['violations']:
                    for v_id, v_info in res['violations'].items():
                        v_type = v_info['type']
                        if 'severity' in v_info:
                            severity = v_info['severity']
                            violation_stats[v_type][severity] += 1

            print("Violation statistics:")
            for v_type, stats in violation_stats.items():
                print(f"  {v_type.replace('_', ' ').title()}:")
                for severity, count in stats.items():
                    if count > 0:
                        print(f"    {severity.title()} severity: {count}")

            print("\nDetailed violations (up to 5 examples):")
            violation_count = 0
            for t, res in enumerate(optimization_result['ac_power_flow_results']):
                if res['violations'] and violation_count < 5:
                    print(f"  Time step {t}:")
                    for v_id, v_info in res['violations'].items():
                        if 'severity' in v_info and v_info['severity'] in ['high', 'medium']:
                            print(f"    {v_info['name']} ({v_info['type']}): ", end="")
                            if v_info['type'] == 'line_overload':
                                print(f"Loading = {v_info['loading_percent']:.2f}%")
                            elif v_info['type'] == 'voltage_violation':
                                print(f"Voltage = {v_info['vm_pu']:.4f} p.u.")
                            violation_count += 1
                            if violation_count >= 5:
                                break

    # 保存结果到CSV
    results_data = {
        'Time': time_indices[:len(optimization_result['optimal_power'])],
        'Battery_Power_kW': optimization_result['optimal_power'],
        'SOC': optimization_result['soc_profile'][:-1],  # 去掉最后一个使长度匹配
        'Electricity_Price': electricity_price,
        'PV_Power_kW': pv_prediction['prediction'],
        'Wind_Power_kW': wind_prediction['prediction'],
        'Load_kW': load_values
    }
    if 'losses_kw' in optimization_result:
        results_data['Line_Losses_kW'] = optimization_result['losses_kw']
    if 'ac_power_flow_results' in optimization_result:
        grid_power = [res['grid_power_kw'] if res['success'] else np.nan
                      for res in optimization_result['ac_power_flow_results']]
        results_data['Grid_Power_kW'] = grid_power
        # 节点电压
        if (optimization_result['ac_power_flow_results'][0]['success'] and
                optimization_result['ac_power_flow_results'][0]['voltage_profiles']):
            bus_indices = list(optimization_result['ac_power_flow_results'][0]['voltage_profiles'].keys())
            for bus_idx in bus_indices:
                voltages = []
                for res in optimization_result['ac_power_flow_results']:
                    if res['success'] and res['voltage_profiles']:
                        voltages.append(res['voltage_profiles'][bus_idx])
                    else:
                        voltages.append(np.nan)
                results_data[f'Voltage_Bus_{bus_idx}_pu'] = voltages
        # 线路负载率
        if (optimization_result['ac_power_flow_results'][0]['success'] and
                optimization_result['ac_power_flow_results'][0]['line_loadings']):
            line_indices = list(optimization_result['ac_power_flow_results'][0]['line_loadings'].keys())
            for line_idx in line_indices:
                loadings = []
                for res in optimization_result['ac_power_flow_results']:
                    if res['success'] and res['line_loadings']:
                        loadings.append(res['line_loadings'][line_idx])
                    else:
                        loadings.append(np.nan)
                results_data[f'Line_{line_idx}_Loading_Percent'] = loadings

    results_df = pd.DataFrame(results_data)
    output_file = f'BESS_Optimization_Results_{optimization_horizon}steps.csv'
    results_df.to_csv(output_file, index=False)
    print(f"已将结果保存到: '{output_file}'")


if __name__ == "__main__":
    main()