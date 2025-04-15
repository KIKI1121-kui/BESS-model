# visualization.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def visualize_results(time_indices, optimization_result, electricity_price, load_data,
                      pv_prediction=None, wind_prediction=None, ac_power_flow=True):
    """
    可视化优化结果
    """
    plt.rcParams['font.sans-serif'] = ['SimHei','Arial Unicode MS','Microsoft YaHei','SimSun']
    plt.rcParams['axes.unicode_minus'] = False

    # 确保 time_indices 可以画图
    if isinstance(time_indices[0], str):
        try:
            time_indices = pd.to_datetime(time_indices)
        except:
            time_indices = np.arange(len(electricity_price))

    if ac_power_flow:
        fig = plt.figure(figsize=(15, 24))
        gs = fig.add_gridspec(7, 1)
    else:
        fig = plt.figure(figsize=(15, 18))
        gs = fig.add_gridspec(5, 1)

    # 1) 充放电功率 & 电价
    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(time_indices[:len(optimization_result['optimal_power'])],
             optimization_result['optimal_power'], 'b-', label='Battery Power (kW)')
    ax1_2 = ax1.twinx()
    ax1_2.plot(time_indices[:len(electricity_price)], electricity_price, 'r--', label='Electricity Price')

    ax1.set_ylabel('Power (kW)')
    ax1_2.set_ylabel('Price ($/kWh)')
    ax1.set_title('BESS Charging/Discharging Power and Electricity Price')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # 2) SOC
    ax2 = fig.add_subplot(gs[1,0])
    soc_profile = optimization_result['soc_profile']
    # 跳过初始SOC,只画对应时段
    if len(soc_profile) > len(time_indices):
        soc_to_plot = soc_profile[1:]
    else:
        soc_to_plot = soc_profile
    plot_length = min(len(time_indices), len(soc_to_plot))
    ax2.plot(time_indices[:plot_length], soc_to_plot[:plot_length], 'g-', label='SOC')
    ax2.axhline(y=0.1, color='r', linestyle='--', label='Min SOC')
    ax2.axhline(y=0.9, color='r', linestyle='--', label='Max SOC')
    ax2.set_ylabel('SOC')
    ax2.set_title('BESS State of Charge (SOC)')
    ax2.legend()

    # 3) 收益(示例:简易累积曲线)
    ax3 = fig.add_subplot(gs[2,0])
    cumulative_revenue = np.zeros(len(optimization_result['optimal_power']))
    for t in range(len(optimization_result['optimal_power'])):
        if t == 0:
            cumulative_revenue[t] = electricity_price[t]*optimization_result['optimal_power'][t]
        else:
            cumulative_revenue[t] = cumulative_revenue[t-1] + \
                                    electricity_price[t]*optimization_result['optimal_power'][t]
    ax3.plot(time_indices[:len(cumulative_revenue)], cumulative_revenue, color='purple', label='Cumulative Revenue')
    ax3.set_ylabel('Revenue ($)')
    ax3.set_title('BESS Cumulative Revenue')
    ax3.legend()

    # 4) 可再生发电 & 负荷
    ax4 = fig.add_subplot(gs[3,0])
    if pv_prediction is not None:
        ax4.plot(time_indices[:len(pv_prediction['prediction'])],
                 pv_prediction['prediction'], 'orange', label='PV Power')
    if wind_prediction is not None:
        ax4.plot(time_indices[:len(wind_prediction['prediction'])],
                 wind_prediction['prediction'], 'c', label='Wind Power')
    if load_data is not None:
        ax4.plot(time_indices[:len(load_data)], load_data, 'k--', label='Load')
    ax4.set_ylabel('Power (kW)')
    ax4.set_title('Renewable Generation and Load')
    ax4.legend()

    # 5) 线路损耗
    ax5 = fig.add_subplot(gs[4,0])
    if 'losses_kw' in optimization_result:
        ax5.plot(time_indices[:len(optimization_result['losses_kw'])],
                 optimization_result['losses_kw'], 'r-', label='Line Losses (kW)')
        avg_loss = np.mean(optimization_result['losses_kw'])
        ax5.axhline(y=avg_loss, color='k', linestyle='--', label=f'Avg Loss: {avg_loss:.2f} kW')
        ax5.set_ylabel('Power Loss (kW)')
        ax5.set_title('Power System Line Losses')
        ax5.legend()

    # 6) 电网注入功率 & 潮流违约标记(若有AC结果)
    if ac_power_flow and 'ac_power_flow_results' in optimization_result:
        ax6 = fig.add_subplot(gs[5,0])
        ac_flow_res = optimization_result['ac_power_flow_results']
        grid_power = [res['grid_power_kw'] if res['success'] else 0 for res in ac_flow_res]
        ax6.plot(time_indices[:len(grid_power)], grid_power, 'b-', label='Grid Power (kW)')

        violation_times = []
        for t, res in enumerate(ac_flow_res):
            if res['violations']:
                violation_times.append(t)
        if violation_times:
            for t in violation_times:
                ax6.axvline(x=time_indices[t], color='r', linestyle='--', alpha=0.3)
            ax6.scatter([time_indices[t] for t in violation_times],
                        [grid_power[t] for t in violation_times],
                        color='r', marker='x', label='AC Power Flow Violations')
        ax6.set_ylabel('Power (kW)')
        ax6.set_title('Grid Interaction with AC Power Flow')
        ax6.legend()

        # 7) 节点电压
        ax7 = fig.add_subplot(gs[6,0])
        if ac_flow_res and ac_flow_res[0]['success'] and ac_flow_res[0]['voltage_profiles']:
            bus_indices = list(ac_flow_res[0]['voltage_profiles'].keys())
            voltage_data = {f'Bus {b}': [] for b in bus_indices}

            for t, res in enumerate(ac_flow_res):
                if res['success'] and res['voltage_profiles']:
                    for b in bus_indices:
                        voltage_data[f'Bus {b}'].append(res['voltage_profiles'][b])
                else:
                    for b in bus_indices:
                        voltage_data[f'Bus {b}'].append(np.nan)

            line_styles = ['-', '--', '-.', ':']
            markers = ['o', 's', '^', 'd']
            colors = ['blue', 'green', 'red', 'purple']
            offset = 0.0005

            for i, (bus_name, voltages) in enumerate(voltage_data.items()):
                adjusted_voltages = np.array(voltages) + i*offset
                ax7.plot(time_indices[:len(voltages)], adjusted_voltages,
                         linestyle=line_styles[i % len(line_styles)],
                         color=colors[i % len(colors)],
                         marker=markers[i % len(markers)],
                         markersize=4, markevery=10,
                         label=f'{bus_name} (offset:+{i*offset:.4f})')
            ax7.axhline(y=1.05, color='red', linestyle='--', label='Upper Limit (1.05 p.u.)')
            ax7.axhline(y=0.95, color='red', linestyle='--', label='Lower Limit (0.95 p.u.)')

            for i in range(len(bus_indices)):
                ax7.axhline(y=1.0 + i*offset, color=colors[i % len(colors)],
                            linestyle=':', alpha=0.3)

            min_v = min([min(v) for v in voltage_data.values() if len(v)>0]) - 0.001
            max_v = max([max(v) for v in voltage_data.values() if len(v)>0]) + 0.001 + len(bus_indices)*offset
            if max_v - min_v < 0.01:
                mean_v = (max_v + min_v)/2
                min_v = mean_v - 0.005
                max_v = mean_v + 0.005 + len(bus_indices)*offset
            ax7.set_ylim(min_v, max_v)
            ax7.text(0.02, 0.02, f"注：各母线电压添加了{offset:.4f}的偏移", transform=ax7.transAxes, fontsize=8)

            ax7.set_ylabel('Voltage (p.u.)')
            ax7.set_xlabel('Time')
            ax7.set_title('Bus Voltage Profiles (with offset)')
            ax7.legend(loc='best', fontsize=8)
            ax7.grid(True, alpha=0.3)
    else:
        # 若不包含AC潮流图，此处可省略
        pass

    plt.tight_layout()
    if ac_power_flow:
        plt.savefig('BESS_Optimization_Results_with_AC_Power_Flow.png', dpi=300)
        print("Optimization results chart saved: 'BESS_Optimization_Results_with_AC_Power_Flow.png'")
    else:
        plt.savefig('BESS_Optimization_Results.png', dpi=300)
        print("Optimization results chart saved: 'BESS_Optimization_Results.png'")
    plt.show()
