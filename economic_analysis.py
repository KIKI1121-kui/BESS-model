# economic_analysis.py (修改版)

import numpy as np
import numpy_financial as npf
import time
import matplotlib.pyplot as plt



class EconomicAnalysis:
    """
    BESS系统经济性分析模块 - 修改版
    增加了向量化计算和缓存机制，提高计算效率
    修正了收益计算逻辑，正确反映电池充放电的经济价值
    """

    def __init__(self, capital_cost=250000, capacity_kwh=200, project_lifetime=15,
                 discount_rate=0.08, o_m_cost_percent=0.02, replacement_cost_percent=0.6,
                 replacement_year=10, degradation_rate=0.03, electricity_buy_price=None,
                 electricity_sell_price=None):
        """
        初始化经济分析模块
        参数:
            capital_cost: 初始投资成本 ($)
            capacity_kwh: BESS容量 (kWh)
            project_lifetime: 项目生命周期 (年)
            discount_rate: 折现率
            o_m_cost_percent: 年运维成本占投资的比例
            replacement_cost_percent: 更换成本占初始投资的比例
            replacement_year: 需要更换的年份
            degradation_rate: 每年电池容量衰减率
            electricity_buy_price: 电价购入向量或标量
            electricity_sell_price: 电价售出向量或标量
        """
        self.capital_cost = capital_cost
        self.capacity_kwh = capacity_kwh
        self.project_lifetime = project_lifetime
        self.discount_rate = discount_rate
        self.o_m_cost_percent = o_m_cost_percent
        self.replacement_cost_percent = replacement_cost_percent
        self.replacement_year = replacement_year
        self.degradation_rate = degradation_rate

        self.electricity_buy_price = electricity_buy_price
        self.electricity_sell_price = electricity_sell_price if electricity_sell_price is not None else (
            0.8 * electricity_buy_price if electricity_buy_price is not None else None)

        self.annual_o_m_cost = self.capital_cost * self.o_m_cost_percent
        self.cycles_per_year = 365  # 假设每天一个循环

        # 每kWh的投资成本
        self.capital_cost_per_kwh = self.capital_cost / self.capacity_kwh

        # 缓存机制
        self._cache = {}

        # 预计算常用值
        self._years = np.arange(1, self.project_lifetime + 1)
        self._discount_factors = np.power(1 + self.discount_rate, -self._years)
        self._degradation_factors = np.power(1 - self.degradation_rate, self._years - 1)

        # 创建替换成本掩码
        self._replacement_mask = (self._years == self.replacement_year)

        print(
            f"初始化经济分析模块 - 初始投资: ${capital_cost}, 容量: {capacity_kwh}kWh, 折现率: {discount_rate * 100}%")

    def _get_cache_key(self, **kwargs):
        """生成缓存键"""
        # 将可能的浮点数和numpy数组转换为可哈希类型
        processed_items = []
        for k, v in sorted(kwargs.items()):
            if isinstance(v, np.ndarray):
                v = hash(v.tobytes())
            elif isinstance(v, float):
                v = round(v, 6)  # 限制精度以提高命中率
            processed_items.append((k, v))
        return hash(frozenset(processed_items))

    def calculate_lcos(self, annual_energy_throughput):
        """
        计算平准化储能成本 (Levelized Cost of Storage) - 优化版
        使用向量化操作代替循环，显著提高计算速度

        参数:
            annual_energy_throughput: 年度能量传输量 (kWh)
        返回:
            lcos: 平准化储能成本 ($/kWh)
        """
        # 使用缓存检查
        cache_key = self._get_cache_key(throughput=annual_energy_throughput)
        if cache_key in self._cache.get('lcos', {}):
            return self._cache.get('lcos')[cache_key]

        # 计算有效吞吐量
        effective_throughputs = annual_energy_throughput * self._degradation_factors
        discounted_throughputs = effective_throughputs * self._discount_factors
        total_energy = np.sum(discounted_throughputs)

        # 计算成本
        annual_o_m_costs = np.ones_like(self._years) * self.annual_o_m_cost
        replacement_costs = np.zeros_like(self._years, dtype=float)
        replacement_costs[self._replacement_mask] = self.capital_cost * self.replacement_cost_percent

        total_discounted_costs = np.sum(annual_o_m_costs * self._discount_factors) + \
                                 np.sum(replacement_costs * self._discount_factors) + \
                                 self.capital_cost

        lcos = total_discounted_costs / total_energy if total_energy > 0 else float('inf')

        # 存入缓存
        if 'lcos' not in self._cache:
            self._cache['lcos'] = {}
        self._cache['lcos'][cache_key] = lcos

        return lcos

    def calculate_npv(self, annual_revenue):
        """
        计算净现值 (Net Present Value) - 优化版
        使用向量化操作代替循环

        参数:
            annual_revenue: 每年收益 ($) 或者年度收益列表
        返回:
            npv: 净现值 ($)
        """
        # 使用缓存检查
        cache_key = self._get_cache_key(revenue=annual_revenue)
        if cache_key in self._cache.get('npv', {}):
            return self._cache.get('npv')[cache_key]

        if isinstance(annual_revenue, (int, float)):
            # 如果是标量，转换为统一形式
            annual_revenues = np.ones(self.project_lifetime) * annual_revenue
        else:
            # 如果是列表或数组，确保长度匹配
            if len(annual_revenue) < self.project_lifetime:
                # 如果年度收益列表短于项目生命周期，用最后一年的收益填充剩余年份
                temp_revenues = np.zeros(self.project_lifetime)
                temp_revenues[:len(annual_revenue)] = annual_revenue
                temp_revenues[len(annual_revenue):] = annual_revenue[-1]
                annual_revenues = temp_revenues
            else:
                annual_revenues = np.array(annual_revenue[:self.project_lifetime])

        # 考虑电池衰减导致的收益下降
        annual_revenues = annual_revenues * self._degradation_factors

        # 计算运维成本
        o_m_costs = np.ones(self.project_lifetime) * self.annual_o_m_cost

        # 计算更换成本
        replacement_costs = np.zeros(self.project_lifetime)
        if self.replacement_year <= self.project_lifetime:
            replacement_costs[self.replacement_year - 1] = self.capital_cost * self.replacement_cost_percent

        # 计算净现金流
        net_cash_flows = annual_revenues - o_m_costs - replacement_costs

        # 添加初始投资
        cash_flows = np.zeros(self.project_lifetime + 1)
        cash_flows[0] = -self.capital_cost
        cash_flows[1:] = net_cash_flows

        # 计算NPV
        npv = npf.npv(self.discount_rate, cash_flows)

        # 存入缓存
        if 'npv' not in self._cache:
            self._cache['npv'] = {}
        self._cache['npv'][cache_key] = npv

        return npv

    def calculate_irr(self, annual_revenue):
        """
        计算内部收益率 (Internal Rate of Return) - 优化版
        使用向量化操作代替循环

        参数:
            annual_revenue: 每年收益 ($) 或者年度收益列表
        返回:
            irr: 内部收益率
        """
        # 使用缓存检查
        cache_key = self._get_cache_key(revenue=annual_revenue)
        if cache_key in self._cache.get('irr', {}):
            return self._cache.get('irr')[cache_key]

        if isinstance(annual_revenue, (int, float)):
            # 如果是标量，转换为统一形式
            annual_revenues = np.ones(self.project_lifetime) * annual_revenue
        else:
            # 如果是列表或数组，确保长度匹配
            if len(annual_revenue) < self.project_lifetime:
                temp_revenues = np.zeros(self.project_lifetime)
                temp_revenues[:len(annual_revenue)] = annual_revenue
                temp_revenues[len(annual_revenue):] = annual_revenue[-1]
                annual_revenues = temp_revenues
            else:
                annual_revenues = np.array(annual_revenue[:self.project_lifetime])

        # 考虑电池衰减导致的收益下降
        annual_revenues = annual_revenues * self._degradation_factors

        # 计算运维成本
        o_m_costs = np.ones(self.project_lifetime) * self.annual_o_m_cost

        # 计算更换成本
        replacement_costs = np.zeros(self.project_lifetime)
        if self.replacement_year <= self.project_lifetime:
            replacement_costs[self.replacement_year - 1] = self.capital_cost * self.replacement_cost_percent

        # 计算净现金流
        net_cash_flows = annual_revenues - o_m_costs - replacement_costs

        # 添加初始投资
        cash_flows = np.zeros(self.project_lifetime + 1)
        cash_flows[0] = -self.capital_cost
        cash_flows[1:] = net_cash_flows

        try:
            irr = npf.irr(cash_flows)
            if np.isnan(irr):
                print("警告: IRR计算结果为NaN，可能是现金流全为负或格式不正确")
                irr = None
        except Exception as e:
            print(f"IRR计算出错: {e}")
            irr = None

        # 存入缓存
        if 'irr' not in self._cache:
            self._cache['irr'] = {}
        self._cache['irr'][cache_key] = irr

        return irr

    def calculate_payback_period(self, annual_revenue):
        """
        计算投资回收期 - 优化版
        使用向量化操作提高效率

        参数:
            annual_revenue: 每年收益 ($) 或者年度收益列表
        返回:
            payback_period: 投资回收期 (年)
        """
        # 使用缓存检查
        cache_key = self._get_cache_key(revenue=annual_revenue)
        if cache_key in self._cache.get('payback', {}):
            return self._cache.get('payback')[cache_key]

        if isinstance(annual_revenue, (int, float)):
            # 如果是标量，转换为统一形式
            annual_revenues = np.ones(self.project_lifetime) * annual_revenue
        else:
            # 如果是列表或数组，确保长度匹配
            if len(annual_revenue) < self.project_lifetime:
                temp_revenues = np.zeros(self.project_lifetime)
                temp_revenues[:len(annual_revenue)] = annual_revenue
                temp_revenues[len(annual_revenue):] = annual_revenue[-1]
                annual_revenues = temp_revenues
            else:
                annual_revenues = np.array(annual_revenue[:self.project_lifetime])

        # 考虑电池衰减导致的收益下降
        annual_revenues = annual_revenues * self._degradation_factors

        # 计算运维成本
        o_m_costs = np.ones(self.project_lifetime) * self.annual_o_m_cost

        # 计算更换成本
        replacement_costs = np.zeros(self.project_lifetime)
        if self.replacement_year <= self.project_lifetime:
            replacement_costs[self.replacement_year - 1] = self.capital_cost * self.replacement_cost_percent

        # 计算净现金流
        net_cash_flows = annual_revenues - o_m_costs - replacement_costs

        # 计算累积现金流
        cumulative_cash_flows = np.zeros(self.project_lifetime + 1)
        cumulative_cash_flows[0] = -self.capital_cost
        cumulative_cash_flows[1:] = np.cumsum(net_cash_flows)

        # 找到累积现金流转正的点
        positive_indices = np.where(cumulative_cash_flows >= 0)[0]
        if len(positive_indices) > 0:
            first_positive_idx = positive_indices[0]
            if first_positive_idx > 0:
                # 插值计算精确的回收期
                prev_cf = cumulative_cash_flows[first_positive_idx - 1]
                current_cf = cumulative_cash_flows[first_positive_idx]
                if current_cf != prev_cf:  # 避免除以零
                    fraction = abs(prev_cf) / (current_cf - prev_cf) if current_cf != prev_cf else 0
                    payback_period = first_positive_idx - 1 + fraction
                else:
                    payback_period = first_positive_idx
            else:
                payback_period = 0  # 首期就回收
        else:
            payback_period = float('inf')  # 在项目周期内无法回收

        # 存入缓存
        if 'payback' not in self._cache:
            self._cache['payback'] = {}
        self._cache['payback'][cache_key] = payback_period

        return payback_period

    def calculate_revenue_from_optimization(self, optimization_result, days_per_year=365):
        """
        基于优化结果计算年度收益 - 修改版
        正确处理90天模拟窗口的年化，并调整额外收益计算

        参数:
            optimization_result: 优化结果字典
            days_per_year: 一年中的运行天数
        返回:
            annual_revenue: 年度收益 ($)
        """
        # 使用缓存检查
        cache_key = self._get_cache_key(
            result=optimization_result['revenue'] if 'revenue' in optimization_result else 0,
            days=days_per_year)
        if cache_key in self._cache.get('annual_revenue', {}):
            return self._cache.get('annual_revenue')[cache_key]

        # 从优化结果中提取收益
        if 'revenue' in optimization_result:
            # 计算实际模拟天数
            time_steps = len(optimization_result['optimal_power'])
            hours_per_day = 24
            actual_days = time_steps / hours_per_day

            # 提取模拟期间的总收益并转换为每日收益
            simulation_revenue = optimization_result['revenue']
            daily_revenue = simulation_revenue / actual_days

            print(f"调试信息 - 模拟天数: {actual_days:.1f}, "
                  f"模拟总收益: ${simulation_revenue:.2f}, "
                  f"每日收益: ${daily_revenue:.2f}")

            # 计算BESS的功率容量（用于额外收益估计）
            power_capacity_kw = self.capacity_kwh * 0.5  # 假设功率/能量比为0.5

            # 额外收益计算（每日基准）
            # 1. 辅助服务收益 - 假设每kW每年$50的收益，其中50%容量可用于辅助服务
            daily_ancillary_service = power_capacity_kw * 0.5 * 50 / 365  # 每天

            # 2. 需求响应收益 - 假设每月5次，每kW每次$5，90%容量可用
            daily_demand_response = (5 * power_capacity_kw * 0.9 * 5) / 30  # 每天

            # 3. 容量市场收益 - 假设每kW每年$40，80%容量可认证
            daily_capacity_market = power_capacity_kw * 0.8 * 40 / 365  # 每天

            # 4. 电网延缓投资收益 - 固定每kW每年$30的收益
            daily_grid_deferral = power_capacity_kw * 30 / 365  # 每天

            # 合并所有每日收益
            additional_daily_revenues = np.array([
                daily_ancillary_service,
                daily_demand_response,
                daily_capacity_market,
                daily_grid_deferral
            ])

            # 打印额外收益明细（调试用）
            print(f"调试信息 - 每日额外收益明细: "
                  f"辅助服务=${daily_ancillary_service:.2f}, "
                  f"需求响应=${daily_demand_response:.2f}, "
                  f"容量市场=${daily_capacity_market:.2f}, "
                  f"电网延缓=${daily_grid_deferral:.2f}")

            total_daily_revenue = daily_revenue + np.sum(additional_daily_revenues)
            print(f"调试信息 - 总每日收益: ${total_daily_revenue:.2f}")

            # 年化（使用实际的days_per_year参数）
            annual_revenue = total_daily_revenue * days_per_year
            print(f"调试信息 - 年化收益(基于{days_per_year}天): ${annual_revenue:.2f}")

            # 存入缓存
            if 'annual_revenue' not in self._cache:
                self._cache['annual_revenue'] = {}
            self._cache['annual_revenue'][cache_key] = annual_revenue

            return annual_revenue
        else:
            print("优化结果中没有收益信息")
            return 0

    def calculate_annual_energy_throughput(self, optimization_result, days_per_year=365):
        """
        计算年度能量传输量 - 优化版
        适配90天仿真窗口的年化计算

        参数:
            optimization_result: 优化结果字典
            days_per_year: 一年中的运行天数
        返回:
            annual_throughput: 年度能量传输量 (kWh)
        """
        # 使用缓存检查
        if 'optimal_power' not in optimization_result:
            return 0

        cache_key = self._get_cache_key(power=np.array(optimization_result['optimal_power']).sum(),
                                        days=days_per_year)
        if cache_key in self._cache.get('throughput', {}):
            return self._cache.get('throughput')[cache_key]

        optimal_power = np.array(optimization_result['optimal_power'])

        # 使用向量化操作计算充放电能量
        charging_mask = optimal_power < 0
        discharging_mask = optimal_power > 0

        charging_energy = -np.sum(optimal_power[charging_mask])  # 充电功率为负值
        discharging_energy = np.sum(optimal_power[discharging_mask])  # 放电功率为正值

        # 总能量吞吐量是充电和放电能量的总和
        daily_throughput = charging_energy + discharging_energy

        # 计算实际模拟天数
        time_steps = len(optimal_power)
        hours_per_day = 24
        actual_days = time_steps / hours_per_day

        # 计算模拟期间每日平均吞吐量
        daily_avg_throughput = daily_throughput / actual_days

        # 放大到年度能量吞吐量
        annual_throughput = daily_avg_throughput * days_per_year

        # 调试信息
        print(f"调试信息 - 吞吐量: 充电={charging_energy:.2f}kWh, "
              f"放电={discharging_energy:.2f}kWh, "
              f"总吞吐量={daily_throughput:.2f}kWh")
        print(f"调试信息 - 实际模拟天数: {actual_days:.1f}, "
              f"每日平均吞吐量: {daily_avg_throughput:.2f}kWh, "
              f"年化吞吐量: {annual_throughput:.2f}kWh")

        # 存入缓存
        if 'throughput' not in self._cache:
            self._cache['throughput'] = {}
        self._cache['throughput'][cache_key] = annual_throughput

        return annual_throughput

    def analyze_scenario(self, optimization_result, scenario_name, days_per_year=365):
        """
        分析特定场景的经济性指标 - 优化版
        使用缓存机制避免重复计算

        参数:
            optimization_result: 优化结果字典
            scenario_name: 场景名称
            days_per_year: 一年中的运行天数
        返回:
            economic_metrics: 包含经济指标的字典
        """
        # 使用缓存检查
        if 'revenue' in optimization_result:
            cache_key = self._get_cache_key(scenario=scenario_name,
                                            revenue=optimization_result['revenue'],
                                            days=days_per_year)
            if cache_key in self._cache.get('metrics', {}):
                return self._cache.get('metrics')[cache_key]

        # 计时
        start_time = time.time()

        # 打印场景信息
        print(f"\n===== 分析场景: {scenario_name} =====")

        # 计算各项指标
        annual_revenue = self.calculate_revenue_from_optimization(optimization_result, days_per_year)
        annual_throughput = self.calculate_annual_energy_throughput(optimization_result, days_per_year)

        lcos = self.calculate_lcos(annual_throughput)
        npv = self.calculate_npv(annual_revenue)
        irr = self.calculate_irr(annual_revenue)
        payback = self.calculate_payback_period(annual_revenue)

        # 收集经济指标结果
        economic_metrics = {
            'scenario': scenario_name,
            'annual_revenue': annual_revenue,
            'annual_throughput': annual_throughput,
            'lcos': lcos,
            'npv': npv,
            'irr': irr if irr is not None else float('nan'),
            'payback_period': payback,
            'computation_time': time.time() - start_time
        }

        # 打印主要结果
        print(f"场景分析完成: {scenario_name}")
        print(f"年度收益: ${annual_revenue:.2f}")
        print(f"年度吞吐量: {annual_throughput:.2f}kWh")
        print(f"LCOS: ${lcos:.4f}/kWh")
        print(f"NPV: ${npv:.2f}")
        print(f"IRR: {economic_metrics['irr'] * 100:.2f}%" if not np.isnan(economic_metrics['irr']) else "IRR: N/A")
        print(f"回收期: {payback:.2f}年" if payback != float('inf') else "回收期: >项目寿命")

        # 存入缓存
        if 'metrics' not in self._cache:
            self._cache['metrics'] = {}
        if 'revenue' in optimization_result:
            self._cache['metrics'][cache_key] = economic_metrics

        return economic_metrics

    def clear_cache(self):
        """清除所有缓存"""
        self._cache = {}
        return True


def print_economic_results(economic_metrics):
    """
    打印经济分析结果 - 优化版
    参数:
        economic_metrics: 经济指标字典或字典列表
    """
    if isinstance(economic_metrics, list):
        # 多场景结果
        print("\n============= 经济性分析结果比较 =============")
        print(
            f"{'场景':<15}{'年收益($)':<15}{'LCOS($/kWh)':<15}{'NPV($)':<15}{'IRR(%)':<15}{'回收期(年)':<15}{'计算时间(秒)':<15}")
        print("-" * 105)

        for metrics in economic_metrics:
            irr_str = f"{metrics['irr'] * 100:.2f}" if not np.isnan(metrics['irr']) else "N/A"
            payback_str = f"{metrics['payback_period']:.2f}" if metrics['payback_period'] != float('inf') else ">项目周期"
            comp_time = metrics.get('computation_time', 0)

            print(f"{metrics['scenario']:<15}{metrics['annual_revenue']:.2f}{'':>5}"
                  f"{metrics['lcos']:.4f}{'':>7}{metrics['npv']:.2f}{'':>5}"
                  f"{irr_str}{'':>10}{payback_str}{'':>5}{comp_time:.4f}{'':>8}")
    else:
        # 单场景结果
        print("\n============= 经济性分析结果 =============")
        print(f"场景: {economic_metrics['scenario']}")
        print(f"年度收益: ${economic_metrics['annual_revenue']:.2f}")
        print(f"年度能量吞吐量: {economic_metrics['annual_throughput']:.2f} kWh")
        print(f"平准化储能成本 (LCOS): ${economic_metrics['lcos']:.4f}/kWh")
        print(f"净现值 (NPV): ${economic_metrics['npv']:.2f}")

        if not np.isnan(economic_metrics['irr']):
            print(f"内部收益率 (IRR): {economic_metrics['irr'] * 100:.2f}%")
        else:
            print("内部收益率 (IRR): 无法计算")

        if economic_metrics['payback_period'] != float('inf'):
            print(f"投资回收期: {economic_metrics['payback_period']:.2f} 年")
        else:
            print("投资回收期: 超过项目周期")

        comp_time = economic_metrics.get('computation_time', 0)
        print(f"计算耗时: {comp_time:.4f} 秒")


def plot_economic_comparison(economic_metrics):
    """
    绘制经济指标对比图 - 折线图版本
    参数:
        economic_metrics: 经济指标字典列表
    """
    if not isinstance(economic_metrics, list) or len(economic_metrics) < 2:
        print("需要至少两个场景才能进行比较")
        return

    # 提取数据
    scenarios = [m['scenario'] for m in economic_metrics]
    annual_revenues = [m['annual_revenue'] for m in economic_metrics]
    lcos_values = [m['lcos'] for m in economic_metrics]
    npv_values = [m['npv'] for m in economic_metrics]
    irr_values = [m['irr'] * 100 if not np.isnan(m['irr']) else 0 for m in economic_metrics]
    payback_values = [min(m['payback_period'], 20) for m in economic_metrics]  # 限制最大值为20年

    # 创建图表 - 使用子图布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 年度收益对比 - 折线图
    ax1 = axes[0, 0]
    ax1.plot(scenarios, annual_revenues, 'o-', color='blue', linewidth=2, markersize=8)
    ax1.set_title('年度收益对比', fontsize=12)
    ax1.set_ylabel('年度收益 ($)', fontsize=10)
    ax1.tick_params(axis='x', rotation=45, labelsize=9)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 添加数据标签
    for i, revenue in enumerate(annual_revenues):
        ax1.annotate(f'${revenue:.0f}',
                     (scenarios[i], revenue),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=8)

    # 2. LCOS对比 - 折线图
    ax2 = axes[0, 1]
    ax2.plot(scenarios, lcos_values, 'o-', color='green', linewidth=2, markersize=8)
    ax2.set_title('平准化储能成本(LCOS)对比', fontsize=12)
    ax2.set_ylabel('LCOS ($/kWh)', fontsize=10)
    ax2.tick_params(axis='x', rotation=45, labelsize=9)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 添加数据标签
    for i, lcos in enumerate(lcos_values):
        ax2.annotate(f'${lcos:.3f}',
                     (scenarios[i], lcos),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=8)

    # 3. NPV对比 - 折线图
    ax3 = axes[1, 0]
    ax3.plot(scenarios, npv_values, 'o-', color='red', linewidth=2, markersize=8)
    ax3.set_title('净现值(NPV)对比', fontsize=12)
    ax3.set_ylabel('NPV ($)', fontsize=10)
    ax3.tick_params(axis='x', rotation=45, labelsize=9)
    ax3.grid(True, linestyle='--', alpha=0.7)

    # 添加数据标签
    for i, npv in enumerate(npv_values):
        ax3.annotate(f'${npv:.0f}',
                     (scenarios[i], npv),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=8)

    # 4. IRR和回收期对比 - 双Y轴折线图
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()

    # IRR线
    line1, = ax4.plot(scenarios, irr_values, 'o-', color='blue', linewidth=2, markersize=8, label='IRR (%)')
    # 回收期线
    line2, = ax4_twin.plot(scenarios, payback_values, 's-', color='red', linewidth=2, markersize=8, label='回收期 (年)')

    ax4.set_title('IRR和回收期对比', fontsize=12)
    ax4.set_ylabel('IRR (%)', fontsize=10, color='blue')
    ax4_twin.set_ylabel('回收期 (年)', fontsize=10, color='red')
    ax4.tick_params(axis='x', rotation=45, labelsize=9)
    ax4.tick_params(axis='y', colors='blue')
    ax4_twin.tick_params(axis='y', colors='red')
    ax4.grid(True, linestyle='--', alpha=0.7)

    # 添加IRR数据标签
    for i, irr in enumerate(irr_values):
        ax4.annotate(f'{irr:.1f}%',
                     (scenarios[i], irr),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=8,
                     color='blue')

    # 添加回收期数据标签
    for i, payback in enumerate(payback_values):
        label = f'{payback:.1f}年' if payback < 20 else '>20年'
        ax4_twin.annotate(label,
                          (scenarios[i], payback),
                          textcoords="offset points",
                          xytext=(0, -15),
                          ha='center',
                          fontsize=8,
                          color='red')

    # 创建合并的图例
    lines = [line1, line2]
    labels = ['IRR (%)', '回收期 (年)']
    ax4.legend(lines, labels, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig('经济性指标对比折线图.png', dpi=300)
    print("经济性分析结果图表已保存到: '经济性指标对比折线图.png'")
    plt.close()  # 关闭图表释放内存