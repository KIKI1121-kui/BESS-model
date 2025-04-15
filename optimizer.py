# optimizer.py (优化版)

import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize, linprog
import time
import pyomo.environ as pyo
import warnings

warnings.filterwarnings("ignore")


class BESSOptimizer:
    """
    改进的BESS优化调度求解器 - 使用分解方法处理交流潮流约束
    """

    def __init__(self, bess_model, electricity_price, load_data,
                 pv_prediction=None, wind_prediction=None, network_model=None):
        self.bess_model = bess_model
        self.electricity_price = electricity_price
        self.load_data = load_data
        self.pv_prediction = pv_prediction
        self.wind_prediction = wind_prediction
        self.network_model = network_model

        self.time_steps = bess_model.time_steps
        self.initial_guess = np.zeros(self.time_steps)
        self.violation_penalty = 1000
        self.losses = np.zeros(self.time_steps)

        # 缓存约束违反记录
        self.constraint_violations_history = []

        # 设置敏感度系数
        if network_model:
            # 计算线路功率传输分布因子(PTDF)和电压敏感度因子(VSF)
            self.ptdf_matrix, self.vsf_matrix = self._compute_sensitivity_factors()

            # 提前计算基础潮流结果，用于线性化
            self.base_power_flow = self._calculate_base_power_flow()

    def _compute_sensitivity_factors(self):
        """
        计算PTDF和VSF敏感度矩阵，用于线性化电力潮流模型
        """
        print("计算网络敏感度系数...")

        # 使用network_model计算敏感度因子
        ptdf_matrix, vsf_matrix = self.network_model.calculate_sensitivity_factors()

        return ptdf_matrix, vsf_matrix

    def _calculate_base_power_flow(self):
        """
        计算基础潮流状态，用于线性化
        """
        print("计算基础潮流...")
        base_flow = {}

        # 考虑没有BESS参与时的基础潮流
        for t in range(self.time_steps):
            load = self.load_data[t] if self.load_data is not None else 0
            pv_power = self.pv_prediction['prediction'][t] if self.pv_prediction else 0
            wind_power = self.wind_prediction['prediction'][t] if self.wind_prediction else 0
            bess_power = 0  # 不考虑BESS

            success, ac_results, _, _ = self.network_model.integrate_with_bess(
                load, pv_power, wind_power, bess_power
            )

            if success and ac_results:
                base_flow[t] = ac_results

        return base_flow

    def solve_with_two_stage_approach(self):
        """
        使用两阶段方法求解BESS优化问题:
        1. 第一阶段：使用线性化模型快速优化
        2. 第二阶段：检查AC约束，添加违约约束，重新优化
        """
        print("开始两阶段优化...")
        start_time = time.time()

        # 第一阶段：线性化模型的快速优化
        result_stage1 = self.optimize_with_dc_approximation()

        # 如果没有网络模型或不需要考虑AC约束，直接返回结果
        if not self.network_model:
            return result_stage1

        # 第二阶段：检查AC约束，添加违约切割面
        max_iterations = 5
        current_solution = result_stage1['optimal_power']
        current_revenue = result_stage1['revenue']
        best_solution = current_solution.copy()
        best_revenue = current_revenue
        best_violations = float('inf')

        # 跟踪违反约束的时段和类型
        violation_cuts = []

        for iteration in range(max_iterations):
            print(f"\n迭代 {iteration + 1}/{max_iterations}:")

            # 检查当前解的AC约束违反情况
            total_violations, violations_by_time = self.check_ac_constraints(current_solution)
            print(f"- 当前约束违反总数: {total_violations}")

            # 如果没有违反或比之前的解更好，更新最佳解
            if total_violations == 0:
                print("- 找到可行解，优化成功!")
                best_solution = current_solution.copy()
                best_revenue = current_revenue
                break
            elif total_violations < best_violations:
                best_violations = total_violations
                best_solution = current_solution.copy()
                best_revenue = current_revenue

            # 根据违反约束生成切割面约束
            new_cuts = self._generate_constraint_cuts(violations_by_time)
            violation_cuts.extend(new_cuts)

            # 使用新的切割面约束重新优化
            result = self.optimize_with_constraint_cuts(violation_cuts)

            if not result['success']:
                print("- 优化失败，使用之前的最佳解")
                break

            current_solution = result['optimal_power']
            current_revenue = result['revenue']

            # 检查收敛性
            if np.allclose(current_solution, best_solution, rtol=1e-3, atol=1e-3):
                print("- 解收敛，停止迭代")
                break

        # 使用最佳解计算最终结果
        final_result = self._prepare_final_result(best_solution, best_revenue)

        end_time = time.time()
        print(f"两阶段优化完成，耗时: {end_time - start_time:.2f}秒")

        return final_result

    def optimize_with_dc_approximation(self):
        """
        使用DC潮流近似的快速优化
        """
        print("执行线性化优化...")

        # 根据是否有网络模型选择不同的优化方法
        if self.network_model and self.ptdf_matrix is not None:
            # 使用线性化的网络约束进行优化
            return self.optimize_with_linear_network_model()
        else:
            # 如果没有网络模型，使用简单优化
            return self.optimize_simple()

    def optimize_with_linear_network_model(self):
        """
        使用线性化的网络模型进行优化 (PTDF方法)
        """
        try:
            # 创建优化模型
            model = pyo.ConcreteModel()

            # 定义变量
            model.time = pyo.RangeSet(0, self.time_steps - 1)
            model.bess_power = pyo.Var(model.time, domain=pyo.Reals,
                                       bounds=(-self.bess_model.max_power_kw, self.bess_model.max_power_kw))
            model.soc = pyo.Var(pyo.RangeSet(0, self.time_steps), domain=pyo.Reals,
                                bounds=(self.bess_model.min_soc, self.bess_model.max_soc))

            # 添加SOC初始值约束
            model.soc[0].fix(self.bess_model.initial_soc)

            # 添加SOC与功率关系约束
            def soc_rule(model, t):
                if model.bess_power[t] >= 0:  # 放电
                    energy_change = -model.bess_power[t] * 1.0 / (
                                self.bess_model.discharge_efficiency * self.bess_model.capacity_kwh)
                else:  # 充电
                    energy_change = -model.bess_power[
                        t] * self.bess_model.charge_efficiency * 1.0 / self.bess_model.capacity_kwh
                return model.soc[t + 1] == model.soc[t] + energy_change

            model.soc_constraint = pyo.Constraint(model.time, rule=soc_rule)

            # 添加线性化的网络约束 (使用PTDF)
            num_lines = len(self.ptdf_matrix)
            line_limits = self.network_model.get_line_limits()  # 从network_model获取线路限制

            def line_limit_rule(model, t, line_idx):
                # 计算可再生能源和负荷的净影响
                load = self.load_data[t] if self.load_data is not None else 0
                pv_power = self.pv_prediction['prediction'][t] if self.pv_prediction else 0
                wind_power = self.wind_prediction['prediction'][t] if self.wind_prediction else 0
                net_injection = pv_power + wind_power - load

                # 基础潮流
                base_flow = 0
                if t in self.base_power_flow and line_idx in self.base_power_flow[t]:
                    base_flow = self.base_power_flow[t][line_idx]

                # BESS对线路潮流的影响 (使用PTDF)
                bess_impact = self.ptdf_matrix[line_idx][self.bess_model.bess_bus] * model.bess_power[t]

                # 总潮流约束
                return base_flow + bess_impact <= line_limits[line_idx]

            # 只对有限制的线路添加约束
            for line_idx in range(num_lines):
                if line_idx in line_limits:
                    model.add_component(f'line_limit_{line_idx}',
                                        pyo.Constraint(model.time,
                                                       rule=lambda m, t, l=line_idx: line_limit_rule(m, t, l)))

            # 目标函数：最大化收益
            def objective_rule(model):
                total_revenue = 0
                for t in model.time:
                    bess_power = model.bess_power[t]
                    price = self.electricity_price[t]

                    # 充电时(负功率)的成本
                    if bess_power < 0:  # 充电
                        charging_power = -bess_power  # 转为正值
                        charging_cost = charging_power * price
                        total_revenue -= charging_cost

                    # 放电时(正功率)的收益
                    else:  # 放电
                        discharging_power = bess_power
                        discharging_revenue = discharging_power * price
                        total_revenue += discharging_revenue

                    # 计算电池循环退化成本
                    cycle_depth_equivalent = abs(bess_power) / self.bess_model.capacity_kwh
                    cycle_cost_per_kwh = 250 / 4000  # 每kWh每完整循环的成本
                    degradation_cost = cycle_depth_equivalent * self.bess_model.capacity_kwh * cycle_cost_per_kwh

                    total_revenue -= degradation_cost

                return total_revenue

            model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

            # 求解模型
            solver = pyo.SolverFactory('glpk')  # 使用GLPK求解器处理线性问题
            results = solver.solve(model, tee=False)

            # 处理结果
            if results.solver.status == pyo.SolverStatus.ok:
                # 提取优化结果
                optimal_power = np.array([model.bess_power[t].value for t in model.time])
                soc_profile = np.array([model.soc[t].value for t in range(self.time_steps + 1)])
                revenue = model.objective()

                return {
                    'success': True,
                    'message': '线性化优化成功',
                    'optimal_power': optimal_power,
                    'soc_profile': soc_profile,
                    'revenue': revenue,
                    'objective_value': -revenue,  # 保持与原来的接口一致
                    'num_iterations': 1,
                    'losses_kw': np.zeros(self.time_steps)  # 线性化模型不直接计算损耗
                }
            else:
                print("线性化优化失败，回退到简单优化...")
                return self.optimize_simple()

        except Exception as e:
            print(f"线性化优化出错: {e}")
            return self.optimize_simple()

    def optimize_simple(self):
        """
        使用原始的优化方法，但不考虑AC潮流约束
        """

        def obj_func(power_vector):
            # 计算收益
            revenue = self.bess_model.revenue_function(
                power_vector, self.electricity_price,
                pv_power=self.pv_prediction['prediction'] if self.pv_prediction else None,
                wind_power=self.wind_prediction['prediction'] if self.wind_prediction else None,
                load=self.load_data
            )
            return -revenue  # 最大化收益 => 最小化负收益

        # 构造约束
        constraints = []
        # 功率约束
        for t in range(self.time_steps):
            constraints.append({'type': 'ineq',
                                'fun': lambda x, idx=t: self.bess_model.max_power_kw - x[idx]})
            constraints.append({'type': 'ineq',
                                'fun': lambda x, idx=t: x[idx] + self.bess_model.max_power_kw})
        # SOC约束
        for t in range(1, self.time_steps + 1):
            def soc_upper_constraint(x, step=t):
                soc = self.bess_model.update_soc(x)
                return self.bess_model.max_soc - soc[step]

            def soc_lower_constraint(x, step=t):
                soc = self.bess_model.update_soc(x)
                return soc[step] - self.bess_model.min_soc

            constraints.append({'type': 'ineq', 'fun': soc_upper_constraint})
            constraints.append({'type': 'ineq', 'fun': soc_lower_constraint})

        # 进行优化
        result = minimize(
            obj_func, self.initial_guess,
            method='SLSQP', constraints=constraints,
            options={'maxiter': 100, 'disp': True}
        )

        final_soc = self.bess_model.update_soc(result.x)
        final_revenue = -obj_func(result.x)

        return {
            'success': result.success,
            'message': result.message,
            'optimal_power': result.x,
            'soc_profile': final_soc,
            'revenue': final_revenue,
            'objective_value': result.fun,
            'num_iterations': result.nit,
            'losses_kw': np.zeros(self.time_steps)  # 简单模型不直接计算损耗
        }

    def check_ac_constraints(self, power_vector):
        """
        检查给定功率向量的AC约束违反情况
        返回: 总违反数, 每个时间步的违反详情
        """
        print("检查AC潮流约束...")
        total_violations = 0
        violations_by_time = {}

        # 对每个时段执行潮流，记录潮流违反
        for t in tqdm(range(self.time_steps)):
            bess_power = power_vector[t]
            pv_power = self.pv_prediction['prediction'][t] if self.pv_prediction else 0
            wind_power = self.wind_prediction['prediction'][t] if self.wind_prediction else 0
            load = self.load_data[t] if self.load_data is not None else 0

            success, ac_results, violations, _ = self.network_model.integrate_with_bess(
                load, pv_power, wind_power, bess_power
            )

            if violations:
                violations_by_time[t] = violations
                total_violations += len(violations)

        return total_violations, violations_by_time

    def _generate_constraint_cuts(self, violations_by_time):
        """
        根据违反约束生成切割面约束
        """
        cuts = []

        for t, violations in violations_by_time.items():
            for v_id, v_info in violations.items():
                if v_info['type'] == 'line_overload':
                    # 针对线路过载生成切割面
                    line_idx = int(v_id.split('_')[1])
                    cut = {
                        'type': 'line_overload',
                        'time_step': t,
                        'element_idx': line_idx,
                        'current_value': v_info['loading_percent'],
                        'limit': 100.0,  # 100%负载率
                        'sensitivity': self.ptdf_matrix[line_idx] if hasattr(self, 'ptdf_matrix') else None
                    }
                    cuts.append(cut)

                elif v_info['type'] == 'voltage_violation':
                    # 针对电压越限生成切割面
                    bus_idx = int(v_id.split('_')[1])
                    voltage = v_info['vm_pu']
                    limit = 0.95 if voltage < 0.95 else 1.05
                    cut = {
                        'type': 'voltage_violation',
                        'time_step': t,
                        'element_idx': bus_idx,
                        'current_value': voltage,
                        'limit': limit,
                        'sensitivity': self.vsf_matrix[bus_idx] if hasattr(self, 'vsf_matrix') else None
                    }
                    cuts.append(cut)

        return cuts

    def optimize_with_constraint_cuts(self, cuts):
        """
        使用切割面约束重新优化
        """
        print(f"使用 {len(cuts)} 个切割面约束重新优化...")

        def obj_func(power_vector):
            # 计算收益
            revenue = self.bess_model.revenue_function(
                power_vector, self.electricity_price,
                pv_power=self.pv_prediction['prediction'] if self.pv_prediction else None,
                wind_power=self.wind_prediction['prediction'] if self.wind_prediction else None,
                load=self.load_data
            )

            # 添加切割面违反惩罚
            penalty = 0
            for cut in cuts:
                t = cut['time_step']
                bess_power = power_vector[t]

                if cut['type'] == 'line_overload' and cut['sensitivity'] is not None:
                    # 使用PTDF计算线路负载
                    line_idx = cut['element_idx']
                    sensitivity = cut['sensitivity'][self.bess_model.bess_bus]
                    load = self.load_data[t] if self.load_data is not None else 0
                    pv_power = self.pv_prediction['prediction'][t] if self.pv_prediction else 0
                    wind_power = self.wind_prediction['prediction'][t] if self.wind_prediction else 0

                    # 估计线路负载
                    base_load = cut['current_value'] - sensitivity * power_vector[t]
                    new_load = base_load + sensitivity * bess_power
                    if new_load > cut['limit']:
                        penalty += (new_load - cut['limit']) * self.violation_penalty

                elif cut['type'] == 'voltage_violation' and cut['sensitivity'] is not None:
                    # 使用VSF计算节点电压
                    bus_idx = cut['element_idx']
                    sensitivity = cut['sensitivity'][self.bess_model.bess_bus]
                    base_voltage = cut['current_value'] - sensitivity * power_vector[t]
                    new_voltage = base_voltage + sensitivity * bess_power

                    if cut['limit'] == 0.95 and new_voltage < cut['limit']:
                        penalty += (cut['limit'] - new_voltage) * self.violation_penalty * 100
                    elif cut['limit'] == 1.05 and new_voltage > cut['limit']:
                        penalty += (new_voltage - cut['limit']) * self.violation_penalty * 100

            return -(revenue - penalty)  # 最大化收益 => 最小化负收益

        # 构造约束
        constraints = []
        # 功率约束
        for t in range(self.time_steps):
            constraints.append({'type': 'ineq',
                                'fun': lambda x, idx=t: self.bess_model.max_power_kw - x[idx]})
            constraints.append({'type': 'ineq',
                                'fun': lambda x, idx=t: x[idx] + self.bess_model.max_power_kw})
        # SOC约束
        for t in range(1, self.time_steps + 1):
            def soc_upper_constraint(x, step=t):
                soc = self.bess_model.update_soc(x)
                return self.bess_model.max_soc - soc[step]

            def soc_lower_constraint(x, step=t):
                soc = self.bess_model.update_soc(x)
                return soc[step] - self.bess_model.min_soc

            constraints.append({'type': 'ineq', 'fun': soc_upper_constraint})
            constraints.append({'type': 'ineq', 'fun': soc_lower_constraint})

        # 添加显式线路约束 (使用线性近似)
        for cut in cuts:
            if cut['type'] == 'line_overload' and cut['sensitivity'] is not None:
                t = cut['time_step']
                line_idx = cut['element_idx']
                sensitivity = cut['sensitivity'][self.bess_model.bess_bus]

                def line_constraint(x, t=t, s=sensitivity, base=cut['current_value'], limit=cut['limit']):
                    # 检查t是否在约束范围内
                    if t >= len(x):
                        return 0
                    # 估计新的线路负载
                    new_load = base - s * self.initial_guess[t] + s * x[t]
                    return limit - new_load

                constraints.append({'type': 'ineq', 'fun': line_constraint})

        # 进行优化
        result = minimize(
            obj_func, self.initial_guess,
            method='SLSQP', constraints=constraints,
            options={'maxiter': 50, 'disp': False}
        )

        final_soc = self.bess_model.update_soc(result.x)
        final_revenue = -obj_func(result.x)

        return {
            'success': result.success,
            'message': result.message,
            'optimal_power': result.x,
            'soc_profile': final_soc,
            'revenue': final_revenue,
            'objective_value': result.fun,
            'num_iterations': result.nit,
            'losses_kw': np.zeros(self.time_steps)  # 暂不计算损耗
        }

    def _prepare_final_result(self, optimal_power, revenue):
        """
        准备最终优化结果
        """
        # 计算SOC轮廓
        final_soc = self.bess_model.update_soc(optimal_power)

        # 对每个时段执行潮流，记录潮流结果
        ac_power_flow_results = []
        losses = np.zeros(self.time_steps)

        if self.network_model:
            for t in range(self.time_steps):
                bess_power = optimal_power[t]
                pv_power = self.pv_prediction['prediction'][t] if self.pv_prediction else 0
                wind_power = self.wind_prediction['prediction'][t] if self.wind_prediction else 0
                load = self.load_data[t] if self.load_data is not None else 0

                success, ac_results, violations, grid_power = self.network_model.integrate_with_bess(
                    load, pv_power, wind_power, bess_power
                )

                # 记录损耗
                if success and ac_results and 'total_results' in ac_results:
                    losses[t] = ac_results['total_results']['total_p_loss_mw'] * 1000

                ac_power_flow_results.append({
                    'time_step': t,
                    'success': success,
                    'violations': violations,
                    'grid_power_kw': grid_power if success else None,
                    'voltage_profiles': ac_results['bus_results']['vm_pu'] if success and ac_results else None,
                    'line_loadings': ac_results['line_results']['loading_percent'] if success and ac_results else None
                })

        # 准备最终结果
        optimization_result = {
            'success': True,
            'message': '多阶段优化成功',
            'optimal_power': optimal_power,
            'soc_profile': final_soc,
            'revenue': revenue,
            'objective_value': -revenue,
            'num_iterations': 1,  # 这里不反映实际迭代次数
            'losses_kw': losses,
            'ac_power_flow_results': ac_power_flow_results
        }

        return optimization_result

    def evaluate_constraint_violation(self, power_at_t, t):
        """
        保留原有的方法以维持向后兼容性
        """
        if t % 5 == 0:
            print(f"正在优化时间步 {t}/{self.bess_model.time_steps - 1}...")

        load = self.load_data[t] if self.load_data is not None else 0
        pv_power = self.pv_prediction['prediction'][t] if self.pv_prediction else 0
        wind_power = self.wind_prediction['prediction'][t] if self.wind_prediction else 0

        success, ac_results, violations, _ = self.network_model.integrate_with_bess(
            load, pv_power, wind_power, power_at_t
        )
        if not success:
            return 1000.0, None

        violation_degree = 0.0
        if violations:
            for v_id, v_info in violations.items():
                if v_info['type'] == 'line_overload':
                    violation_degree += (v_info['loading_percent'] - 100) / 100 * 10
                elif v_info['type'] == 'voltage_violation':
                    vm_pu = v_info['vm_pu']
                    if vm_pu < 0.95:
                        violation_degree += (0.95 - vm_pu) / 0.95 * 100
                    else:
                        violation_degree += (vm_pu - 1.05) / 1.05 * 100

        if ac_results and 'total_results' in ac_results:
            self.losses[t] = ac_results['total_results']['total_p_loss_mw'] * 1000

        return violation_degree, ac_results

    def optimize(self):
        """
        维持原优化方法向后兼容，但推荐使用新的solver模块
        """
        print("注意: 正在使用旧的优化方法，建议使用 solve_with_two_stage_approach() 获得更好的结果")

        def obj_func(power_vector):
            # 重置损耗
            self.losses = np.zeros(self.time_steps)
            total_violation = 0

            # 计算潮流约束违反和线路损耗
            if self.network_model:
                for t in range(self.time_steps):
                    violation, _ = self.evaluate_constraint_violation(power_vector[t], t)
                    total_violation += violation

            # 计算收益(加上违反惩罚)
            revenue = self.bess_model.revenue_function(
                power_vector, self.electricity_price,
                pv_power=self.pv_prediction['prediction'] if self.pv_prediction else None,
                wind_power=self.wind_prediction['prediction'] if self.wind_prediction else None,
                load=self.load_data,
                losses=self.losses
            )
            revenue -= self.violation_penalty * total_violation
            return -revenue  # 最大化收益 => 最小化负收益

        # 构造约束
        constraints = []
        for t in range(self.time_steps):
            constraints.append({'type': 'ineq',
                                'fun': lambda x, idx=t: self.bess_model.max_power_kw - x[idx]})
            constraints.append({'type': 'ineq',
                                'fun': lambda x, idx=t: x[idx] + self.bess_model.max_power_kw})
        # SOC约束
        for t in range(1, self.time_steps + 1):
            def soc_upper_constraint(x, step=t):
                soc = self.bess_model.update_soc(x)
                return self.bess_model.max_soc - soc[step]

            def soc_lower_constraint(x, step=t):
                soc = self.bess_model.update_soc(x)
                return soc[step] - self.bess_model.min_soc

            constraints.append({'type': 'ineq', 'fun': soc_upper_constraint})
            constraints.append({'type': 'ineq', 'fun': soc_lower_constraint})

        result = minimize(
            obj_func, self.initial_guess,
            method='SLSQP', constraints=constraints,
            options={'maxiter': 20, 'disp': False, 'ftol': 1e-2}
        )

        final_soc = self.bess_model.update_soc(result.x)
        final_revenue = -obj_func(result.x)

        optimization_result = {
            'success': result.success,
            'message': result.message,
            'optimal_power': result.x,
            'soc_profile': final_soc,
            'revenue': final_revenue,
            'objective_value': result.fun,
            'num_iterations': result.nit,
            'losses_kw': self.losses.copy()
        }

        # 对每个时段执行潮流，记录潮流结果
        if self.network_model:
            ac_power_flow_results = []
            for t in range(self.time_steps):
                bess_power = result.x[t]
                pv_power = self.pv_prediction['prediction'][t] if self.pv_prediction else 0
                wind_power = self.wind_prediction['prediction'][t] if self.wind_prediction else 0
                load = self.load_data[t] if self.load_data is not None else 0

                success, ac_results, violations, grid_power = self.network_model.integrate_with_bess(
                    load, pv_power, wind_power, bess_power
                )
                ac_power_flow_results.append({
                    'time_step': t,
                    'success': success,
                    'violations': violations,
                    'grid_power_kw': grid_power if success else None,
                    'voltage_profiles': ac_results['bus_results']['vm_pu'] if success and ac_results else None,
                    'line_loadings': ac_results['line_results']['loading_percent'] if success and ac_results else None
                })
            optimization_result['ac_power_flow_results'] = ac_power_flow_results

        print(f"优化完成 - 成功: {result.success}, 收益: {final_revenue:.2f}")
        return optimization_result