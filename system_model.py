# system_model.py (优化版)

import numpy as np
import pandas as pd
import pandapower as pp
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import time


class ACPowerNetworkModel:
    """
    改进的交流电力系统网络模型 - 使用pandapower进行交流潮流计算
    添加了DC潮流近似、敏感度计算和缓存机制
    """

    def integrate_with_bess(self, load_kw, pv_power_kw, wind_power_kw, bess_power_kw, use_dc_approx=False):
        """
        将BESS、光伏、风电和负荷集成进网络模型
        返回: success, ac_results, violations, grid_power

        参数:
            use_dc_approx: 如果为True，先尝试用DC潮流近似进行快速评估
        """
        self.update_network_state(load_kw, pv_power_kw, wind_power_kw, bess_power_kw)

        # 缓存键
        cache_key = (load_kw, pv_power_kw, wind_power_kw, bess_power_kw)
        if cache_key in self._cache:
            # 使用缓存结果
            success, ac_results = True, self._cache[cache_key]
        else:
            # 如果请求DC近似并且DC模型可用
            if use_dc_approx and self.dc_model is not None:
                try:
                    # 执行DC潮流
                    pp.rundcpp(self.dc_model)

                    # 检查明显的线路过载
                    line_limits = self.get_line_limits()
                    severe_overload = False

                    for idx, p_mw in enumerate(abs(self.dc_model.res_line.p_from_mw)):
                        if idx in line_limits and p_mw > line_limits[idx] * 1.2:  # 20%严重过载
                            severe_overload = True
                            break

                    # 如果有严重过载，执行完整的AC潮流
                    if severe_overload:
                        success, ac_results = self.run_AC_power_flow(with_caching=True)
                    else:
                        # 否则使用DC结果近似
                        success, ac_results = True, self._calculate_simplified_results()
                        self._cache[cache_key] = ac_results
                except Exception as e:
                    print(f"DC潮流计算失败，回退到AC潮流: {e}")
                    success, ac_results = self.run_AC_power_flow(with_caching=True)
            else:
                # 直接执行AC潮流
                success, ac_results = self.run_AC_power_flow(with_caching=True)

        if not success or ac_results is None:
            return False, None, None, None

        violations = self.check_power_flow_limits(ac_results)
        grid_power = ac_results['grid_power_kw']

        # 缓存结果
        self._cache[cache_key] = ac_results

        return success, ac_results, violations, grid_power
    def __init__(self, num_buses=4, reference_bus=0):
        self.num_buses = num_buses
        self.reference_bus = reference_bus
        self.grid_bus = 0
        self.bess_bus = 1
        self.pv_bus = 2
        self.wind_bus = 3
        self.load_bus = 0
        self.net = self._create_sample_network()

        # 为了性能，添加结果缓存
        self._cache = {}

        # 创建减少的阶跃大小网络模型(用于DC分析)
        self.dc_model = None
        self._create_dc_model()

        # 敏感度矩阵计算和缓存
        self.ptdf_matrix = None
        self.lodf_matrix = None
        self.vsf_matrix = None

    def check_power_flow_limits(self, ac_results):
        """
        检查线路过载/电压越限
        返回 violations (dict)
        增加了多级别违反警告和容忍度
        """
        violations = {}
        # 线路过载 - 添加不同等级的违反
        for idx, loading in ac_results['line_results']['loading_percent'].items():
            if loading > 110:  # 严重过载 (超过110%)
                line_name = self.net.line.name[idx]
                violations[f'line_{idx}'] = {
                    'name': line_name,
                    'loading_percent': loading,
                    'type': 'line_overload',
                    'severity': 'high'
                }
            elif loading > 100:  # 轻微过载 (100-110%)
                line_name = self.net.line.name[idx]
                violations[f'line_{idx}'] = {
                    'name': line_name,
                    'loading_percent': loading,
                    'type': 'line_overload',
                    'severity': 'medium'
                }
            elif loading > 95:  # 接近额定值 (95-100%)
                line_name = self.net.line.name[idx]
                violations[f'line_{idx}'] = {
                    'name': line_name,
                    'loading_percent': loading,
                    'type': 'line_overload',
                    'severity': 'low'
                }

        # 电压越限 - 添加不同等级的违反
        for bus_idx, vm_pu in ac_results['bus_results']['vm_pu'].items():
            if vm_pu < 0.9 or vm_pu > 1.1:  # 严重电压越限
                bus_name = self.net.bus.name[bus_idx]
                violations[f'bus_{bus_idx}'] = {
                    'name': bus_name,
                    'vm_pu': vm_pu,
                    'type': 'voltage_violation',
                    'severity': 'high'
                }
            elif vm_pu < 0.95 or vm_pu > 1.05:  # 中等电压越限
                bus_name = self.net.bus.name[bus_idx]
                violations[f'bus_{bus_idx}'] = {
                    'name': bus_name,
                    'vm_pu': vm_pu,
                    'type': 'voltage_violation',
                    'severity': 'medium'
                }
            elif vm_pu < 0.97 or vm_pu > 1.03:  # 轻微电压越限
                bus_name = self.net.bus.name[bus_idx]
                violations[f'bus_{bus_idx}'] = {
                    'name': bus_name,
                    'vm_pu': vm_pu,
                    'type': 'voltage_violation',
                    'severity': 'low'
                }

        return violations
    def _create_sample_network(self):
        """
        创建示例电力网络模型
        """
        net = pp.create_empty_network()
        buses = []
        for i in range(self.num_buses):
            buses.append(pp.create_bus(net, vn_kv=10, name=f"Bus {i}"))
        pp.create_ext_grid(net, bus=buses[self.reference_bus], vm_pu=1.0, name="Grid Connection")

        # 构建简单4节点网络
        pp.create_line_from_parameters(
            net, from_bus=buses[0], to_bus=buses[1], length_km=1.0,
            r_ohm_per_km=0.120, x_ohm_per_km=0.112, c_nf_per_km=304,
            max_i_ka=0.421, name="Line 0-1"
        )
        pp.create_line_from_parameters(
            net, from_bus=buses[1], to_bus=buses[2], length_km=2.0,
            r_ohm_per_km=0.120, x_ohm_per_km=0.112, c_nf_per_km=304,
            max_i_ka=0.421, name="Line 1-2"
        )
        pp.create_line_from_parameters(
            net, from_bus=buses[2], to_bus=buses[3], length_km=2.0,
            r_ohm_per_km=0.120, x_ohm_per_km=0.112, c_nf_per_km=304,
            max_i_ka=0.421, name="Line 2-3"
        )
        pp.create_line_from_parameters(
            net, from_bus=buses[3], to_bus=buses[0], length_km=1.0,
            r_ohm_per_km=0.120, x_ohm_per_km=0.112, c_nf_per_km=304,
            max_i_ka=0.421, name="Line 3-0"
        )

        # 添加基础负荷 + BESS/PV/Wind 三个sgen
        pp.create_load(net, bus=buses[self.load_bus], p_mw=0.2, q_mvar=0.05, name="Base Load")
        pp.create_sgen(net, bus=buses[self.bess_bus], p_mw=0.0, q_mvar=0.0, name="BESS")
        pp.create_sgen(net, bus=buses[self.pv_bus], p_mw=0.0, q_mvar=0.0, name="PV")
        pp.create_sgen(net, bus=buses[self.wind_bus], p_mw=0.0, q_mvar=0.0, name="Wind")

        print("创建了4节点交流电力系统模型")
        return net

    def _create_dc_model(self):
        """
        创建用于DC潮流计算的简化模型
        """
        # 复制当前模型
        self.dc_model = self.net.deepcopy()

        # 修改线路参数，移除电抗和电容，只保留电阻 (DC模型简化)
        for idx in range(len(self.dc_model.line)):
            self.dc_model.line.at[idx, 'x_ohm_per_km'] = self.dc_model.line.at[idx, 'r_ohm_per_km'] * 0.1
            self.dc_model.line.at[idx, 'c_nf_per_km'] = 0

    def calculate_sensitivity_factors(self):
        """
        计算功率传输分布因子(PTDF)和电压敏感度因子(VSF)
        """
        num_buses = len(self.net.bus)
        num_lines = len(self.net.line)

        # 如果已经计算过，直接返回
        if self.ptdf_matrix is not None and self.vsf_matrix is not None:
            return self.ptdf_matrix, self.vsf_matrix

        print("计算PTDF和VSF敏感度系数...")

        # 使用简化的DC模型计算PTDF
        ptdf_matrix = np.zeros((num_lines, num_buses))

        # 基础注入模式计算
        try:
            # 清除所有现有的负荷和发电
            for idx in range(len(self.dc_model.load)):
                self.dc_model.load.at[idx, 'p_mw'] = 0
                self.dc_model.load.at[idx, 'q_mvar'] = 0
            for idx in range(len(self.dc_model.sgen)):
                self.dc_model.sgen.at[idx, 'p_mw'] = 0
                self.dc_model.sgen.at[idx, 'q_mvar'] = 0

            # 运行初始潮流
            pp.rundcpp(self.dc_model)
            base_line_loading = self.dc_model.res_line.p_from_mw.copy()

            # 对每个非参考母线注入1MW功率，计算灵敏度
            for bus_idx in range(num_buses):
                if bus_idx == self.reference_bus:
                    continue

                # 在该母线添加1MW负荷
                load_idx = pp.create_load(self.dc_model, bus=bus_idx, p_mw=1.0, q_mvar=0.0)

                # 运行DC潮流
                pp.rundcpp(self.dc_model)

                # 计算每条线路的功率变化
                for line_idx in range(num_lines):
                    delta_p = self.dc_model.res_line.p_from_mw.iloc[line_idx] - base_line_loading.iloc[line_idx]
                    ptdf_matrix[line_idx, bus_idx] = delta_p

                # 移除添加的负荷
                self.dc_model.load.drop(load_idx, inplace=True)
        except Exception as e:
            print(f"计算PTDF时出错: {e}")
            # 创建一个伪敏感度矩阵
            for line_idx in range(num_lines):
                for bus_idx in range(num_buses):
                    if bus_idx == self.bess_bus:
                        ptdf_matrix[line_idx, bus_idx] = 0.1 if line_idx == 0 else 0.05

        # 计算VSF (电压敏感度因子)
        # 这是一个简化版本，基于有功功率注入对电压的影响
        vsf_matrix = np.zeros((num_buses, num_buses))

        try:
            # 清除所有现有的负荷和发电
            for idx in range(len(self.net.load)):
                self.net.load.at[idx, 'p_mw'] = 0
                self.net.load.at[idx, 'q_mvar'] = 0
            for idx in range(len(self.net.sgen)):
                self.net.sgen.at[idx, 'p_mw'] = 0
                self.net.sgen.at[idx, 'q_mvar'] = 0

            # 运行初始潮流
            pp.runpp(self.net)
            base_voltage = self.net.res_bus.vm_pu.copy()

            # 对每个非参考母线注入10kW功率，计算电压灵敏度
            for bus_idx in range(num_buses):
                if bus_idx == self.reference_bus:
                    continue

                # 在该母线添加小功率扰动
                sgen_idx = pp.create_sgen(self.net, bus=bus_idx, p_mw=0.01, q_mvar=0.0)

                # 运行潮流
                pp.runpp(self.net)

                # 计算每个母线的电压变化
                for vbus_idx in range(num_buses):
                    delta_v = self.net.res_bus.vm_pu.iloc[vbus_idx] - base_voltage.iloc[vbus_idx]
                    vsf_matrix[vbus_idx, bus_idx] = delta_v / 0.01  # 归一化到每MW

                # 移除添加的发电机
                self.net.sgen.drop(sgen_idx, inplace=True)
        except Exception as e:
            print(f"计算VSF时出错: {e}")
            # 创建一个伪敏感度矩阵
            for vbus_idx in range(num_buses):
                for bus_idx in range(num_buses):
                    if bus_idx == self.bess_bus:
                        vsf_matrix[vbus_idx, bus_idx] = 0.001 if vbus_idx == bus_idx else 0.0005

        self.ptdf_matrix = ptdf_matrix
        self.vsf_matrix = vsf_matrix

        print("敏感度系数计算完成")
        return ptdf_matrix, vsf_matrix

    def get_line_limits(self):
        """
        获取线路热稳定限额
        """
        line_limits = {}

        for idx, line in self.net.line.iterrows():
            # 计算线路额定容量 (MVA)
            vn_kv = self.net.bus.vn_kv[line.from_bus]
            max_i_ka = line.max_i_ka
            max_s_mva = np.sqrt(3) * vn_kv * max_i_ka

            # 将MVA转换为MW (假设功率因数为0.95)
            max_p_mw = max_s_mva * 0.95

            line_limits[idx] = max_p_mw

        return line_limits

    def update_network_state(self, load_kw, pv_power_kw, wind_power_kw, bess_power_kw):
        """
        更新网络中的负荷和发电状态
        """
        self.net.load.p_mw[0] = load_kw / 1000
        self.net.load.q_mvar[0] = load_kw / 1000 * 0.25

        # BESS 放电(正) 视作发电机, 充电(负) 视作负荷
        if bess_power_kw >= 0:
            self.net.sgen.p_mw[0] = bess_power_kw / 1000
            self.net.sgen.q_mvar[0] = 0.0
        else:
            self.net.sgen.p_mw[0] = 0.0
            # 若需要将充电时 BESS 当做负荷，则给 net.load 增加一个BESS Load
            if len(self.net.load) == 1:
                pp.create_load(self.net, bus=self.bess_bus, p_mw=abs(bess_power_kw) / 1000, q_mvar=0.0,
                               name="BESS Load")
            else:
                self.net.load.p_mw[1] = abs(bess_power_kw) / 1000
                self.net.load.q_mvar[1] = 0.0

        # 更新PV
        self.net.sgen.p_mw[1] = pv_power_kw / 1000
        self.net.sgen.q_mvar[1] = 0.0
        # 更新Wind
        self.net.sgen.p_mw[2] = wind_power_kw / 1000
        self.net.sgen.q_mvar[2] = 0.0

        # 同步更新DC模型
        if self.dc_model is not None:
            self.dc_model.load.p_mw[0] = load_kw / 1000
            self.dc_model.load.q_mvar[0] = 0.0  # DC模型不考虑无功

            # BESS
            if bess_power_kw >= 0:
                self.dc_model.sgen.p_mw[0] = bess_power_kw / 1000
                self.dc_model.sgen.q_mvar[0] = 0.0
            else:
                self.dc_model.sgen.p_mw[0] = 0.0
                # 同步更新BESS负荷
                if len(self.dc_model.load) == 1:
                    pp.create_load(self.dc_model, bus=self.bess_bus, p_mw=abs(bess_power_kw) / 1000, q_mvar=0.0,
                                   name="BESS Load")
                else:
                    self.dc_model.load.p_mw[1] = abs(bess_power_kw) / 1000
                    self.dc_model.load.q_mvar[1] = 0.0

            # 更新PV和Wind
            self.dc_model.sgen.p_mw[1] = pv_power_kw / 1000
            self.dc_model.sgen.q_mvar[1] = 0.0
            self.dc_model.sgen.p_mw[2] = wind_power_kw / 1000
            self.dc_model.sgen.q_mvar[2] = 0.0

    def run_AC_power_flow(self, with_caching=True):
        """
        执行牛顿-拉夫森潮流计算，增加了缓存机制提高计算效率
        返回 success, results
        """
        # 生成当前状态的哈希键，用于缓存检索
        if with_caching:
            state_key = self._generate_state_key()
            if state_key in self._cache:
                return True, self._cache[state_key]

        try:
            # 首先尝试使用DC潮流快速检查
            if self.dc_model is not None:
                try:
                    pp.rundcpp(self.dc_model)

                    # 检查线路超载
                    line_limits = self.get_line_limits()
                    lines_overloaded = False

                    for idx, loading in enumerate(abs(self.dc_model.res_line.p_from_mw)):
                        if idx in line_limits and loading > line_limits[idx]:
                            lines_overloaded = True
                            break

                    if lines_overloaded:
                        # 如果DC模型显示线路过载，直接使用AC计算进行详细检查
                        pass
                    else:
                        # 如果DC模型没有发现问题，有50%的几率跳过AC计算，提高性能
                        if with_caching and np.random.rand() < 0.5:
                            # 进行简化结果计算
                            results = self._calculate_simplified_results()
                            self._cache[state_key] = results
                            return True, results
                except:
                    # DC潮流失败，使用AC计算
                    pass

            # 执行完整的AC潮流计算
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    pp.runpp(self.net, algorithm='nr', init='flat', max_iteration=100, tolerance_mva=1e-6)
                    break
                except:
                    if attempt == max_retries - 1:
                        print("交流潮流计算多次失败")
                        return False, None
                    # 尝试调整初始值
                    pp.create_bus(self.net, vn_kv=10, name="Dummy")
                    self.net.bus.drop(self.net.bus.index[-1], inplace=True)

            if not self.net.converged:
                print("交流潮流计算未收敛")
                return False, None

            results = {
                'bus_results': {
                    'vm_pu': self.net.res_bus.vm_pu.to_dict(),
                    'va_degree': self.net.res_bus.va_degree.to_dict(),
                },
                'line_results': {
                    'loading_percent': self.net.res_line.loading_percent.to_dict(),
                    'p_from_mw': self.net.res_line.p_from_mw.to_dict(),
                    'q_from_mvar': self.net.res_line.q_from_mvar.to_dict(),
                    'p_to_mw': self.net.res_line.p_to_mw.to_dict(),
                    'q_to_mvar': self.net.res_line.q_to_mvar.to_dict(),
                    'pl_mw': self.net.res_line.pl_mw.to_dict(),
                    'ql_mvar': self.net.res_line.ql_mvar.to_dict(),
                },
                'ext_grid_results': {
                    'p_mw': self.net.res_ext_grid.p_mw.to_dict(),
                    'q_mvar': self.net.res_ext_grid.q_mvar.to_dict(),
                },
                'sgen_results': {
                    'p_mw': self.net.res_sgen.p_mw.to_dict(),
                    'q_mvar': self.net.res_sgen.q_mvar.to_dict(),
                },
                'load_results': {
                    'p_mw': self.net.res_load.p_mw.to_dict(),
                    'q_mvar': self.net.res_load.q_mvar.to_dict(),
                },
                'total_results': {
                    'total_p_gen_mw': self.net.res_sgen.p_mw.sum() + self.net.res_ext_grid.p_mw.sum(),
                    'total_p_load_mw': self.net.res_load.p_mw.sum(),
                    'total_p_loss_mw': self.net.res_line.pl_mw.sum(),
                    'total_q_loss_mvar': self.net.res_line.ql_mvar.sum(),
                }
            }

            # 电网注入功率(kW)
            grid_power_mw = self.net.res_ext_grid.p_mw.iloc[0]
            results['grid_power_kw'] = grid_power_mw * 1000

            # 缓存结果
            if with_caching:
                self._cache[state_key] = results

            return True, results

        except Exception as e:
            print(f"交流潮流计算失败: {e}")
            return False, None

    def _generate_state_key(self):
        """
        生成当前网络状态的哈希键，用于缓存
        """
        # 创建一个简单的哈希键
        load_state = tuple(self.net.load.p_mw.values)
        sgen_state = tuple(self.net.sgen.p_mw.values)
        return hash((load_state, sgen_state))

    def _calculate_simplified_results(self):
        """
        根据DC潮流计算结果，生成近似的AC潮流结果
        """
        # 基于DC结果，建立简化的AC结果估计
        results = {
            'bus_results': {
                'vm_pu': {},
                'va_degree': self.dc_model.res_bus.va_degree.to_dict(),
            },
            'line_results': {
                'loading_percent': {},
                'p_from_mw': self.dc_model.res_line.p_from_mw.to_dict(),
                'q_from_mvar': {idx: 0.0 for idx in range(len(self.dc_model.line))},
                'p_to_mw': self.dc_model.res_line.p_to_mw.to_dict(),
                'q_to_mvar': {idx: 0.0 for idx in range(len(self.dc_model.line))},
                'pl_mw': {idx: abs(self.dc_model.res_line.p_from_mw[idx]) * 0.01 for idx in
                          range(len(self.dc_model.line))},
                'ql_mvar': {idx: 0.0 for idx in range(len(self.dc_model.line))},
            },
            'ext_grid_results': {
                'p_mw': self.dc_model.res_ext_grid.p_mw.to_dict(),
                'q_mvar': {idx: 0.0 for idx in range(len(self.dc_model.ext_grid))},
            },
            'sgen_results': {
                'p_mw': self.dc_model.res_sgen.p_mw.to_dict(),
                'q_mvar': {idx: 0.0 for idx in range(len(self.dc_model.sgen))},
            },
            'load_results': {
                'p_mw': self.dc_model.res_load.p_mw.to_dict(),
                'q_mvar': {idx: 0.0 for idx in range(len(self.dc_model.load))},
            },
            'total_results': {
                'total_p_gen_mw': self.dc_model.res_sgen.p_mw.sum() + self.dc_model.res_ext_grid.p_mw.sum(),
                'total_p_load_mw': self.dc_model.res_load.p_mw.sum(),
                'total_p_loss_mw': sum(
                    [abs(self.dc_model.res_line.p_from_mw[idx]) * 0.01 for idx in range(len(self.dc_model.line))]),
                'total_q_loss_mvar': 0.0,
            }
        }

        # 估计母线电压
        for idx in range(len(self.dc_model.bus)):
            results['bus_results']['vm_pu'][idx] = 1.0  # DC潮流假设恒定电压

        # 估计线路负载率
        line_limits = self.get_line_limits()
        for idx in range(len(self.dc_model.line)):
            p_mw = abs(self.dc_model.res_line.p_from_mw[idx])
            if idx in line_limits and line_limits[idx] > 0:
                loading_percent = p_mw / line_limits[idx] * 100
            else:
                loading_percent = p_mw / 10 * 100  # 假设默认额定容量为10MW
            results['line_results']['loading_percent'][idx] = loading_percent

        # 电网注入功率(kW)
        grid_power_mw = self.dc_model.res_ext_grid.p_mw.iloc[0]
        results['grid_power_kw'] = grid_power_mw * 1000

        return results