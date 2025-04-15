# forecasting.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class PVPowerPrediction:
    """
    光伏发电预测模型 - 基于天气因素的不确定性建模
    使用随机森林回归模型进行预测
    """
    def __init__(self, weather_data):
        """
        初始化光伏发电预测模型
        参数:
            weather_data (DataFrame): 包含天气数据的DataFrame
        """
        self.weather_data = weather_data
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.features = ['solar_radiation', 'temperature', 'precipitation']

        # 确保所需的特征存在于数据中
        for feature in self.features:
            if feature not in weather_data.columns:
                raise ValueError(f"天气数据缺少必要的特征: {feature}")

    def simulate_pv_power(self, capacity_kw=100):
        """
        简易模拟光伏发电功率
        """
        max_radiation = 1000
        pv_power = self.weather_data['solar_radiation'] / max_radiation * capacity_kw

        # 温度影响(每升高1度效率降低0.4%, 基准温度25度)
        temp_effect = 1 - 0.004 * (self.weather_data['temperature'] - 25)
        pv_power = pv_power * temp_effect

        # 降水影响(简化：有降水时效率降低20%)
        rain_effect = 1 - 0.2 * (self.weather_data['precipitation'] > 0)
        pv_power = pv_power * rain_effect

        pv_power = pv_power.clip(lower=0)
        np.random.seed(42)
        uncertainty = np.random.normal(1, 0.05, len(pv_power))
        pv_power = pv_power * uncertainty
        return pv_power.clip(lower=0)

    def train_prediction_model(self, simulated_power):
        """
        训练随机森林预测模型
        """
        X = self.weather_data[self.features]
        y = simulated_power
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        print(f"PV预测模型R²分数 - 训练集: {train_score:.4f}, 测试集: {test_score:.4f}")
        return self.model

    def predict_with_uncertainty(self, weather_input=None):
        """
        使用模型进行预测，并给出置信区间
        返回包含prediction, lower_bound, upper_bound, uncertainty的字典
        """
        if weather_input is None:
            weather_input = self.weather_data[self.features]

        y_pred_list = []
        for tree in self.model.estimators_:
            y_pred_list.append(tree.predict(weather_input))

        predictions_array = np.array(y_pred_list)
        mean_prediction = predictions_array.mean(axis=0)
        std_prediction = predictions_array.std(axis=0)

        # 95%置信区间
        lower_bound = mean_prediction - 1.96 * std_prediction
        upper_bound = mean_prediction + 1.96 * std_prediction
        lower_bound = np.maximum(lower_bound, 0)

        return {
            'prediction': mean_prediction,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'uncertainty': std_prediction
        }


class WindPowerPrediction:
    """
    风电发电预测模型 - 基于风速/风电数据
    """
    def __init__(self, weather_data=None, wind_data=None):
        self.weather_data = weather_data
        self.wind_data = wind_data
        self.has_wind_speed = (weather_data is not None and 'wind_speed' in weather_data.columns)
        self.has_wind_power = (wind_data is not None and 'wind_power_kw' in wind_data.columns)

        self.rated_power = 100
        self.cut_in_speed = 3.0
        self.rated_speed = 12.0
        self.cut_out_speed = 25.0

    def wind_to_power(self, wind_speed):
        wind_power = np.zeros_like(wind_speed)
        for i, speed in enumerate(wind_speed):
            if speed < self.cut_in_speed or speed > self.cut_out_speed:
                wind_power[i] = 0
            elif speed < self.rated_speed:
                wind_power[i] = self.rated_power * (speed - self.cut_in_speed)/(self.rated_speed - self.cut_in_speed)
            else:
                wind_power[i] = self.rated_power
        return wind_power

    def predict_wind_power(self):
        """
        返回字典: { 'prediction': ..., 'lower_bound':..., 'upper_bound':..., 'uncertainty':... }
        """
        if self.has_wind_power:
            # 直接使用已有风电数据
            wind_power = self.wind_data['wind_power_kw'].values
            np.random.seed(44)
            uncertainty = 0.1
            lower_bound = wind_power * (1 - uncertainty)
            upper_bound = wind_power * (1 + uncertainty)
        elif self.has_wind_speed:
            # 根据风速推算风电功率
            wind_speed = self.weather_data['wind_speed'].values
            wind_power = self.wind_to_power(wind_speed)
            np.random.seed(44)
            uncertainty = 0.15
            lower_bound = wind_power * (1 - uncertainty)
            upper_bound = wind_power * (1 + uncertainty)
        else:
            # 随机模拟
            n_steps = len(self.weather_data) if self.weather_data is not None else 90
            t = np.arange(n_steps)
            base_pattern = 30 + 20 * np.sin(2 * np.pi * t / (24 * n_steps/90))
            random_variations = 15 * np.random.randn(n_steps)
            wind_power = base_pattern + random_variations
            wind_power = np.clip(wind_power, 0, self.rated_power)
            uncertainty = 0.2
            lower_bound = np.clip(wind_power*(1 - uncertainty), 0, None)
            upper_bound = np.clip(wind_power*(1 + uncertainty), None, self.rated_power)

        return {
            'prediction': wind_power,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'uncertainty': np.abs(upper_bound - lower_bound)/2
        }
