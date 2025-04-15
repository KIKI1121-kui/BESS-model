# wind_data_generator.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def generate_wind_speed_data(weather_data_file, output_file='风速数据.csv'):
    """
    基于现有天气数据生成风速数据
    参数:
        weather_data_file: 处理后的天气数据文件路径
        output_file: 输出的风速数据文件路径
    返回:
        wind_data: 包含风速数据的DataFrame
    """
    print(f"正在生成风速数据，基于天气数据文件: {weather_data_file}")

    # 读取现有天气数据以获取时间戳
    weather_data = pd.read_csv(weather_data_file)

    # 创建新的DataFrame来存储风速数据
    wind_data = pd.DataFrame()

    # 复制时间戳列
    wind_data['timestamp'] = weather_data['timestamp']

    # 转换时间戳为datetime格式以便于时间序列分析
    try:
        wind_data['timestamp'] = pd.to_datetime(wind_data['timestamp'])
    except:
        # 如果无法解析时间戳，创建新的时间序列
        start_date = datetime(2024, 12, 1)
        wind_data['timestamp'] = [start_date + timedelta(days=i) for i in range(len(weather_data))]

    # 获取时间特征
    wind_data['day_of_year'] = wind_data['timestamp'].dt.dayofyear
    wind_data['hour'] = wind_data['timestamp'].dt.hour

    # 基于季节性趋势生成基础风速模式
    # 使用正弦函数模拟季节性和日内变化
    seasonal_component = 5 + 3 * np.sin(2 * np.pi * wind_data['day_of_year'] / 365)
    daily_component = 1.5 * np.sin(2 * np.pi * wind_data['hour'] / 24 - np.pi / 4)  # 风速通常在下午达到峰值

    # 结合天气参数来调整风速
    # 温度影响：温度越高，风速差异越大
    temp_effect = 0.5 * (
                (weather_data['temperature'] - weather_data['temperature'].mean()) / weather_data['temperature'].std())

    # 降水影响：有降水时风速通常更大
    precip_effect = 0.8 * (weather_data['precipitation'] > 0).astype(int)

    # 太阳辐射影响：辐射强度大时通常伴随高压系统，风速较小
    radiation_effect = -0.5 * (
                (weather_data['solar_radiation'] - weather_data['solar_radiation'].mean()) / weather_data[
            'solar_radiation'].std())

    # 随机成分
    np.random.seed(42)  # 固定随机种子以确保可重复性
    random_component = np.random.normal(0, 1.5, len(wind_data))

    # 组合所有成分生成风速数据
    wind_data[
        'wind_speed'] = seasonal_component + daily_component + temp_effect + precip_effect + radiation_effect + random_component

    # 确保风速为正且在合理范围内 (通常0~25 m/s)
    wind_data['wind_speed'] = np.clip(wind_data['wind_speed'], 0, 25)

    # 添加额外的突发风速事件
    # 随机选择几天作为风暴日
    storm_days = np.random.choice(wind_data['day_of_year'].unique(), size=3, replace=False)
    for day in storm_days:
        day_mask = wind_data['day_of_year'] == day
        # 在风暴日增加风速
        storm_hours = np.random.choice(24, size=8, replace=False)  # 随机8小时风暴
        for hour in storm_hours:
            hour_mask = wind_data['hour'] == hour
            mask = day_mask & hour_mask
            if any(mask):
                wind_data.loc[mask, 'wind_speed'] += np.random.uniform(5, 12)

    # 确保风速在合理范围内
    wind_data['wind_speed'] = np.clip(wind_data['wind_speed'], 0, 25)

    # 移除辅助列
    wind_data = wind_data[['timestamp', 'wind_speed']]

    # 保存到CSV文件
    wind_data.to_csv(output_file, index=False)
    print(f"风速数据已生成并保存至 '{output_file}'")
    print(
        f"数据统计: 平均风速 = {wind_data['wind_speed'].mean():.2f} m/s, 最大风速 = {wind_data['wind_speed'].max():.2f} m/s")

    # 绘制风速数据分布
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(wind_data['timestamp'], wind_data['wind_speed'])
    plt.title('风速时间序列')
    plt.xlabel('时间')
    plt.ylabel('风速 (m/s)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(wind_data['wind_speed'], bins=20)
    plt.title('风速分布直方图')
    plt.xlabel('风速 (m/s)')
    plt.ylabel('频率')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('风速数据分析.png', dpi=300)

    return wind_data


if __name__ == "__main__":
    generate_wind_speed_data("处理后的天气数据.csv")