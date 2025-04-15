# data_loader.py

import os
import numpy as np
import pandas as pd

def load_and_preprocess_data(weather_file, price_file, load_file=None, wind_file=None):
    """
    加载天气数据、电价数据、负荷数据和风电数据，进行必要的预处理
    返回: weather_data, price_data, load_data, wind_data
    """
    print(f"加载数据文件: {weather_file}, {price_file}")
    if load_file:
        print(f"加载负荷数据: {load_file}")
    if wind_file:
        print(f"加载风电数据: {wind_file}")

    # 1) 加载天气数据
    weather_data = pd.read_csv(weather_file) if os.path.exists(weather_file) else None

    # 2) 加载电价数据
    price_data = pd.read_csv(price_file) if os.path.exists(price_file) else None

    # 3) 加载或模拟负荷数据
    load_data = None
    if load_file and os.path.exists(load_file):
        load_data = pd.read_csv(load_file)
    else:
        # 如果没有真实负荷数据则进行模拟
        if price_data is not None and 'timestamp' in price_data.columns:
            timestamps = price_data['timestamp']
            load_data = pd.DataFrame({'timestamp': pd.to_datetime(timestamps)})
            if 'hour' not in load_data.columns:
                load_data['hour'] = load_data['timestamp'].dt.hour
                base_load = 200  # 基础负荷 (kW)
                peak_factor = 1.5  # 高峰负荷因子

                # 简单日负荷模式
                load_pattern = np.ones(24) * base_load
                load_pattern[7:10] = base_load * peak_factor
                load_pattern[18:22] = base_load * peak_factor * 1.2
                load_pattern[0:6] = base_load * 0.7

                np.random.seed(42)
                load_data['load_kw'] = load_data['hour'].apply(lambda h: load_pattern[h])
                load_data['load_kw'] = load_data['load_kw'] * np.random.uniform(0.9, 1.1, len(load_data))

                load_data.drop(columns=['hour'], inplace=True)
            else:
                # 若无法从timestamp提取小时，采用正弦波负荷模拟
                t = np.arange(len(load_data))
                load_data['load_kw'] = 200 + 50 * np.sin(2 * np.pi * t / (24 * len(t)/90)) + 20 * np.random.randn(len(t))
                load_data['load_kw'] = load_data['load_kw'].clip(lower=100)
        else:
            # 如果price_data 里没timestamp信息，用更简单的正弦波模拟
            t = np.arange(len(price_data)) if price_data is not None else np.arange(24)
            load_data = pd.DataFrame({
                'load_kw': 200 + 50 * np.sin(2 * np.pi * t / (24 * len(t)/90)) + 20 * np.random.randn(len(t))
            })
            load_data['load_kw'] = load_data['load_kw'].clip(lower=100)

    # 4) 加载或模拟风电数据
    wind_data = None
    if wind_file and os.path.exists(wind_file):
        wind_data = pd.read_csv(wind_file)
    else:
        # 没有真实风电数据则进行模拟
        if weather_data is not None and 'wind_speed' in weather_data.columns:
            wind_data = pd.DataFrame()
            rated_power = 100  # kW
            cut_in_speed = 3.0
            rated_speed = 12.0
            cut_out_speed = 25.0
            wind_speed = weather_data['wind_speed'].values
            wind_power = np.zeros_like(wind_speed)

            for i, speed in enumerate(wind_speed):
                if speed < cut_in_speed or speed > cut_out_speed:
                    wind_power[i] = 0
                elif speed < rated_speed:
                    wind_power[i] = rated_power * (speed - cut_in_speed)/(rated_speed - cut_in_speed)
                else:
                    wind_power[i] = rated_power

            wind_data['wind_power_kw'] = wind_power
        else:
            # 完全随机模拟
            np.random.seed(43)
            if price_data is not None:
                t = np.arange(len(price_data))
            else:
                t = np.arange(24)  # 若无其他信息，模拟24个时段
            base_pattern = 30 + 20 * np.sin(2 * np.pi * t / (24 * len(t)/90))
            random_variations = 15 * np.random.randn(len(t))
            wind_power = base_pattern + random_variations
            wind_power = np.clip(wind_power, 0, 100)
            wind_data = pd.DataFrame({'wind_power_kw': wind_power})

    # 确保 timestamp 列为datetime
    if weather_data is not None and 'timestamp' in weather_data.columns:
        weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
    if price_data is not None and 'timestamp' in price_data.columns:
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
    if load_data is not None and 'timestamp' in load_data.columns:
        load_data['timestamp'] = pd.to_datetime(load_data['timestamp'])
    if wind_data is not None and 'timestamp' in wind_data.columns:
        wind_data['timestamp'] = pd.to_datetime(wind_data['timestamp'])

    # 检测并处理缺失值
    def fill_missing(df):
        if df is not None and df.isnull().sum().sum() > 0:
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
        return df

    weather_data = fill_missing(weather_data)
    price_data = fill_missing(price_data)
    load_data = fill_missing(load_data)
    wind_data = fill_missing(wind_data)

    return weather_data, price_data, load_data, wind_data
