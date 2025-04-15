import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def process_weather_data(input_file='天气数据.csv', output_file='处理后的天气数据.csv'):
    # 读取原始天气数据
    try:
        weather_data = pd.read_csv(input_file, encoding='cp1252')
        print(f"成功读取天气数据文件，包含 {len(weather_data)} 行")
    except Exception as e:
        print(f"读取天气数据文件时出错: {e}")
        print("尝试使用不同的编码...")

        # 尝试不同的编码
        try:
            weather_data = pd.read_csv(input_file, encoding='utf-8')
            print(f"使用UTF-8编码成功读取文件，包含 {len(weather_data)} 行")
        except Exception as e2:
            print(f"使用UTF-8编码读取也失败: {e2}")
            return

    # 查看原始数据结构
    print("\n原始数据的列名:")
    print(weather_data.columns.tolist())

    # 检查数据中是否已有必要的列
    required_columns = ['solar_radiation', 'temperature', 'precipitation']
    missing_columns = [col for col in required_columns if col not in weather_data.columns]

    # 如果缺少必要的列，尝试从现有列中映射
    if missing_columns:
        print(f"\n缺少以下必要列: {missing_columns}")
        print("尝试从现有列中映射...")

        # 创建一个新的DataFrame来存储处理后的数据
        processed_data = pd.DataFrame()

        # 处理时间戳列
        if 'timestamp' not in weather_data.columns:
            if 'date' in weather_data.columns:
                processed_data['timestamp'] = pd.to_datetime(weather_data['date'])
            elif 'Date' in weather_data.columns:
                processed_data['timestamp'] = pd.to_datetime(weather_data['Date'])
            else:
                # 如果没有日期列，创建一个从2024年12月1日开始的日期序列
                start_date = datetime(2024, 12, 1)
                processed_data['timestamp'] = [start_date + timedelta(days=i) for i in range(len(weather_data))]
        else:
            processed_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])

        # 处理太阳辐射列
        if 'solar_radiation' not in weather_data.columns:
            # 尝试从其他可能的列名中找到
            solar_cols = [col for col in weather_data.columns if
                          'solar' in col.lower() or 'radiation' in col.lower() or 'rad' in col.lower()]
            if solar_cols:
                print(f"使用 '{solar_cols[0]}' 作为太阳辐射数据")
                processed_data['solar_radiation'] = weather_data[solar_cols[0]]
            else:
                # 如果找不到相关列，创建合理的模拟数据
                print("未找到太阳辐射相关列，创建合理的模拟数据")
                # 夏季澳大利亚悉尼的太阳辐射在200-1000 W/m²之间
                processed_data['solar_radiation'] = np.random.uniform(200, 1000, len(weather_data))
        else:
            processed_data['solar_radiation'] = weather_data['solar_radiation']

        # 处理温度列
        if 'temperature' not in weather_data.columns:
            temp_cols = [col for col in weather_data.columns if 'temp' in col.lower() or 'temperature' in col.lower()]
            if temp_cols:
                print(f"使用 '{temp_cols[0]}' 作为温度数据")
                processed_data['temperature'] = weather_data[temp_cols[0]]
            else:
                # 夏季澳大利亚悉尼的温度在18-35℃之间
                print("未找到温度相关列，创建合理的模拟数据")
                processed_data['temperature'] = np.random.uniform(18, 35, len(weather_data))
        else:
            processed_data['temperature'] = weather_data['temperature']

        # 处理降水列
        if 'precipitation' not in weather_data.columns:
            precip_cols = [col for col in weather_data.columns if 'precip' in col.lower() or 'rain' in col.lower()]
            if precip_cols:
                print(f"使用 '{precip_cols[0]}' 作为降水数据")
                processed_data['precipitation'] = weather_data[precip_cols[0]]
            else:
                # 大多数天降水量为0，少数天有降水
                print("未找到降水相关列，创建合理的模拟数据")
                precipitation = np.zeros(len(weather_data))
                # 随机选择约20%的天有降水
                rainy_days = np.random.choice(len(weather_data), size=int(len(weather_data) * 0.2), replace=False)
                precipitation[rainy_days] = np.random.uniform(0.1, 30, len(rainy_days))
                processed_data['precipitation'] = precipitation
        else:
            processed_data['precipitation'] = weather_data['precipitation']
    else:
        # 如果所有必要的列都存在，直接使用原始数据
        processed_data = weather_data.copy()
        # 确保timestamp列存在
        if 'timestamp' not in processed_data.columns:
            # 如果没有时间戳列，创建一个从2024年12月1日开始的日期序列
            start_date = datetime(2024, 12, 1)
            processed_data['timestamp'] = [start_date + timedelta(days=i) for i in range(len(processed_data))]

    # 确保数据类型正确
    processed_data['solar_radiation'] = processed_data['solar_radiation'].astype(float)
    processed_data['temperature'] = processed_data['temperature'].astype(float)
    processed_data['precipitation'] = processed_data['precipitation'].astype(float)

    # 检查并处理缺失值
    if processed_data.isnull().sum().sum() > 0:
        print(f"\n处理 {processed_data.isnull().sum().sum()} 个缺失值")
        processed_data = processed_data.fillna(method='ffill').fillna(method='bfill')

    # 保存处理后的数据
    processed_data.to_csv(output_file, index=False)
    print(f"\n处理后的天气数据已保存至 '{output_file}'")
    print(f"数据包含 {len(processed_data)} 行，列: {processed_data.columns.tolist()}")

    # 显示数据前几行
    print("\n处理后的数据前5行:")
    print(processed_data.head())


if __name__ == "__main__":
    process_weather_data()