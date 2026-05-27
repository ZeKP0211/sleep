import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_sample_data(num_users: int = 100, days_per_user: int = 30, output_dir: str = "data"):
    """生成示例睡眠数据用于测试"""

    os.makedirs(output_dir, exist_ok=True)

    # 生成用户ID
    user_ids = [f"user_{i:03d}" for i in range(num_users)]

    # 气味类型映射
    odor_types = ['无', '薰衣草', '沉香', '川芎', '其他']

    # 环境质量标签
    env_quality_labels = ['舒适', '过热', '过冷', '气味差']

    all_env_data = []
    all_static_data = []
    all_sleep_history = []
    all_env_labels = []
    all_control_data = []

    for user_id in user_ids:
        # 静态协变量
        age = np.random.randint(18, 80)
        gender = np.random.choice([0, 1])  # 0:女, 1:男
        bmi = np.random.normal(23, 3)
        health_nose = np.random.choice([0, 1])
        health_asthma = np.random.choice([0, 1])
        health_depression = np.random.choice([0, 1])
        habit_alcohol = np.random.choice([0, 1, 2])  # 0:无, 1:少量, 2:大量
        habit_caffeine = np.random.choice([0, 1, 2])
        habit_exercise = np.random.choice([0, 1, 2])
        habit_screen_time = np.random.uniform(0, 4)  # 小时

        static_data = {
            'user_id': user_id,
            'age': age,
            'gender': gender,
            'bmi': bmi,
            'season': np.random.randint(1, 5),  # 1-4代表四季
            'health_nose': health_nose,
            'health_asthma': health_asthma,
            'health_depression': health_depression,
            'habit_alcohol': habit_alcohol,
            'habit_caffeine': habit_caffeine,
            'habit_exercise': habit_exercise,
            'habit_screen_time': habit_screen_time
        }
        all_static_data.append(static_data)

        # 生成每日数据
        start_date = datetime(2023, 1, 1)
        for day in range(days_per_user):
            current_date = start_date + timedelta(days=day)

            # 睡眠历史数据
            sleep_efficiency = np.random.beta(8, 2)  # 睡眠效率
            sleep_latency = np.random.exponential(15) + 5  # 入睡时间
            deep_sleep_duration = np.random.normal(1.5, 0.3)  # 小时
            awakenings = np.random.poisson(2)
            apnea_index = np.random.exponential(5)

            subjective_sleep_quality = float(np.random.beta(6, 3))  #睡眠质量0~1
            sleep_data = {
                'user_id': user_id,
                'date': current_date.strftime('%Y-%m-%d'),
                'sleep_efficiency': sleep_efficiency,
                'sleep_latency': sleep_latency,
                'deep_sleep_duration': max(0, deep_sleep_duration),
                'awakenings': max(0, awakenings),
                'apnea_index': max(0, apnea_index),
                'subjective_sleep_quality': subjective_sleep_quality,
            }
            all_sleep_history.append(sleep_data)

            # 环境数据 (每小时采样，24小时)
            for hour in range(24):
                timestamp = current_date + timedelta(hours=hour)

                # 基础环境特征
                base_temp = 22 + 3 * np.sin(2 * np.pi * hour / 24)  # 日变化
                temp_noise = np.random.normal(0, 2)
                temp = base_temp + temp_noise

                humidity = np.random.normal(50, 10)
                temp_humidity_interaction = temp * humidity / 100  # 简化交互

                # 人体表面温度/湿度（与环境相关但有偏差）
                body_temp = temp + np.random.normal(0.5, 0.5)  # 通常略高于室温
                body_humidity = humidity + np.random.normal(2, 5)  # 人体表面湿度略高

                odor_type = np.random.choice(odor_types)
                odor_intensity = np.random.uniform(0, 1)
                odor_duration = np.random.uniform(0, 60)  # 分钟
                odor_preference = np.random.uniform(0, 1)

                env_data = {
                    'user_id': user_id,
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'temp': temp,
                    'humidity': humidity,
                    'temp_humidity_interaction': temp_humidity_interaction,
                    'body_temp': body_temp,
                    'body_humidity': body_humidity,
                    'odor_type': odor_type,
                    'odor_intensity': odor_intensity,
                    'odor_duration': odor_duration,
                    'odor_preference': odor_preference
                }
                all_env_data.append(env_data)

                # 环境质量标签 (基于简单规则)
                if temp > 26:
                    label = 1  # 过热
                elif temp < 18:
                    label = 2  # 过冷
                elif odor_intensity > 0.7 and odor_type != '无':
                    label = 3  # 气味差
                else:
                    label = 0  # 舒适

                env_label_data = {
                    'user_id': user_id,
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'env_quality_label': label
                }
                all_env_labels.append(env_label_data)

                # 控制策略数据
                sleep_stage = np.random.choice([0, 1, 2, 3])  # 0:清醒, 1:N1, 2:N2, 3:N3
                time_of_day = hour / 24.0

                # 动作
                action_discrete = np.random.randint(0, 3)  # 0:无动作, 1:开香薰, 2:调空调
                action_temp_adjust = np.random.normal(0, 1)
                action_humidity_adjust = np.random.normal(0, 5)

                control_data = {
                    'user_id': user_id,
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'temp': temp,
                    'humidity': humidity,
                    'odor_intensity': odor_intensity,
                    'sleep_stage': sleep_stage,
                    'time_of_day': time_of_day,
                    'action_discrete': action_discrete,
                    'action_temp_adjust': action_temp_adjust,
                    'action_humidity_adjust': action_humidity_adjust
                }
                all_control_data.append(control_data)

    # 保存到CSV
    pd.DataFrame(all_env_data).to_csv(f"{output_dir}/env_features.csv", index=False)
    pd.DataFrame(all_static_data).to_csv(f"{output_dir}/static_covariates.csv", index=False)
    pd.DataFrame(all_sleep_history).to_csv(f"{output_dir}/sleep_history.csv", index=False)
    pd.DataFrame(all_env_labels).to_csv(f"{output_dir}/env_labels.csv", index=False)
    pd.DataFrame(all_control_data).to_csv(f"{output_dir}/control_data.csv", index=False)

    print(f"生成示例数据完成,共 {num_users} 个用户，{days_per_user} 天数据")
    print(f"文件保存在 {output_dir}/ 目录下")


if __name__ == "__main__":
    generate_sample_data(num_users=50, days_per_user=7)  # 测试数据