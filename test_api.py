#!/usr/bin/env python3
"""API 全端点测试脚本。

使用方式:
    python test_api.py                          # 默认 http://localhost:8000
    python test_api.py --base-url http://1.2.3.4:8000

依赖: 标准库 urllib（无需安装额外依赖）。
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from typing import Any


def request(method: str, url: str, body: dict | None = None) -> tuple[int, Any]:
    """发送 HTTP 请求，返回 (状态码, 解析后的 JSON)。"""
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8") if e.fp else ""
        return e.code, body_text


def test_health(base_url: str) -> bool:
    """测试 /v1/health"""
    print("─" * 50)
    print("1. GET /v1/health")
    code, data = request("GET", f"{base_url}/v1/health")
    print(f"   状态码: {code}")
    if isinstance(data, dict):
        print(f"   响应: status={data.get('status')}, models={data.get('models')}")
        ok = data.get("status") == "ok" and len(data.get("models", [])) > 0
        print(f"   结果: {'[OK] 通过' if ok else '[FAIL] 失败（models 为空）'}")
        return ok
    print(f"   结果: [FAIL] 失败 - {data}")
    return False


def test_env_quality(base_url: str) -> bool:
    """测试 /v1/predict/env-quality"""
    print("─" * 50)
    print("2. POST /v1/predict/env-quality")
    body = {
        "temp": 26.5,
        "humidity": 55.0,
        "temp_humidity_interaction": 0.0,
        "body_temp": 27.0,
        "body_humidity": 57.0,
        "odor_type": "薰衣草",
        "odor_intensity": 0.3,
        "odor_duration": 30.0,
        "odor_preference": 0.8,
    }
    code, data = request("POST", f"{base_url}/v1/predict/env-quality", body)
    print(f"   状态码: {code}")
    if code == 200 and isinstance(data, dict):
        comfort = data.get("comfort", "N/A")
        class_label = data.get("class_label", "N/A")
        risk = [f"{r:.2%}" for r in data.get("risk_probs", [])]
        print(f"   舒适度: {comfort:.4f}" if isinstance(comfort, (int, float)) else f"   舒适度: {comfort}")
        print(f"   分类: {class_label}")
        print(f"   风险: {risk}")
        print("   结果: [OK] 通过")
        return True
    print(f"   结果: [FAIL] 失败 - {data}")
    return False


def test_sleep_impact(base_url: str) -> bool:
    """测试 /v1/predict/sleep-impact"""
    print("─" * 50)
    print("3. POST /v1/predict/sleep-impact")
    body = {
        "env_history": [
            {"temp": 26.5, "humidity": 55.0, "temp_humidity_interaction": 0.0, "odor_intensity": 0.3}
        ],
        "static_features": {
            "age": 30, "gender": 1, "bmi": 22.5, "season": 2,
            "health_nose": 0, "health_asthma": 0, "health_depression": 0,
            "habit_alcohol": 0, "habit_caffeine": 1, "habit_exercise": 2, "habit_screen_time": 6,
        },
        "prev_sleep": {
            "sleep_efficiency": 0.85, "sleep_latency": 15, "deep_sleep_duration": 2.5,
            "awakenings": 2, "apnea_index": 1.5, "subjective_sleep_quality": 7,
        },
    }
    code, data = request("POST", f"{base_url}/v1/predict/sleep-impact", body)
    print(f"   状态码: {code}")
    if code == 200 and isinstance(data, dict):
        for k in ["sleep_efficiency", "sleep_latency", "deep_sleep_duration",
                   "awakenings", "apnea_index", "subjective_sleep_quality"]:
            print(f"   {k}: {data.get(k, 'N/A')}")
        print("   结果: [OK] 通过")
        return True
    print(f"   结果: [FAIL] 失败 - {data}")
    return False


def test_control_policy(base_url: str) -> bool:
    """测试 /v1/predict/control-policy (确定性 + 采样)"""
    print("─" * 50)
    print("4. POST /v1/predict/control-policy")
    body = {
        "state_history": [
            {"temp": 26.5, "humidity": 55.0, "odor_intensity": 0.3, "sleep_stage": 2, "time_of_day": 22}
        ],
    }

    # 子测试 4a: 确定性模式 (默认)
    code, data = request("POST", f"{base_url}/v1/predict/control-policy", body)
    print(f"   4a (确定性, 默认) 状态码: {code}")
    if code != 200 or not isinstance(data, dict):
        print(f"   结果: [FAIL] 失败 - {data}")
        return False
    print(f"   discrete_action: {data.get('discrete_action', 'N/A')}")
    print(f"   discrete_action_label: {data.get('discrete_action_label', 'N/A')}")
    print(f"   continuous_action: {data.get('continuous_action', 'N/A')}")
    print(f"   state_value: {data.get('state_value', 'N/A')}")
    det_action = data.get("discrete_action")

    # 子测试 4b: 随机采样模式
    body_sampling = {**body, "deterministic": False}
    code2, data2 = request("POST", f"{base_url}/v1/predict/control-policy", body_sampling)
    print(f"   4b (随机采样) 状态码: {code2}")
    if code2 != 200 or not isinstance(data2, dict):
        print(f"   结果: [FAIL] 失败 - {data2}")
        return False
    print(f"   discrete_action: {data2.get('discrete_action', 'N/A')}")
    print(f"   continuous_action: {data2.get('continuous_action', 'N/A')}")
    print(f"   state_value: {data2.get('state_value', 'N/A')}")
    print("   结果: [OK] 通过")
    return True


def main():
    parser = argparse.ArgumentParser(description="睡眠预测服务 API 全端点测试")
    parser.add_argument(
        "--base-url", default="http://localhost:8000",
        help="API 服务地址（默认: http://localhost:8000）",
    )
    args = parser.parse_args()

    print(f"[TEST] 测试目标: {args.base_url}")
    print()

    base_url = args.base_url.rstrip("/")
    results = []

    # 按顺序执行测试
    for test_fn in [test_health, test_env_quality, test_sleep_impact, test_control_policy]:
        try:
            ok = test_fn(base_url)
            results.append(ok)
        except urllib.error.URLError as e:
            print(f"   结果: [FAIL] 连接失败 - {e.reason}")
            results.append(False)
        except Exception as e:
            print(f"   结果: [FAIL] 异常 - {e}")
            results.append(False)

    # 汇总
    print()
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    labels = ["健康检查", "环境质量", "睡眠影响", "控制策略"]
    for i, (label, ok) in enumerate(zip(labels, results)):
        print(f"  [{i+1}] {label}: {'[OK] 通过' if ok else '[FAIL] 失败'}")
    print(f"  总计: {passed}/{total} 通过")
    print("=" * 50)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
