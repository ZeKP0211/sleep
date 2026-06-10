"""FastAPI 睡眠预测服务。

对外提供三个预测接口，支持小程序/App 直接调用。

启动方式::

    cd inference
    python api.py --checkpoint-dir ../data/checkpoints --port 8000
    # 或
    uvicorn inference.api:create_app --factory --host 0.0.0.0 --port 8000
"""

import argparse
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .model_loader import ModelManager
from .preprocessing import InferencePreprocessor

logger = logging.getLogger("sleep_api")

# ── 全局单例 ──
_preprocessor: Optional[InferencePreprocessor] = None
_model_manager: Optional[ModelManager] = None


# ── 请求体 Pydantic 模型 ──

class EnvQualityRequest(BaseModel):
    """环境质量预测请求。"""
    temp: float = Field(..., description="室温（℃）")
    humidity: float = Field(..., description="相对湿度（%）")
    temp_humidity_interaction: float = Field(default=0.0, description="温湿度交互项")
    body_temp: float = Field(default=0.0, description="人体表面温度（℃）")
    body_humidity: float = Field(default=0.0, description="人体表面湿度（%）")
    odor_type: str = Field(default="无", description="气味类型：无/薰衣草/沉香/川芎/其他")
    odor_intensity: float = Field(default=0.0, description="气味强度 0~1")
    odor_duration: float = Field(default=0.0, description="气味时长（分钟）")
    odor_preference: float = Field(default=0.0, description="气味喜好度 0~1")


class EnvQualityResponse(BaseModel):
    """环境质量预测响应。"""
    comfort: float = Field(..., description="舒适度分数 0~1")
    risk_probs: list[float] = Field(..., description="风险概率 [无风险, 过热, 过冷, 气味不适]")
    class_probs: list[float] = Field(..., description="分类概率 [优, 良, 中, 差]")
    class_pred: int = Field(..., description="预测类别索引")
    class_label: str = Field(..., description="预测类别标签")


class SleepImpactRequest(BaseModel):
    """睡眠影响预测请求。"""
    env_history: list[dict] = Field(..., description="最近 N 条环境记录")
    static_features: dict = Field(..., description="静态协变量 (age, gender, bmi, ...)")
    prev_sleep: Optional[dict] = Field(default=None, description="前一日睡眠指标")


class SleepImpactResponse(BaseModel):
    """睡眠影响预测响应。"""
    sleep_efficiency: float = Field(..., description="睡眠效率")
    sleep_latency: float = Field(..., description="入睡潜伏期")
    deep_sleep_duration: float = Field(..., description="深睡时长")
    awakenings: float = Field(..., description="觉醒次数")
    apnea_index: float = Field(..., description="呼吸暂停低通气指数")
    subjective_sleep_quality: float = Field(..., description="主观睡眠质量")


class ControlPolicyRequest(BaseModel):
    """控制策略预测请求。"""
    state_history: list[dict] = Field(..., description="最近 N 条状态记录")
    deterministic: bool = Field(default=True, description="True=确定性动作, False=随机采样")


class ControlPolicyResponse(BaseModel):
    """控制策略预测响应。"""
    discrete_action: int = Field(..., description="离散动作索引")
    discrete_action_label: str = Field(..., description="离散动作标签")
    continuous_action: list[float] = Field(..., description="连续动作 [温度调节, 湿度调节]")
    state_value: float = Field(..., description="状态价值 V(s)")


class HealthResponse(BaseModel):
    status: str
    models: list[str]


# ── 可读标签（与 predict.py 一致） ──

_CLASS_LABELS = ["优", "良", "中", "差"]
_RISK_LABELS = ["无风险", "过热风险", "过冷风险", "气味不适风险"]
_DISCRETE_ACTION_LABELS = ["香薰开关", "空调开关", "风扇开关"]


# ── 后处理函数 ──

def _postprocess_env_quality(outputs: dict) -> EnvQualityResponse:
    comfort = float(outputs["comfort_score"].item())
    risk_logits = outputs["risk_logits"].squeeze(0)  # (4,)
    class_logits = outputs["class_logits"].squeeze(0)  # (4,)

    risk_probs = _softmax(risk_logits)
    class_probs = _softmax(class_logits)
    class_pred = int(np.argmax(class_probs))

    return EnvQualityResponse(
        comfort=round(comfort, 6),
        risk_probs=[round(float(p), 6) for p in risk_probs],
        class_probs=[round(float(p), 6) for p in class_probs],
        class_pred=class_pred,
        class_label=_CLASS_LABELS[class_pred],
    )


def _postprocess_sleep_impact(predictions: np.ndarray) -> SleepImpactResponse:
    arr = predictions.squeeze(0)  # (6,)
    return SleepImpactResponse(
        sleep_efficiency=round(float(arr[0]), 6),
        sleep_latency=round(float(arr[1]), 4),
        deep_sleep_duration=round(float(arr[2]), 4),
        awakenings=round(float(arr[3]), 4),
        apnea_index=round(float(arr[4]), 4),
        subjective_sleep_quality=round(float(arr[5]), 4),
    )


def _postprocess_control_policy(outputs: dict) -> ControlPolicyResponse:
    """从 ModelManager 返回的动作字典构建 API 响应。

    ModelManager.predict_control_policy() 已根据 deterministic 参数
    完成了 diff(采样)或确定性(deterministic)动作选择，此处直接使用。
    """
    disc_action = int(outputs["discrete_action"].item())
    cont_action = outputs["continuous_action"].squeeze(0)  # (cont_dim,)
    state_val = float(outputs["state_value"].item())

    return ControlPolicyResponse(
        discrete_action=disc_action,
        discrete_action_label=_DISCRETE_ACTION_LABELS[disc_action],
        continuous_action=[round(float(v), 4) for v in cont_action],
        state_value=round(state_val, 6),
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()


# ── 应用生命周期 ──

def _init_globals(checkpoint_dir: str) -> None:
    global _preprocessor, _model_manager
    _preprocessor = InferencePreprocessor(f"{checkpoint_dir}/preprocessing.pkl")
    _model_manager = ModelManager(checkpoint_dir)
    logger.info("模型与预处理管线已加载（checkpoint_dir=%s）", checkpoint_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时由 create_app 已完成初始化
    yield
    logger.info("服务关闭")


def create_app(checkpoint_dir: str = "data/checkpoints") -> FastAPI:
    """工厂函数：创建 FastAPI 应用并注入依赖。"""
    _init_globals(checkpoint_dir)

    app = FastAPI(
        title="睡眠预测服务",
        description="环境质量评估 · 睡眠影响预测 · 设备控制策略",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS 放开给小程序/App 调用
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ──────────────────────── 路由 ────────────────────────

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            models=_model_manager.loaded_models if _model_manager else [],
        )

    @app.post("/v1/predict/env-quality", response_model=EnvQualityResponse)
    async def predict_env_quality(req: EnvQualityRequest):
        try:
            raw = req.model_dump()
            numeric, odor_idx = _preprocessor.env_quality(raw)
            outputs = _model_manager.predict_env_quality(numeric, odor_idx)
            return _postprocess_env_quality(outputs)
        except Exception as e:
            logger.exception("env_quality 推理失败")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/predict/sleep-impact", response_model=SleepImpactResponse)
    async def predict_sleep_impact(req: SleepImpactRequest):
        try:
            env_seq, static, hist, seq_len = _preprocessor.sleep_impact(
                req.env_history, req.static_features, req.prev_sleep
            )
            preds = _model_manager.predict_sleep_impact(env_seq, static, hist, seq_len)
            return _postprocess_sleep_impact(preds)
        except Exception as e:
            logger.exception("sleep_impact 推理失败")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/predict/control-policy", response_model=ControlPolicyResponse)
    async def predict_control_policy(req: ControlPolicyRequest):
        try:
            state_seq, seq_len = _preprocessor.control_policy(req.state_history)
            outputs = _model_manager.predict_control_policy(
                state_seq, seq_len, deterministic=req.deterministic
            )
            return _postprocess_control_policy(outputs)
        except Exception as e:
            logger.exception("control_policy 推理失败")
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ── 独立启动入口 ──

def main():
    parser = argparse.ArgumentParser(description="睡眠预测 FastAPI 服务")
    parser.add_argument("--checkpoint-dir", type=str, default="data/checkpoints",
                        help="模型与预处理工件目录")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    app = create_app(checkpoint_dir=args.checkpoint_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
