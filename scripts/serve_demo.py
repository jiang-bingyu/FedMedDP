from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import random
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from fedmeddp.inference import DemoSkinPredictor


def create_app(
    mode: str = "ensemble",
    device: str = "auto",
    max_upload_mb: int = 12,
    preload_models: bool = False,
) -> FastAPI:
    predictor = DemoSkinPredictor(mode=mode, device=device, root=ROOT)
    max_upload_bytes = max_upload_mb * 1024 * 1024

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.startup_error = None
        if preload_models:
            try:
                predictor.load()
            except Exception as exc:  # noqa: BLE001 - keep the demo page reachable.
                app.state.startup_error = str(exc)
        yield

    app = FastAPI(title="FedMedDP 答辩演示服务", lifespan=lifespan)

    @app.middleware("http")
    async def no_cache_headers(request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        payload = predictor.status()
        payload["startup_error"] = app.state.startup_error
        return payload

    @app.post("/api/predict")
    async def predict(image: UploadFile = File(...)) -> dict[str, Any]:
        if image.content_type and not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="请上传图片文件。")

        content = await image.read(max_upload_bytes + 1)
        if not content:
            raise HTTPException(status_code=400, detail="图片文件为空。")
        if len(content) > max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"图片不能超过 {max_upload_mb} MB。")

        try:
            return predictor.predict_bytes(content)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=f"模型推理失败：{exc}") from exc

    data_dir = ROOT / "data"
    if data_dir.exists():
        app.mount("/data", StaticFiles(directory=data_dir), name="data")

    app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
    return app


def is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def pick_port(host: str, port_value: str, preferred_start: int = 8000, preferred_end: int = 8099) -> int:
    port_value = str(port_value).strip().lower()
    if port_value not in {"", "auto"}:
        try:
            preferred = int(port_value)
        except ValueError as exc:
            raise ValueError(f"无效端口：{port_value}") from exc
        if preferred > 0 and is_port_free(host, preferred):
            return preferred
        if preferred > 0:
            candidates = list(range(max(1, preferred), preferred_end + 1))
            random.shuffle(candidates)
            for candidate in candidates:
                if is_port_free(host, candidate):
                    return candidate

    candidates = list(range(preferred_start, preferred_end + 1))
    random.shuffle(candidates)
    for candidate in candidates:
        if is_port_free(host, candidate):
            return candidate

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def browser_watcher(url: str) -> None:
    health_url = f"{url.rstrip('/')}/api/health"
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            with urllib_request.urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(url, new=2)
                    return
        except Exception:
            time.sleep(0.5)


def normalize_display_host(host: str) -> str:
    if host in {"0.0.0.0", "::", ""}:
        return "127.0.0.1"
    return host


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 FedMedDP 答辩演示页面和现场识别接口。")
    parser.add_argument("--host", default="127.0.0.1", help="服务监听地址。")
    parser.add_argument(
        "--port",
        default="auto",
        help="服务端口，默认自动选择空闲端口；也可手动指定如 8000。",
    )
    parser.add_argument(
        "--mode",
        choices=["ensemble", "single"],
        default="ensemble",
        help="ensemble 使用五模型集成；single 使用 seed2030 单模型备用方案。",
    )
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda。")
    parser.add_argument("--max-upload-mb", type=int, default=12, help="上传图片大小限制。")
    parser.add_argument(
        "--preload-models",
        action="store_true",
        help="启动时预加载模型；不加时会在第一次识别时加载。",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="不自动打开浏览器。",
    )
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    port = pick_port(args.host, args.port)
    display_host = normalize_display_host(args.host)
    url = f"http://{display_host}:{port}/"
    print(f"答辩演示地址: {url}", flush=True)
    if not args.no_browser:
        threading.Thread(target=browser_watcher, args=(url,), daemon=True).start()
    uvicorn.run(
        create_app(
            mode=args.mode,
            device=args.device,
            max_upload_mb=args.max_upload_mb,
            preload_models=args.preload_models,
        ),
        host=args.host,
        port=port,
    )


if __name__ == "__main__":
    main()
