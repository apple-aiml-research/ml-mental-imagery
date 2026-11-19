#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""vLLM server subprocess and a greedy chat client, for evaluation."""

from __future__ import annotations

import base64
import io
import os
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from openai import OpenAI


def visible_device_ids(gpu_indices) -> list[str]:
    """GPUs for vLLM: ``gpu_indices`` index this process's visible GPUs (None = all of them)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else [str(i) for i in range(torch.cuda.device_count())]
    return devices if gpu_indices is None else [devices[i] for i in gpu_indices]


# CUDA compiler for the kernels FlashInfer compiles when vLLM starts (installed by scripts/install_cuda.sh).
CUDA_HOME = "/usr/local/cuda-12.9"


class VLLMServer:
    def __init__(
        self,
        model_path,
        gpu_ids,
        port,
        max_model_len,
        gpu_memory_utilization=None,
        enforce_eager=False,
        revision=None,
        chat_template=None,
        host="127.0.0.1",
    ):
        self.port = port
        self.command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_path,
            "--host",
            host,
            "--port",
            str(port),
            "--max-model-len",
            str(max_model_len),
            "--dtype",
            "bfloat16",
            "--tensor-parallel-size",
            str(len(gpu_ids)),
            "--no-enable-log-requests",
            "--uvicorn-log-level",
            "warning",
        ]
        if revision is not None:  # a Hugging Face Hub model at a pinned revision
            self.command += ["--revision", revision]
        if chat_template is not None:  # the training prompt, for models whose own template differs
            self.command += ["--chat-template", chat_template]
        if gpu_memory_utilization is not None:
            self.command += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
        if enforce_eager:
            self.command.append("--enforce-eager")
        if torch.cuda.get_device_capability(0)[0] >= 10:
            # Blackwell: SDPA vision attention avoids PTX errors; NCCL replaces the custom all-reduce.
            self.command += [
                "--mm-encoder-attn-backend",
                "TORCH_SDPA",
                "--disable-custom-all-reduce",
            ]
        self.env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE")
            and not k.startswith("TORCHELASTIC_")
        }
        if not os.path.isfile(os.path.join(CUDA_HOME, "bin", "nvcc")):
            raise RuntimeError(
                f"vLLM compiles kernels with {CUDA_HOME}/bin/nvcc, which is missing; run scripts/install_cuda.sh"
            )
        self.env.update(
            CUDA_VISIBLE_DEVICES=",".join(gpu_ids), VLLM_NO_USAGE_STATS="1", CUDA_HOME=CUDA_HOME
        )
        self.process = None

    def start(self, timeout_s: float = 900) -> None:
        # Another server on this port would answer our requests with a different model.
        # SO_REUSEADDR ignores TIME_WAIT sockets left by a previous evaluation but not live listeners.
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("0.0.0.0", self.port))
            except OSError as error:
                raise RuntimeError(f"port {self.port} is in use; set vllm.port") from error
        print(f"[vllm] {' '.join(self.command)}", flush=True)
        self.process = subprocess.Popen(
            self.command, env=self.env
        )  # inherit stdout: piping it can deadlock
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"vLLM server exited with code {self.process.returncode}")
            try:
                if requests.get(f"http://127.0.0.1:{self.port}/health", timeout=5).ok:
                    return
            except requests.ConnectionError:
                pass  # still starting
            time.sleep(5)
        self.stop()
        raise TimeoutError(f"vLLM server not ready after {timeout_s}s")

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.send_signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process = None


def chat_messages(image, user_text: str) -> list[dict]:
    """One user turn with an image and text, as in training."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": user_text},
            ],
        }
    ]


class VLLMClient:
    def __init__(self, port: int, model_path: str, stop_token_ids: list[int]):
        self.client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="unused")
        served = [model.id for model in self.client.models.list().data]
        if served != [model_path]:
            raise RuntimeError(f"port {port} serves {served}, expected {model_path}")
        self.model = model_path
        self.stop_token_ids = stop_token_ids

    def generate(self, messages: list[dict], max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            extra_body={"skip_special_tokens": False, "stop_token_ids": self.stop_token_ids},
        )
        message = response.choices[0].message
        # vLLM's reasoning parser may move the text before </think> into reasoning_content.
        return (getattr(message, "reasoning_content", None) or "") + (message.content or "")

    def generate_batch(self, batch_messages: list[list[dict]], max_tokens: int) -> list[str]:
        with ThreadPoolExecutor(max_workers=len(batch_messages)) as pool:
            return list(
                pool.map(lambda messages: self.generate(messages, max_tokens), batch_messages)
            )
