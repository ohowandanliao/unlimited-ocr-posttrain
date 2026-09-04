#!/usr/bin/env python3
"""Serve the Unlimited-OCR base model and trained OCR adapters.

Serve base Unlimited-OCR and two LoRA adapters on one GPU.

The trained route calls the original Unlimited-OCR inference methods with a
PeftModel receiver, so generation goes through the adapter. The base route
temporarily disables the adapter on the same in-memory model.
"""

from __future__ import annotations

import argparse
import tempfile
from contextlib import nullcontext
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer


BASE_MODEL_NAME = "unlimited-ocr-base"
FULL_CE_MODEL_NAME = "unlimited-ocr-full-ce"
TITLE_WEIGHTED_MODEL_NAME = "unlimited-ocr-title-weighted"
MODEL_NAMES = (BASE_MODEL_NAME, FULL_CE_MODEL_NAME, TITLE_WEIGHTED_MODEL_NAME)
TRAINED_MODEL_NAMES = frozenset({FULL_CE_MODEL_NAME, TITLE_WEIGHTED_MODEL_NAME})
ADAPTER_NAMES = {
    FULL_CE_MODEL_NAME: "full_ce",
    TITLE_WEIGHTED_MODEL_NAME: "title_weighted",
}


class InferRequest(BaseModel):
    model: str = BASE_MODEL_NAME
    image_paths: list[str] = Field(min_length=1)
    prompt: str | None = None
    max_length: int = Field(default=20480, ge=1, le=32768)
    no_repeat_ngram_size: int = Field(default=35, ge=0, le=128)
    ngram_window: int | None = Field(default=None, ge=0, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class PeftInferenceAdapter:
    """Expose Unlimited-OCR's custom infer methods on a PeftModel."""

    def __init__(self, peft_model: PeftModel):
        self.peft_model = peft_model
        self.base_model = peft_model.get_base_model()

    def infer(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model.__class__.infer(self.peft_model, *args, **kwargs)

    def infer_multi(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model.__class__.infer_multi(self.peft_model, *args, **kwargs)


class OCRService:
    def __init__(
        self,
        base_model_path: Path,
        adapter_paths: Mapping[str, Path],
        device: str,
    ):
        self.base_model_path = base_model_path.resolve()
        self.adapter_paths = {
            model_name: adapter_path.resolve()
            for model_name, adapter_path in adapter_paths.items()
        }
        self.device = torch.device(device)
        self.lock = Lock()

        if not self.base_model_path.is_dir():
            raise FileNotFoundError(f"base model directory does not exist: {self.base_model_path}")
        missing_adapters = set(ADAPTER_NAMES) - set(self.adapter_paths)
        unknown_adapters = set(self.adapter_paths) - set(ADAPTER_NAMES)
        if missing_adapters or unknown_adapters:
            raise ValueError(
                "adapter paths must contain exactly full-ce and title-weighted; "
                f"missing={sorted(missing_adapters)}, unknown={sorted(unknown_adapters)}"
            )
        for model_name, adapter_path in self.adapter_paths.items():
            if not adapter_path.is_dir():
                raise FileNotFoundError(
                    f"{model_name} adapter directory does not exist: {adapter_path}"
                )

        print(f"[uocr] loading tokenizer: {self.base_model_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        print(f"[uocr] loading base model: {self.base_model_path}", flush=True)
        base_model = AutoModel.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.base_model = base_model.eval().to(self.device)

        first_model_name = FULL_CE_MODEL_NAME
        first_adapter_path = self.adapter_paths[first_model_name]
        first_adapter_name = ADAPTER_NAMES[first_model_name]
        print(f"[uocr] loading LoRA adapter {first_model_name}: {first_adapter_path}", flush=True)
        self.peft_model = PeftModel.from_pretrained(
            self.base_model,
            first_adapter_path,
            adapter_name=first_adapter_name,
            is_trainable=False,
        ).eval()
        for model_name, adapter_path in self.adapter_paths.items():
            if model_name == first_model_name:
                continue
            adapter_name = ADAPTER_NAMES[model_name]
            print(f"[uocr] loading LoRA adapter {model_name}: {adapter_path}", flush=True)
            self.peft_model.load_adapter(
                adapter_path,
                adapter_name=adapter_name,
                is_trainable=False,
            )
        self.peft_model.set_adapter(first_adapter_name)
        self.trained_infer_model = PeftInferenceAdapter(self.peft_model)

        print(
            f"[uocr] ready models={','.join(MODEL_NAMES)} "
            f"device={next(self.base_model.parameters()).device}",
            flush=True,
        )

    @staticmethod
    def _default_prompt(model_name: str, count: int) -> str:
        if count == 1:
            return "<image>document parsing."
        if model_name in TRAINED_MODEL_NAMES:
            return "<image>Multi page merge."
        return "<image>Multi page parsing."

    @staticmethod
    def _validate_paths(image_paths: list[str]) -> list[str]:
        resolved = []
        for raw_path in image_paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise HTTPException(status_code=400, detail=f"image does not exist: {path}")
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
                raise HTTPException(status_code=400, detail=f"unsupported image type: {path}")
            resolved.append(str(path))
        return resolved

    def infer(self, request: InferRequest) -> dict[str, Any]:
        if request.model not in MODEL_NAMES:
            raise HTTPException(
                status_code=400,
                detail=f"model must be one of {sorted(MODEL_NAMES)}, got {request.model!r}",
            )
        image_paths = self._validate_paths(request.image_paths)
        prompt = request.prompt or self._default_prompt(request.model, len(image_paths))
        if "<image>" not in prompt:
            prompt = "<image>" + prompt
        window = request.ngram_window
        if window is None:
            window = 128 if len(image_paths) == 1 else 1024

        with self.lock:
            if request.model in TRAINED_MODEL_NAMES:
                self.peft_model.set_adapter(ADAPTER_NAMES[request.model])
                adapter_context = nullcontext()
                infer_model = self.trained_infer_model
            else:
                adapter_context = self.peft_model.disable_adapter()
                infer_model = self.base_model
            with adapter_context:
                with torch.inference_mode():
                    with tempfile.TemporaryDirectory(prefix="uocr-service-") as output_dir:
                        if len(image_paths) == 1:
                            text = infer_model.infer(
                                self.tokenizer,
                                prompt=prompt,
                                image_file=image_paths[0],
                                output_path=output_dir,
                                base_size=1024,
                                image_size=1024,
                                crop_mode=False,
                                eval_mode=True,
                                max_length=request.max_length,
                                no_repeat_ngram_size=request.no_repeat_ngram_size,
                                ngram_window=window,
                                temperature=request.temperature,
                            )
                            output_tokens = None
                        else:
                            text, output_tokens = infer_model.infer_multi(
                                self.tokenizer,
                                prompt=prompt,
                                image_files=image_paths,
                                output_path=output_dir,
                                image_size=1024,
                                save_results=False,
                                max_length=request.max_length,
                                no_repeat_ngram_size=request.no_repeat_ngram_size,
                                ngram_window=window,
                                temperature=request.temperature,
                            )

        return {
            "model": request.model,
            "prompt": prompt,
            "image_paths": image_paths,
            "num_images": len(image_paths),
            "output_tokens": output_tokens,
            "text": text,
        }


def create_app(service: OCRService) -> FastAPI:
    app = FastAPI(title="Unlimited-OCR local service")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "models": sorted(MODEL_NAMES),
            "device": str(service.device),
            "trained_generation": "peft_model.generate",
            "prompts": {
                "base_single": "<image>document parsing.",
                "base_multi": "<image>Multi page parsing.",
                "full_ce_single": "<image>document parsing.",
                "full_ce_multi": "<image>Multi page merge.",
                "title_weighted_single": "<image>document parsing.",
                "title_weighted_multi": "<image>Multi page merge.",
            },
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "owned_by": "local"}
                for name in sorted(MODEL_NAMES)
            ],
        }

    @app.post("/infer")
    def infer(request: InferRequest) -> dict[str, Any]:
        return service.infer(request)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--full-ce-adapter", type=Path, required=True)
    parser.add_argument("--title-weighted-adapter", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    service = OCRService(
        args.base_model,
        {
            FULL_CE_MODEL_NAME: args.full_ce_adapter,
            TITLE_WEIGHTED_MODEL_NAME: args.title_weighted_adapter,
        },
        args.device,
    )
    uvicorn.run(create_app(service), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
