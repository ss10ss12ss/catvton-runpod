# runpod_catvton_worker

RunPod Serverless worker — CatVTON を RTX 4090 (CUDA) で直接実行するハンドラー。

## アーキテクチャ

```
[ローカル vton-app]  ──HTTP POST──▶  [RunPod Serverless Endpoint]
  (前処理/後処理)                       (CatVTON CUDA 推論)
  base64画像送信                        RTX 4090 / bf16
  結果受信                              30ステップ / 768×1024
```

## Input JSON (`/runsync` or `/run`)

```json
{
  "input": {
    "person_image_base64": "data:image/png;base64,...",
    "cloth_image_base64": "data:image/png;base64,...",
    "mask_image_base64": "data:image/png;base64,...",
    "width": 768,
    "height": 1024,
    "steps": 30,
    "guidance": 2.5,
    "seed": -1
  }
}
```

## Output JSON

```json
{
  "ok": true,
  "result_image_base64": "data:image/png;base64,...",
  "device": "cuda",
  "mixed_precision": "bf16",
  "size": [768, 1024],
  "steps": 30,
  "guidance": 2.5,
  "elapsed_ms": 5200.0,
  "engine": "catvton-runpod-cuda"
}
```

## 環境変数

| 変数                 | 説明                                           | デフォルト                                     |
|----------------------|------------------------------------------------|------------------------------------------------|
| `CATVTON_SRC`        | CatVTON ソースコードのパス                     | `/app/CatVTON`                                 |
| `BASE_MODEL_PATH`    | Stable Diffusion inpainting モデル             | `booksforcharlie/stable-diffusion-inpainting`  |
| `ATTN_REPO`          | CatVTON attention weights リポジトリ            | `zhengchong/CatVTON`                           |
| `ATTN_VERSION`       | Attention バージョン (mix/vitonhd/dresscode)    | `mix`                                          |

## デプロイ

```bash
# Docker ビルド
docker build -t runpod-catvton-worker .

# Docker Hub にプッシュ
docker tag runpod-catvton-worker YOUR_DOCKERHUB/runpod-catvton-worker:latest
docker push YOUR_DOCKERHUB/runpod-catvton-worker:latest
```

RunPod Console → Serverless → New Endpoint:
- Docker Image: `YOUR_DOCKERHUB/runpod-catvton-worker:latest`
- GPU: RTX 4090 (24GB VRAM)
- Max Workers: 1-3
- Idle Timeout: 5-10 min
- Execution Timeout: 300 sec

## RTX 4090 最適化ポイント

- **bf16 混合精度**: RTX 4090 は bf16 ネイティブ対応 → fp16 より安定
- **768×1024 ネイティブ解像度**: CatVTON mix-48k-1024 のトレーニング解像度
- **30ステップ**: RTX 4090 なら約5-8秒で完了（MPS では35秒）
- **use_tf32=True**: CUDA Tensor Core 活用
- **VAE tiling/slicing**: VRAM 24GB 内で安定動作
- **パイプラインキャッシュ**: warm start 時はモデル再ロード不要
