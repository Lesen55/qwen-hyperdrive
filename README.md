# qwen-hyperdrive — Qwen3.8-27B (qwen35) 専用 128K コンテキスト推論エンジン

## 俺が良い感じにローカルLLM動かしたかったからOpus 5.5 Xhighに書かせたエンジン、普通に早くてビビってる今

`Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_XXS-mtp.gguf` を、このPC（RTX 3080 10GB + Ryzen 5 5600X + DDR4 32GB）で
**128K (131072) トークンのコンテキスト**を確保したまま実用速度で動かすために、C++/CUDA でゼロから書いた推論エンジンです。
既存エンジンのラッパーではありません（llama.cpp は精度・速度比較用のリファレンスとしてのみ flake に入っています）。

## 結果（128K コンテキスト設定、同一マシン、hazkey-server 停止時）

| | llama.cpp (`--fit on`) | llama.cpp（手動チューニング） | **llm-engine** |
|---|---|---|---|
| デコード（短い文脈, temp 1.0, MTP） | 2.07 tok/s | 2.46 tok/s | **36〜42 tok/s** |
| デコード（短い文脈, greedy, MTP） | – | 2.66 tok/s | **38〜44 tok/s** |
| デコード（127K トークン文脈） | – | – | **11〜12.6 tok/s** |
| プリフィル 16K トークン | – | – | **~1100 tok/s** |
| プリフィル 127K トークン | – | 7 tok/s 程度 | **186 秒（683 tok/s）** |
| 短いプロンプト（20 トークン）の処理 | – | – | 約 0.13 秒（キャッシュ再開 0.12 秒） |
| 127K パスキー検索 | – | – | 正解（71432） |

- 手動チューニング版の llama.cpp は全 FFN を CPU に置く構成しか 128K で VRAM に収まらず、CUDA graphs も OOM で無効化が必要でした（`bench-results/BASELINE.md`）。
- 精度（docs テキスト 4×512 トークンの perplexity、同条件同士）: f16 KV 同士で llama.cpp 5.120 / 本エンジン 5.170、4bit KV 同士で 5.148 / 5.129。
- 起動ラッパー `llm-engine` は動作中だけ hazkey-server（IME、約 0.8 GB VRAM）を止め、終了時に戻します（`LLM_ENGINE_STOP_HAZKEY=0` で無効）。

## 使い方

```sh
# OpenAI 互換サーバー（モデルは ~/.cache/huggingface の GGUF を自動検出, ctx=131072）
nix run . -- --port 8080
# または明示
nix run .#server -- -m /path/to/model.gguf -c 131072 --port 8080

curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "こんにちは"}],
  "stream": true,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

エンドポイント: `POST /v1/chat/completions`（stream 対応、`<think>` 部分は `reasoning_content` に分離、
`chat_template_kwargs.enable_thinking: false` で思考なし）、
`POST /v1/completions`、`GET /v1/models`、`GET /health`。
サンプリング既定値はモデル推奨（temperature 1.0, top_k 20, top_p 0.95）。`temperature: 0` で greedy。
**プロンプトキャッシュ**: KV はページ化（2048 位置ブロック）され、会話ごとに保持されます。DeltaNet の再帰状態は巻き戻せないため、
各プロンプトの生成開始位置（最後の `<|im_start|>`）と会話の切り替え時に状態チェックポイントを pinned RAM に保存し、
次のリクエストは最長一致するチェックポイントから再開します（思考部分を落とした履歴・タイトル生成などの割り込み・再生成でも有効、`timings.cache_n`）。

主なオプション（`qwen35-server --help`）:

| オプション | 既定 | 説明 |
|---|---|---|
| `-c` | 131072 | 最大コンテキスト長 |
| `--kv q4\|q3\|q8\|f16` | q4 | KV キャッシュ形式（アダマール回転付き。q3 は Lloyd-Max 3bit、長文脈デコード約 5% 高速・PPL +1.8%） |
| `--cpu-frac auto\|0..1` | auto | ストリーミング層の FFN のうち CPU で計算する割合（auto = 層ごとに量子化型別の実測 CPU 速度と PCIe 帯域で均衡） |
| `--pcie-gbps` | 38 | auto 均衡で仮定する実効 PCIe 帯域 |
| `-t` | 6 | CPU ワーカースレッド（物理コアに固定） |
| `--no-mtp` | | MTP 自己投機デコードを無効化 |
| `--ubatch` | 2048 | プリフィルのチャンク長 |
| `--no-elastic` | | 未使用 KV プールの FFN 層への貸し出しを無効化 |
| `-v` | | 詳細ログ（キャッシュ・貸し出し・TTFT 内訳） |
| `--vram-reserve` | 64 | 空けておく VRAM (MiB) |
| `--spec-relax λ` | 1 | 投機デコードの採択を min(1, λp/q) に緩める（λ>1 で速くなる代わりに分布がわずかに MTP 側へ偏る。λ=4 で約 +13%） |

開発: `nix develop` → `cmake -S . -B build -G Ninja && ninja -C build`。
CLI: `qwen35-engine ppl|gen -m model.gguf --ids tokens.txt`（`nix run .#cli`）。

## 設計

ハードの実測値（`tools/hwprobe.cu`）: DRAM 読み出し 46 GB/s、PCIe H2D 25.6 GB/s、GPU DRAM 703 GB/s。
重み 9.7 GiB + 128K の KV は VRAM 約 8.2 GB に収まらないため、デコード速度は「ホスト側に残る重みをいかに速く読むか」で決まります。

1. **VRAM 計画**（`src/model.cpp: plan_and_load`）: 空き VRAM を 1 つのアリーナとして確保し、KV（Q4: 2.4 GB @128K）・GDN 状態・作業領域を差し引いた残りに
   出力ヘッド → 全アテンション/GDN 射影 → MTP ブロック → FFN 層の順で常駐させます。
2. **ハイブリッド FFN**（デコード時）: 常駐できない FFN 層は行方向に分割し、前半を **CPU（AVX2, `src/cpu_ffn.cpp`）**、後半を
   **PCIe ストリーミング + GPU** で同時に計算します。down 射影は入力次元で分割するので、層ごとの CPU↔GPU 通信は 20 KB の入出力ベクトルのみ。
   同期はホストマップドメモリ上のフラグと `cuStreamWriteValue32/WaitValue32` で行い、ストリーミングはコピー専用ストリームで 1 層先読み（ダブルバッファ）。
   CPU と DMA を同時に走らせることで DRAM 帯域をほぼ使い切ります（PCIe 単独 4.7 tok/s → 6.8 tok/s）。
3. **量子化カーネル**: GGUF の IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS/Q2_K/Q4_K/Q6_K/BF16 を自前実装。
   GPU は int8 活性 + `dp4a` の matvec（全型を「32 要素 int8 + 8 要素毎スケール/オフセット」の共通形式にデコード）、
   CPU は q8_K 形式活性 + `maddubs` で整数ドメインのサブスケール適用。プリフィルは f16 展開 + cuBLAS GEMM。
4. **Gated DeltaNet**（48 層）: conv1d+SiLU、L2 正規化、逐次再帰（状態 128×128/ヘッドをレジスタ保持）を専用カーネルで。
5. **アテンション**（16 層 + MTP）: KV は Walsh–Hadamard 回転後に 4bit 量子化（Q も同じ回転を掛けるので内積は不変、出力は逆回転）。
   デコードは split-K（ヘッド優先レイアウト、lane-per-position スコア）、プリフィルは `mma.sync` の FlashAttention 型カーネル。
6. **MTP 自己投機デコード**: GGUF 同梱の NextN ブロックを連鎖させて最大 3 トークンをドラフトし（長さは採択率と実測ステップ時間から適応的に選択）、本体はまとめて 1 パスで検証
   （ストリーミング重みの読み出しは 1 回分なので採択分がそのまま高速化）。棄却時は GDN 状態をスナップショットとポインタ交換でロールバック。
   サンプリング時はドラフトを MTP 分布からサンプルし、標準的な棄却サンプリング（min(1,p/q)、棄却時は max(0,p−q) から再サンプル）で検証するので出力分布は本体と同一です。
   DeltaNet 状態はピンポンバッファで持ち、部分採択時は保存した入力から採択分だけ再計算します。
7. **Elastic VRAM**: 未使用の KV プールとプリフィル用ワークスペース（ubatch 2048）を、デコード中はストリーミング対象の FFN 層の置き場として貸し出し、必要になったら回収します。
8. **プリフィル**: 長いチャンクは重みを層ごとに一度だけ「int8 値 + 整数サブスケール + 256 要素毎の係数」に展開し、cp.async 3 段パイプライン + ldmatrix の int8 tensor-core GEMM（128×128 タイル、活性化は 256 要素毎 int8）で計算（`QE_GQ=0` で従来の fp16 cuBLAS）。短いチャンク（<192 トークン）は量子化重みに直接かける MMQ。
9. **トークナイザ**（`src/tokenizer.cpp`）: GPT-2 バイトレベル BPE + qwen35 プリトークナイザ（正規表現を手書き実装、Unicode 表は生成）。
   日本語・コード・絵文字を含むテキストで llama.cpp と完全一致。

## 検証ツール

| ツール | 内容 |
|---|---|
| `scripts/check_dequant.py` | 全量子化型の逆量子化を gguf-py と照合（ビット一致） |
| `qe_test_kernels` | GPU matvec / f16 展開を参照実装と照合（全 12 型, B=1..8） |
| `qe_test_cpu` | CPU AVX2 matvec の照合と型別スループット |
| `qe_bench_attn` | 128K KV に対するデコード/プリフィル・アテンションの速度 |
| `qe_tokenize` | トークナイズ/デコード（llama-tokenize との突き合わせ用） |
| `QE_DEBUG_SUMS=1` | 層ごとの中間テンソルを llama.cpp の eval-callback と同形式で出力 |
| `QE_PROFILE=1` | プリフィルの区間別時間 |
| `QE_PROFILE=2` | デコード 1 パスの GPU/CPU/PCIe 内訳（同期なし） |
| `scripts/smoke_test.sh` | 全最適化トグルで貪欲出力が基準構成と一致するか検査 |

## 制限

- 対象アーキテクチャは `qwen35`（Qwen3.5/3.8 系 dense ハイブリッド）のみ。ハード定数（sm_86, AVX2）に最適化。
- 同時処理は 1 リクエスト（サーバーは直列化）。ツール呼び出しの構造化パースは未実装（テキストとして返ります）。
- プリフィルは FlashAttention と GEMM 律速。128K の一括投入には約 3.5 分かかります（会話の続きはプロンプトキャッシュで差分のみ）。
- VRAM の空きが減るとストリーミング量が増えて遅くなります（例: hazkey-server が約 0.8 GB 使用中。止めればその分常駐が増えます）。
