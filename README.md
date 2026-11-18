# qwen-hyperdrive — Qwen3.5 / 3.8 系（qwen35）向け 128K コンテキスト推論エンジン

## 俺が良い感じにローカルLLM動かしたかったからOpus 5.5 Xhighに書かせたエンジン、普通に早くてビビってる今

`Huihui-Qwen3.8-27B-abliterated-GSQ-RCO-IQ3_XXS-mtp.gguf` を、このPC（RTX 3080 10GB + Ryzen 5 5600X + DDR4 32GB）で
**128K (131072) トークンのコンテキスト**を確保したまま実用速度で動かすために、C++/CUDA でゼロから書いた推論エンジンです。
今は Qwen3.5 / 3.8 系の GGUF 全般に対応し、NVIDIA 向けの CUDA バックエンドに加えて、AMD Radeon / Intel Arc なども動く Vulkan バックエンドを持ちます
（Intel Arc 130V の実機で調整済み、[対応範囲](#対応範囲)）。
既存エンジンのラッパーではありません（llama.cpp は精度・速度比較用のリファレンスとしてのみ flake に入っています）。

## 結果（128K コンテキスト設定、同一マシン、hazkey-server 停止時）

`scripts/bench.sh` で計測（2026-11-18、生データは `bench-results/BENCH.md`）。

**短い文脈のデコード**（500 トークン生成、日本語の解説 / 英語の物語 / コードの書き換えの 3 プロンプト ×（greedy + temp 0.7 × 3 シード）= 12 回）

| 設定 | 最速 | 平均 | 最遅 |
|---|---|---|---|
| llama.cpp（`--fit on` / 手動チューニング） | | 2.07 / 2.46〜2.66 tok/s | |
| **速度優先**（`--kv q3 --spec-relax 2 --skip-ffn 29,10`、サーバーの既定運用） | **103.4 tok/s** | **71.9 tok/s** | **57.8 tok/s** |
| **厳密**（q4 KV、厳密な投機サンプリング、層の省略なし） | 104.9 tok/s | 69.7 tok/s | 49.7 tok/s |

最速はコードの書き換え（MTP と n-gram ドラフトがよく当たる）、最遅は日本語の自由な文章生成です。前回（2026-11-17）は速度優先で 80.5 / 64.2 / 54.2 tok/s。MTP 連鎖の上限を 3 → 5 に広げたことで、コードの生成が大きく伸びました。

**プリフィルと長文脈デコード**（速度優先設定、temp 0.7、200 トークン生成、2026-11-18 再計測）

| 文脈長 | プリフィル時間 | プリフィル速度 | デコード速度 |
|---|---|---|---|
| 2K | 1.7 s | 1189 tok/s | 86.8 tok/s |
| 8K | 6.4 s | 1276 tok/s | 83.0 tok/s |
| 32K | 29.2 s | 1122 tok/s | 69.3 tok/s |
| 64K | 68.3 s | 960 tok/s | 55.7 tok/s |
| 127K | 162.1 s | 785 tok/s | 45.0 tok/s |
| 127K（q4 KV・厳密、前回計測） | 169.2 s | 752 tok/s | 26.0 tok/s |

前回（同日の改善前）は 32K / 64K / 127K でプリフィル 1051 / 861 / 641 tok/s、デコード 62.5 / 51.6 / 34.2 tok/s。FlashAttention を 1 SM に 2 ブロック載る形にして KV 展開と Tensor Core 演算を重ね、q3 KV の展開をバイト置換にし、検証パス（最大 8 トークン）の KV 読み出しを 1 回にした効果です。

- 127K パスキー検索は両設定とも正解（前回計測時に確認）。q3 KV は KV プールが小さく（1.9 vs 2.4 GB）その分 FFN 層を VRAM に置けるのでデコードが速い。
- 長文脈のデコードは、VRAM に入らない FFN の重み（127K で毎回約 2.5 GB）をホストから読む速さで決まります（PCIe 25 GB/s、DRAM 45 GB/s を DMA と CPU で共有）。1 回の検証で確定するトークン数（MTP / n-gram の当たり方）で速度が大きく変わり、同じ 127K でも文章によって 30〜58 tok/s の幅があります。
- 会話の続きはプロンプトキャッシュで差分だけを処理します（短い追加入力の TTFT は約 0.12 秒）。

- 手動チューニング版の llama.cpp は全 FFN を CPU に置く構成しか 128K で VRAM に収まらず、CUDA graphs も OOM で無効化が必要でした（`bench-results/BASELINE.md`）。
- 精度（docs テキスト 4×512 トークンの perplexity、同条件同士）: f16 KV 同士で llama.cpp 5.120 / 本エンジン 5.170、4bit KV 同士で 5.148 / 5.129。
- 起動ラッパー `llm-engine` は `~/.config/llm-engine/tunnel-token`（Cloudflare トンネルのトークン）があればサーバーと一緒に `cloudflared tunnel run` を起動・停止します（`LLM_ENGINE_TUNNEL=0` で無効）。
- 起動ラッパー `llm-engine` は動作中だけ hazkey-server（IME、約 0.8 GB VRAM）を止め、終了時に戻します（`LLM_ENGINE_STOP_HAZKEY=0` で無効）。

## 対応範囲

| | 対応 |
|---|---|
| モデル | GGUF の `qwen35` アーキテクチャ（Qwen3.5 / 3.6 / 3.8 系の DeltaNet + ゲート付き注意のハイブリッド。サイズ・層数・ヘッド数は GGUF から読む。注意ヘッド次元 256、DeltaNet 状態 128 が前提）。MTP（NextN）層があれば自己投機デコードに使い、なければ自動で通常デコード |
| 量子化 | IQ1_S / IQ1_M / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS / Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_0 / BF16 / F16 / F32（Q4_K_M・Q5_K_M・Q8_0・IQ 系の混在 GGUF をそのまま読める） |
| GPU（CUDA） | NVIDIA Ampere 以降（compute 8.0 以上: RTX 30 / 40 系、A100、H100 など）。Nix パッケージは sm_80 / 86 / 89 / 90 の機械語と PTX を同梱。VRAM 容量に合わせて常駐・ストリーミング・KV を自動配置。NVIDIA ではこちらが最速 |
| GPU（Vulkan） | Vulkan 1.2 の GPU なら何でも: AMD Radeon（RADV / AMDVLK）、Intel Arc（ANV）、NVIDIA など。必要なのはバッファデバイスアドレス、8 レーン以上のサブグループ（算術・クラスタ演算）、32 KiB の共有メモリだけ。`VK_KHR_cooperative_matrix` があればプリフィルの行列積を行列演算器（RDNA3 WMMA / Arc XMX / Tensor Core）で行う（起動時に自己テストし、合わなければスカラーに戻る） |
| CPU | x86-64-v3（AVX2 / FMA / F16C / BMI2）が最低限。CPU カーネルは AVX2・AVX-VNNI・AVX-512 VNNI 版を同梱し、起動時に CPU を判定して選ぶ（`[hw] CPU kernels: ...`、`QE_CPU_ISA=avx2/vnni/avx512` で指定）。スレッドは物理コアごとに 1 本、Intel のハイブリッド CPU では P コア優先で配置する。物理コア数・PCIe 帯域・DRAM 帯域は起動時に実測して CPU/GPU 分担に使う（`-t` / `--pcie-gbps` で上書き可）。手元向けに最適化するビルドは `cmake -DQE_NATIVE=ON`（`-march=native`） |

| ループモデル | `sensenova_loop`（[SenseNova-Loop-9B GGUF](https://huggingface.co/guideboardlabs/SenseNova-Loop-9B-GGUF)、CUDA のみ）: 32 層中 24 層を 2 回通す 56 段の実行（2 回目は 1 回目の重みを共有し VRAM は 1 回分、DeltaNet 状態と KV は実行位置ごと）、2 本の残差ストリームと各サブレイヤーのルーター（mHC: RMS → 8 ロジット → 入力ゲート / 出力ゲート / Sinkhorn 20 回の 2×2 混合）、YaRN RoPE。MTP が無いので n-gram 投機デコードで高速化 |

検証済み: Huihui-Qwen3.8-27B IQ3_XXS-mtp（10 GB に収まらないためストリーミング + CPU 併用）、Qwen3.5-9B Q4_K_M MTP（全層 VRAM 常駐）。
Qwen3.5-9B Q4_K_M（RTX 3080）: 短い文脈のデコード 141〜259 tok/s、32K プリフィル 9.7 秒（3376 tok/s）。
SenseNova-Loop-9B Q4_K_M（RTX 3080、CUDA）: デコード 57 tok/s（56 段ぶんの重みを毎トークン読む）、プリフィル 400〜1200 tok/s。MTP ヘッドが無いので、同じトークナイザの小さいモデルを下書きにする投機デコード（`--draft-model`、起動ラッパーでは `LLM_ENGINE_DRAFT_MODEL`）に対応:

| SenseNova-Loop-9B（RTX 3080） | 下書きなし（n-gram のみ） | + Qwen3.5-0.8B Q8_0 下書き |
|---|---|---|
| 物語（greedy / temp 0.7） | 57.6 / 56.5 tok/s | 75.2 / 68.7 tok/s |
| 日本語の解説 | 59.0 / 55.7 tok/s | 71.7 / 72.5 tok/s |
| コードの書き換え | 74.9 / 73.5 tok/s | 117.3 / 114.6 tok/s |
| サーバーの会話 | 64〜65 tok/s | 66〜88 tok/s |

下書きモデルは本体より先に必要な分だけ VRAM を取り（0.8B で約 1 GB）、下書き前の DeltaNet 状態を GPU 内に退避して外れた分を巻き戻す。検証は MTP と同じ仕組み（温度ありでは下書きの分布 q による厳密な棄却サンプリング）。Qwen3.5-2B を下書きにすると、1 トークンが重い分かえって遅かった。ループ 2 回目を外すと PPL が 8.4 → 17.7、最後の 4 層を外すと 481 になることで構造の正しさを確認（参照実装 GuideboardLabs/llama.cpp-sensenova-loop の route / merge と一致させた実装）。

### Vulkan バックエンド

`nix build .#vulkan` で CUDA を含まない Vulkan 専用パッケージ（クロージャ約 100 MiB）を作れる。既定パッケージは CUDA と Vulkan の両方を含み、`llm-engine` は NVIDIA ドライバがあれば CUDA、なければ Vulkan を選ぶ（`LLM_ENGINE_BACKEND=cuda|vulkan` で指定、`QE_VK_DEVICE=<番号>` で GPU を選択）。

- 重みは GGUF の形式のままシェーダー内で展開（17 形式すべて、CPU 参照実装とビット一致）。VRAM に入らない FFN 層は GPU がシステム RAM から PCIe 越しに読む
- KV キャッシュは q8（`--kv f16` も可。q4 / q3 は未実装で q8 になる）。DeltaNet 状態のチェックポイントで会話の続きを再利用
- MTP 自己投機デコードも Vulkan で動く。検証パスは複数トークン用の行列ベクトル積（活性化を f16 で共有メモリに置き、行ごとの逆量子化を全トークンで再利用する）で行い、DeltaNet 状態は受理分だけ後から進める（巻き戻し）。下書き用の語彙は CUDA 版と同じく絞り込む。CPU 併用は CUDA 版のみ
- 移植性の確認: RTX 3080（NVIDIA Vulkan ドライバ）と Mesa lavapipe（CPU 実装、8 レーンのサブグループ）の両方で CUDA 版と同じトークン列を出力。`qe_vk_test <model.gguf>` でシェーダーの単体テスト

| Qwen3.5-9B Q4_K_M, RTX 3080 | CUDA | Vulkan |
|---|---|---|
| デコード（短い文脈、MTP あり） | 141〜259 tok/s | 75〜122 tok/s（MTP なし 60） |
| デコード（32K 文脈） | — | 41 tok/s |
| プリフィル 2K / 32K | — / 3376 tok/s | 755 / 471 tok/s |
| perplexity（同一テキスト 2K） | 6.527 | 6.540 |

#### GPU ごとの最適化（Intel Arc 130V / AMD RX 7900 XTX ほか）

起動時に GPU を判別し（`[hw] ... profile amd-rdna3 / intel-xe2-lunarlake / nvidia ...`）、次を GPU ごとに決める。

| | RX 7900 XTX（RDNA3、24 GB） | Arc 130V（Lunar Lake Xe2 iGPU、システム RAM 32 GB 共有） |
|---|---|---|
| デコードの行列ベクトル積 | ワークグループ形状 × wave32 / wave64 を起動時に実測して選択（1 トークン用と、MTP 検証の複数トークン用を別々に） | ワークグループ形状 × SIMD16 / SIMD32 を実測して選択（同上） |
| プリフィルの行列積 | WMMA 16×16×16（cooperative matrix）、スカラー版と実測比較 | XMX 8×16×16 を優先（ドライバが出す形から選ぶ）、スカラー版と実測比較 |
| KV キャッシュ | `--kv q4/q3` 指定時、重み + f16 KV が 24 GB に収まれば f16（逆量子化なし・誤差なし）、だめなら q8 | 常に q8（帯域が細いので半分のバイト数） |
| メモリ配置 | 全重みを VRAM に置き、溢れた分だけシステム RAM から PCIe で読む。ReBAR / SAM があれば重みを直接書き込む（ステージングなし） | GPU メモリ = システム RAM なので、OS 用に 4 GiB 以上（RAM の 1/8）を残して予算を組む。PCIe 退避はせず、足りなければ文脈を縮めるか明確なエラーで止める。転送済みの重みはモデルファイルのページキャッシュから捨てる（27B なら約 10 GB の二重保持を防ぐ） |
| 分割数（split-K / 注意機構） | CU 数（`VK_AMD_shader_core_properties`、96 CU）× 6 のワークグループを目標 | 小さい iGPU（8 Xe コア）向けに少なめ（128） |

自動チューナーは、そのモデルの行列で候補を実測し、結果を `~/.cache/qwen-hyperdrive/vk-tune-<vendor>-<device>-<driver>-<形状>.txt` に保存する。2 回目以降の起動はこの保存値を読むだけ。`QE_VK_RETUNE=1` で測り直し、`QE_VK_TUNE=teams,rows,subgroup,multi_teams,multi_rows,gemm` で手動指定できる。行列演算の各形は起動時に自己テストし、結果が合わない形は使わない。

Arc 130V（ThinkPad T14 Gen 6、Core Ultra 5 228V、LPDDR5X 32 GB、Mesa ANV）での実測:

| | 最初 | 調整後 |
|---|---|---|
| Qwen3.5-9B Q4_K_M デコード（MTP なし / あり） | 8.2 / 5.0 tok/s | 12.0 / 14.5 tok/s |
| 行列ベクトル積の実効帯域（Q8_0 / Q4_K / Q6_K） | 52 / 41 / 28 GB/s | 91 / 60 / 72 GB/s |
| MTP 検証（4 トークン）の重さ（1 トークン比） | 4.5〜6.5 倍 | 1.6〜1.8 倍 |
| Qwen3.8-27B IQ3_XXS | プリフィルで GPU リセット | 5.5〜6.2 tok/s（MTP あり）、RAM は 17.6 GB 空き |

- 効いたのは、32 レーンで 1 つのスーパーブロックを協調して展開する行列ベクトル積（`mvc_*`）。Intel では、レーンごとにばらけたブロックを読む汎用カーネルが遅かった。
- プリフィルは、9B では XMX（8×16×16）が選ばれる（64 ms、スカラー版 104 ms）。
- 27B が落ちたのは、1 回の投入が Xe ドライバのジョブ時間制限を超えていたため。投入を分割して直した。
- ユニファイドメモリ（Lunar Lake）向け:
  - FFN の gate/up の一部の行を CPU（P コア 4 本、AVX-VNNI）が同じメモリから直接計算し、GPU と同時に進める（1 回の投入の中で、ホストフラグと Vulkan イベントで同期）。分担はパスごとの実測で自動選択。
  - 効果は最良で +7%（11.8 → 12.6 tok/s）。CPU と GPU が同じメモリ帯域を取り合うため大きくは伸びない。
  - モデルファイルのマップ領域をそのまま GPU に見せるゼロコピーは、ANV が読み取り専用のファイルページを取り込めず、失敗時にデバイスごと失われるため、既定では使わない（`QE_VK_ZEROCOPY=1` で試せる）。
- CPU スレッドの既定は、ハイブリッド CPU では P コアだけ（Core Ultra 5 228V: P コア 4 本で 57.6 GB/s、E コアを含めた 8 本では 19.3 GB/s）。
- RX 7900 XTX の実機ではまだ測っていない。

Qwen3.8-27B IQ3_XXS も Vulkan で CUDA と同じ出力になる。10 GB では FFN 5 層がシステム RAM 側になるが、MTP で 9.2 → 16.3 tok/s。帯域が足りない環境ほど MTP が効く（Arc 130V もこのケース）。NVIDIA では CUDA 版を使うこと。

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

**262K コンテキスト（オプション）**: `-c 262144`（または `LLM_ENGINE_CTX=262144 llm-engine`）で Qwen3.8-27B のネイティブ上限まで使える。KV（q3）に 3.8 GB を予約する分、GPU 常駐の FFN が減り、短文デコードは約 1 割遅くなるので既定は 131072 のまま。実測（RTX 3080、`--kv q3 --spec-relax 2 --skip-ffn 29,10`、temp 0.7、2026-11-18）:

| 入力 | プリフィル | デコード |
|---|---|---|
| 短文（物語 / 日本語 / コード） | – | 53.2 / 52.6 / 105.2 tok/s |
| 32K | 31.1 s（1054 tok/s） | 66.1 tok/s |
| 127K | 198.7 s（641 tok/s） | 30.9 tok/s |
| 254K | 605.2 s → 459.1 s（421 → 554 tok/s、FlashAttention 改善後） | 26.0 → 23.7 tok/s |

**WebUI**: ブラウザで `http://<host>:8080/` を開くと、チャット（🌐 Web検索: `web_search` ツールとしてモデルに渡し、必要なときだけモデルが呼び出す。DuckDuckGo / Brave の結果と上位ページ本文、出典リンク付き）と、GPU/CPU の温度・電力・使用率・VRAM・PCIe 帯域、プリフィル/デコード速度（最速・平均つき）・TTFT・MTP 採択率・KV/貸し出し層のリアルタイム表示（`GET /api/stats`）が使えます。

エンドポイント: `POST /v1/chat/completions`（stream 対応、`<think>` 部分は `reasoning_content` に分離、
`chat_template_kwargs.enable_thinking: false` で思考なし、モデル名 `qwen-hyperdrive-fast`（思考なし）/ `qwen-hyperdrive-low`（短い思考、上限 2048 トークン）でも切替可、`developer` ロールは system 扱い、`tools` によるツール呼び出し（Qwen の XML 形式を解析して OpenAI 形式の `tool_calls` で返す。ストリーミング対応）、`reasoning_effort: xhigh|medium|low`）、
`POST /v1/completions`、`GET /v1/models`、`GET /health`。
サンプリング既定値はモデル推奨（temperature 1.0, top_k 20, top_p 0.95）。`temperature: 0` で greedy。
**プロンプトキャッシュ**: KV はページ化（2048 位置ブロック）され、会話ごとに保持されます。DeltaNet の再帰状態は巻き戻せないため、
各プロンプトの生成開始位置（最後の `<|im_start|>`）と会話の切り替え時に状態チェックポイントを pinned RAM に保存し、
次のリクエストは最長一致するチェックポイントから再開します（思考部分を落とした履歴・タイトル生成などの割り込み・再生成でも有効、`timings.cache_n`）。

主なオプション（`qwen35-server --help`）:

| オプション | 既定 | 説明 |
|---|---|---|
| `-c` | 131072 | 最大コンテキスト長（起動ラッパーでは `LLM_ENGINE_CTX`。27B は最大 262144 まで可、下表） |
| `--kv q4\|q3\|q8\|f16` | q4 | KV キャッシュ形式（アダマール回転付き。q3 は Lloyd-Max 3bit、長文脈デコード約 5% 高速・PPL +1.8%） |
| `--cpu-frac auto\|0..1` | auto | ストリーミング層の FFN のうち CPU で計算する割合（auto = 層ごとに量子化型別の実測 CPU 速度と PCIe 帯域で均衡） |
| `--pcie-gbps` | 実測 × 1.5 | auto 均衡で仮定する実効 PCIe 帯域（起動時に計測） |
| `-t` | 物理コア数 | CPU ワーカースレッド（物理コアに固定） |
| `--no-mtp` | | MTP 自己投機デコードを無効化 |
| `--ubatch` | 2048 | プリフィルのチャンク長 |
| `--no-elastic` | | 未使用 KV プールの FFN 層への貸し出しを無効化 |
| `-v` | | 詳細ログ（キャッシュ・貸し出し・TTFT 内訳） |
| `--vram-reserve` | 64 | 空けておく VRAM (MiB) |
| `--skip-ffn i,j,..` | | 指定層の FFN を丸ごと省く（非可逆）。PCIe 転送量が 1 層あたり約 64 MB 減る。PPL 影響の小さい順は `bench-results/SKIP_FFN_PPL.txt`（例: 29,10 で PPL +0.3% / +8%、29,10,17,16 で +1.3% / +13%） |
| `--cache-keep` / `--cache-host-mb` | 1 / 6144 | 非アクティブな会話の KV は VRAM から host RAM へ退避（VRAM は FFN 層の貸し出しに戻す）。戻ると書き戻してキャッシュ再開 |
| `--no-ngram` / `--draft n` | | n-gram（プロンプト参照）ドラフトを無効化 / MTP 連鎖の最大長（既定 5、16K 超では 7）。n-gram は MTP の 1 トークン目と一致したときだけ最大 7 トークンまで延長 |
| `--spec-relax λ` | 1 | 投機デコードの採択を min(1, λp/q) に緩める（λ>1 で速くなる代わりに分布がわずかに MTP 側へ偏る。λ=2 で約 +10%。n-gram ドラフトには適用しない） |

開発: `nix develop` → `cmake -S . -B build -G Ninja && ninja -C build`。
CLI: `qwen35-engine ppl|gen -m model.gguf --ids tokens.txt`（`nix run .#cli`）。

Vulkan（AMD / Intel など）:

```sh
nix build .#vulkan -o hd                     # CUDA を含まないパッケージ
LLM_ENGINE_BACKEND=vulkan ./hd/bin/llm-engine -m model.gguf --port 8080
./hd/bin/vk-device-report model.gguf ./hd/bin   # 自動調整 + 単体テスト + 速度の報告を 1 ファイルに
# 開発（CUDA をダウンロードしない）: nix develop .#vulkan → cmake -DQE_CUDA=OFF -DQE_VULKAN=ON
```

主な環境変数: `QE_VK_DEVICE`（GPU 番号）、`QE_VK_RETUNE=1`（自動調整のやり直し）、`QE_VK_PROF=1`（シェーダーごとの GPU 時間）、
`QE_VK_HYBRID=0/1`（ユニファイドメモリでの CPU/GPU 同時実行）、`QE_CPU_ISA=avx2/vnni/avx512`（CPU カーネル）。

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
| `qe_test_cpu` | CPU matvec（AVX2 / AVX-VNNI / AVX-512 VNNI、`QE_CPU_ISA` で切替）の照合と型別スループット |
| `qe_vk_test` | Vulkan シェーダーの照合（全 17 型の逆量子化はビット一致、matvec / GEMM / 協調カーネル）。`QE_VK_TEST_BENCH=1` でバッチ幅ごとの速度とメモリ帯域 |
| `vk-device-report` | 対象 GPU で自動調整・単体テスト・デコード / プリフィル速度・シェーダー別時間を測ってファイルにまとめる |
| `qe_bench_attn` | 128K KV に対するデコード/プリフィル・アテンションの速度 |
| `qe_tokenize` | トークナイズ/デコード（llama-tokenize との突き合わせ用） |
| `QE_DEBUG_SUMS=1` | 層ごとの中間テンソルを llama.cpp の eval-callback と同形式で出力 |
| `QE_PROFILE=1` | プリフィルの区間別時間 |
| `QE_PROFILE=2` | デコード 1 パスの GPU/CPU/PCIe 内訳（同期なし） |
| `scripts/smoke_test.sh` | 全最適化トグルで貪欲出力が基準構成と一致するか検査 |
| `scripts/bench.sh` | 上の結果表のベンチマーク（`bench-results/BENCH.md` を生成） |

## 制限

- 対象アーキテクチャは `qwen35`（Qwen3.5/3.8 系 dense ハイブリッド、注意ヘッド次元 256・DeltaNet 状態 128）のみ。
- 動作環境は Linux（x86-64-v3 以上）のみ。
- Vulkan 版は CPU への FFN 分担（ストリーミング + CPU 計算）を持たない。ただしユニファイドメモリでの同時実行はある。KV は q8 / f16 のみ。
- RX 7900 XTX など AMD GPU は、仕組み（判別・自動調整）だけで実機未確認。
- 同時処理は 1 リクエスト（サーバーは直列化、WebUI は順番待ち中に「待機中」と表示）。
- プリフィルは FlashAttention と GEMM 律速。128K の一括投入には約 3.5 分かかります（会話の続きはプロンプトキャッシュで差分のみ）。
- VRAM の空きが減るとストリーミング量が増えて遅くなります（例: hazkey-server が約 0.8 GB 使用中。止めればその分常駐が増えます）。
