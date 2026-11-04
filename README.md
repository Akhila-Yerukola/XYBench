# XYBench
**XYBench:Can LLMs Respond Pragmatically to Queries with Misconceptions?**

**🏆 Best Resource Paper Award** at EMNLP 2026

[[Paper]](https://arxiv.org/abs/2609.06842) · [[Dataset]](<ADD link>)


## Overview



When people try to solve problems, we often get siloed on a specific method or tool and seek help in executing
an already-chosen approach, rather than the actual goal. For example, a user might ask *"How do I parse XML with regex?"* when their underlying goal is simply to extract data from XML. Answering with a regex may satisfy the literal request, but recommending an XML parser would better address the user's actual problem.

This is known as the **XY problem**. Answering the stated literal request alone can reinforce the misconception rather than address the user's intended goal.


XYBench is a benchmark of 8,115 queries with misconceptions, drawn from technical and everyday domains. Each query contains the user's **stated request** (S-request, what they literally ask for), their **intended request** (I-request, the underlying goal), and three rubrics for what a cooperative response should contain:


1. **S-solution**: solutions to directly address the stated request (or explain why it is infeasible),
2. **S-I-misconception-identification**: identify the misconception between the stated and the intended request,
3. **I-solution**: alternative solutions that pragmatically address the intended request.


We evaluate free-form model responses using an LLM judge, measuring the **presence** of S- and I-solutions, which solution the response **emphasizes**, and whether the **misconception is identified**. We additionally evaluate whether models prefer pragmatic over literal responses in a multiple-choice setting.

## Data


| Subset | Queries | Human reference answers | Source |
|---|---|---|---|
| `stackoverflow` | 1,557 | 2,701 (up to 4 per query) | Stack Overflow threads tagged as XY problems |
| `stackexchange` | 184 | 378 | 17 Stack Exchange sites (electronics, superuser, unix, askubuntu, ...) |
| `manually_curated` | 102 | 102 (one per query) | recent forum, Reddit and chat posts curated by the authors |
| `wikihow` | 6,268 | – | everyday how-to queries constructed from WikiHow articles |


The data is hosted on the Hugging Face Hub as [`TO-DO`](<add link>) and downloaded automatically by the scripts (cached by `huggingface_hub`):


* `{subset}/queries.jsonl`: the benchmark queries with the stated and intended request, the three rubrics, multiple-choice candidates, and source.
* `{subset}/human_reference.jsonl`: human reference answers with their judge outputs and metrics (the human rows of the results table).


Field descriptions are in [docs/DATA.md](docs/DATA.md). To use a local copy instead, pass `--data-dir DIR` (or set `XYBENCH_DATA_DIR`) where `DIR/{subset}/queries.jsonl` exists; `XYBENCH_DATASET` points the scripts at a different Hub repository.


## Setup


Requires Python 3.10+.

```bash
git clone https://github.com/Akhila-Yerukola/XYBench.git
cd XYBench
pip install -r requirements.txt
huggingface-cli login      # or export HF_TOKEN=...; 
```


### API keys


Model calls go through [litellm](https://docs.litellm.ai/docs/providers), which reads each provider's key from the environment. Set the ones you need, either by exporting them:


```bash
export OPENAI_API_KEY=...        # openai/* models
export ANTHROPIC_API_KEY=...     # anthropic/* models (the default judge)
export GEMINI_API_KEY=...        # gemini/* models
export TOGETHERAI_API_KEY=...    # together_ai/* models
```




Models and judges are named by their litellm id, e.g. `openai/gpt-5`, `anthropic/claude-sonnet-4-6`, `gemini/gemini-3.1-pro-preview`, `together_ai/Qwen/Qwen3.5-9B`.


## Evaluating a model
To evaluate a model on a subset, first generate free-form responses, then run the LLM judge, and finally aggregate the results:

```bash
# 1. Generate responses  ->  outputs/{subset}/responses.<model>.<condition>.json
python generate_responses.py --subset stackoverflow --model openai/gpt-5


# 2. Evaluate responses using LLM judge   ->  outputs/{subset}/judgments.<model>.<condition>.<judge>.jsonl
python evaluate_responses.py --subset stackoverflow --responses outputs/stackoverflow/responses.openai__gpt-5.query_only.jsonl


# 3. Compute results (across available subsets, including human reference rows if present)
python compute_results.py
```


Different subsets are: `stackoverflow`, `stackexchange`, `manually_curated`, and `wikihow`



To evaluate several models and subsets in one command, `run_benchmark.py` runs generation, judging, and result computation:


```bash
python run_benchmark.py --subset stackoverflow --models openai/gpt-5 anthropic/claude-sonnet-4-6 openai/gpt-4.1
python run_benchmark.py --subset wikihow stackexchange --models openai/gpt-5 --condition query_only oracle_intent
```


It accepts the same generation flags as `generate_responses.py`, `--steps generate judge mcq score` to pick tasks to run, and re-running the same command only re-runs missing tasks.


### Generation settings


`generate_responses.py` uses the same settings for every subset:


| Setting | Default |
|---|---|
| temperature | 1.0 |
| reasoning effort | `medium` (Claude: extended thinking with a 5000-token budget; GPT-5, Gemini: medium reasoning) |
| verbosity (GPT-5 only) | `medium` |
| max tokens | 10000 for GPT-5 and Claude, 20000 for Qwen3.5 (reasoning tokens count against it), 8000 otherwise |


Unsupported provider parameters are dropped. Use `--temperature`, `--reasoning-effort none|low|medium|high`, `--verbosity`, and `--max-tokens` to override these defaults. The settings are recorded in the `generation` field of each response.



### Oracle Ablation Conditions


`generate_responses.py --condition` controls what the model sees:


| Condition | Prompt |
|---|---|
| `query_only` (default) | the query |
| `oracle_intent` | the query followed by the stated request and the intended request |
| `clear_misconceptions` | the query followed by "please clear any misconceptions in the request" |


### Multiple-choice preference test


Each query also includes two short candidate responses: a literal response addressing only the S-request, and a pragmatic response that addresses the S-request, identifies the misconception, and provides an I-solution. `run_mcq.py` asks the model which response is more helpful and reports the rate at which it prefers the pragmatic response. Candidate responses are presented in randomized order (controlled by `--seed`).


```bash
python run_mcq.py --subset stackoverflow --model openai/gpt-5
```


## Metrics for Evaluation


We use Claude Sonnet 4.5 (`anthropic/claude-sonnet-4-5-20250929`) as the default LLM judge, with extended thinking off, temperature 0.7, and JSON output. Each response is evaluated along three dimensions using the query's reference metadata:


| Evaluation | Uses | Output |
|---|---|---|
| **solution presence** | `stated_solution`, `intended_solution` | extracts the response's answer units and tags each `direct` (S-solution), `alternate` (I-solution), or `other` |
| **misconception identification** | `stated_request`, `intended_request` (`wikihow`: `mistake`, `correct_practice`) | tags each answer unit for whether it identifies the misconception |
| **emphasis** | `stated_solution`, `intended_solution` | whether the response as a whole is organized around the S-solution, the I-solution, balanced between the two, or neither |


Metrics per response:


| Metric | Definition |
|---|---|
| S-soln present | at least one `direct` unit |
| I-soln present | at least one `alternate` unit |
| Emphasis | S-solution / balanced / I-solution / neither (the four rates sum to one) |
| S-I-Miscon. identif. | the misconception is identified (either the stated request is discouraged and the intended request addressed, or the intended request is addressed without entertaining the stated one; for `wikihow`, the user's mistake is named) |
| Fully pragmatic | misconception identified, I-solution present, emphasis on the I-solution |
| Fully cooperative | misconception identified, S- and I-solution present, emphasis balanced or on the I-solution |


`compute_results.py` reports metrics per model and condition. For subsets with human answers, it also reports three reference rows: all answers, accepted answers, and highest-scored answers. Prompts are in `prompts.py`. A different judge can be selected with `--judge`; results are comparable to the paper only with the default judge.



## Citation


```bibtex
@inproceedings{yerukola2026xybench,
 title     = {XYBench: Can LLMs Respond Pragmatically to Queries with Misconceptions?},
 author    = {Yerukola, Akhila and Hwang, Jena D and Zheng, Mingqian and Godsey, Jenna and Kim, Hyunwoo and Pyatkin, Valentina and Hu, Jennifer and Sap, Maarten},
 booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
 year      = {2026}
}
```



## License

The code is licensed under Apache 2.0. The data is licensed under ODC-BY 1.0. The artifacts are intended for research and educational use in accordance with Ai2's [Responsible Use Guidelines](https://allenai.org/responsible-use).
