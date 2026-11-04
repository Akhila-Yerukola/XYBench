# Data files

The data lives in the Hugging Face dataset [`allenai/XYBench-dataset`](https://huggingface.co/datasets/allenai/XYBench-dataset), laid out as `{subset}/queries.jsonl` and `{subset}/human_reference.jsonl`. The scripts download and cache the files; `--data-dir` points them at a local copy with the same layout. All files are JSONL (one JSON object per line).

## `{subset}/queries.jsonl`

One record per query.

| Field | Description |
|---|---|
| `id` | `so_{post_id}`, `se_{site}_{post_id}`, `mc_{nnn}`, `wh_{article}_{step}_{q1\|q2}` |
| `subset` | `stackoverflow` \| `stackexchange` \| `manually_curated` \| `wikihow` |
| `domain` | `stackoverflow`: `programming` (tags in `source.tags`); `stackexchange`: the site; `manually_curated`: topic (cooking, diy, sewing, woodworking, ...); `wikihow`: `how-to` |
| `query` | the query presented to the model |
| `stated_request` | S-request: what the user literally asks for |
| `intended_request` | I-request: the underlying goal |
| `stated_solution` | rubric describing an S-solution (a direct answer to the stated request) |
| `intended_solution` | rubric describing I-solutions (pragmatic alternatives that address the intended request) |
| `misconception_identification` | rubric describing the misconception between the stated and intended request |
| `misconception_type` | `stackoverflow` / `stackexchange`: type of misconception (e.g. "Incomplete knowledge", "Incorrect path", "Syntax fixation", "Library fixation") |
| `gap` | `wikihow`: `strong` / `medium` / `weak`, how much the user stands to lose if only the stated request is answered |
| `mistake` | `wikihow`: what the user did wrong (used by the misconception-identification judge) |
| `correct_practice` | `wikihow`: the correct practice that addresses the root cause (used by the misconception-identification judge) |
| `misconception_mode` | `request_mismatch` (stackoverflow, stackexchange, manually_curated) or `mistake_practice` (wikihow): which misconception-identification prompt and label set applies |
| `mcq_variants` | `{stated_response, hybrid_response}`: candidate responses for the multiple-choice preference test (null for a few queries) |
| `source` | provenance, see below |

`source` fields by subset:

* **stackoverflow / stackexchange**: `site`, `post_id`, `url`, `title`, `body_html` (the original question), `tags`, `score`, `view_count`, `creation_date`, `accepted_answer_id`, `answer_ids`.
* **manually_curated**: `url` (original post, where public), `query_original` (the post before rewriting, where available).
* **wikihow**: `thread_id`, `mistake_id`, `article_title`, `article_goal`, `steps` (the article's steps), `critical_step_id` (the step the user got wrong), `query_variant` (`query1` / `query2`, two variants per mistake), `reference_urls`, and the pieces the solution rubrics were built from: `direct_solutions_to_x`, `answer_salvage`, `answer_next_time`.

## `{subset}/human_reference.jsonl`

One record per human reference answer (`stackoverflow`, `stackexchange`, `manually_curated`). These carry the judge outputs and metrics for the human rows of the results table; they are not re-evaluated.

| Field | Description |
|---|---|
| `id` | query id |
| `answer_id` | answer id on the source site, or `human` for `manually_curated` |
| `answer_text` | the answer (HTML for Stack Overflow / Stack Exchange) |
| `score`, `is_accepted`, `is_high_score`, `is_highest_score` | answer metadata (`is_highest_score` is null for `stackexchange`) |
| `judge` | judge model |
| `solution_presence`, `misconception_identification`, `emphasis` | judge outputs, in the same format as model judgments |
| `metrics` | `s_solution_present`, `i_solution_present`, `emphasis` (`s_solution` / `balanced` / `i_solution` / `neither`), `misconception_identified`, `fully_pragmatic`, `fully_cooperative` |

For `stackoverflow` and `stackexchange`, `misconception_identified` comes from the annotation of whether the answer identifies the misconception; for `manually_curated` it comes from the `misconception_identification` tags.

## Output files

```
outputs/{subset}/responses.{model}.{condition}.jsonl      id, model, condition, prompt, response, generation
outputs/{subset}/judgments.{model}.{condition}.{judge}.jsonl
                                                            id, model, condition, judge,
                                                            solution_presence {answer_units, tagged[{text, explanation, label}]},
                                                            misconception_identification {tagged[...]},
                                                            emphasis {explanation_alignment_with_approach_1, ..._2, ..._balanced, final_response_alignment, justification},
                                                            errors
outputs/{subset}/scores.{model}.{condition}.{judge}.jsonl  id, model, condition, and the six metrics
outputs/{subset}/mcq_choices.{model}.jsonl                 id, model, comparison, order, chosen, reasoning
outputs/summary.{md,csv,tex}
```

`{model}` in file names is the litellm id with `/` replaced by `__`.

