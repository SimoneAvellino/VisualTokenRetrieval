> **NOTE:** This is the template for your report. Delete this note block before submission.

# Query-Conditioned Reconstruction of Visual Tokens for Moment Retrieval

- **Group ID**: [E.g., G07]
- **Project ID**: [E.g., 1]

---

## 1. Introduction and Objective

_Describe the objective of the project, why it is relevant, and what specific problem you are trying to solve. What is your main goal or initial hypothesis?_

The goal is a model that ingests an online stream of video frames, compresses their content into a compact memory, and — given a text query at any later time — recovers information about the moment the query refers to. Concretely, frames are first turned into visual tokens by a frozen vision encoder; the model must then compress the (variable-length) token history and, conditioned on a question, reconstruct the visual tokens of the ground-truth moment.

The central hypothesis is a retrieval one: **can the query alone pull the correct moment out of the compressed video memory?**

## 2. Contribution and Added Value

_Summarize what was done concisely. ("We built a model based on X for task Y"). Highlight your added value compared to merely running existing code (e.g., new loss, different architecture, specific data augmentation, etc.)._

We built two models that share the same data pipeline and evaluation suite:

- **Mean model** (`MeanReconstructor`) — an ablation model that predicts a single pooled vector, trained against the mean of the ground-truth moment tokens. It keeps the video as a full sequence and lets the **question drive** the prediction (the question's CLIP sentence embedding cross-attends every timestep). Its sole purpose is to answer the core question: _does the query retrieve the correct moment?_
- **Full reconstructor** (`QueryConditionedGTReconstructor`) — the actual architecture for the task. Visual tokens are merged with an **xLSTM** backbone (linear `O(T)` cost in sequence length), distilled into a small set of learned queries via a Perceiver-style resampler with **multi-scale temporal pooling**, then cross-attended by the encoded text query. A Transformer decoder with learnable output slots reconstructs the moment's token sequence.

Added value over running existing code: a custom **compound loss** (MSE + cosine + per-token norm + soft-DTW + temporal-hard-negative InfoNCE + length + span), a **same-video temporal-negative** sampling scheme and **GT-window data augmentation**.

## 3. Data Used

Describe formally the used data:

- Where do they come from?
- What are the main statistics (number of samples, train/test split)?
- What kind of preprocessing or augmentation did you apply to prepare them for training?\*

The data come from a set of annotations over the **Ego4D** dataset. Each annotation provides a natural-language `question`, the ground-truth moment span `(gt_start, gt_end)`, and the `query_time` at which the question is asked.

**Preprocessing.** Each video is cut into fixed-length, non-overlapping segments and encoded with a **frozen Qwen3-VL vision stack** (ViT + patch merger + projector). The output is a sequence of projected visual tokens already living in the language-model embedding space (`embed_dim = 4096`), pooled to 1 fps and cached to disk per segment. The text query is tokenized and encoded with a **frozen CLIP** text model.

**Split.** The train/val/test split is done at the **video level** (no annotation of the same video appears in two splits), seeded so that baselines and every model are compared on identical test data. Default ratios are 10% val / 10% test.

**Augmentation.** Targets are sampled with `GTWindowSampler`: at each epoch a window that always contains `[gt_start, gt_end]` is drawn with independent random left/right padding (up to 30 s each), and with probability 0.2 the exact GT window is used. This exposes the model to varied context around the same moment. Validation/test always use the exact GT window (no augmentation).

## 4. Methodology and Architecture

_Detail the experiments and how your system is built. What architecture did you use as a baseline? How did you modify it? Describe the network topology, key layers, the loss function used, and the training logic._

**Baselines (no parameters, never trained).** They put a floor under the learned models, evaluated on the same split and metrics:

- `mean_token` — predict the masked mean of the (cropped) context tokens, query-blind.
- `full_video_mean` — predict the mean of _every_ token of the whole video.
- `center_window` — return the centre window of the context, located by a fixed heuristic rather than by the query.

**Architecture.** The full reconstructor is a four-stage pipeline:

1. **Compression** — the visual-token sequence runs through an xLSTM block stack, giving a linear-cost temporal backbone. A Perceiver resampler then distils its hidden states into `num_queries` learned vectors; with multi-scale strides `[1, 4, 16]` the latents attend a multi-resolution view of the sequence, capturing both fast actions and global context.
2. **Query encoding** — the question is encoded by the frozen CLIP text model plus a trainable projection head.
3. **Conditioning** — the query cross-attends the compressed memory (a residual cross-attention block).
4. **Decoding** — learnable output slots (offset by a question summary) cross-attend the conditioned memory through a Transformer decoder to produce the predicted token sequence.

The mean model is a deliberately stripped-down counterpart: xLSTM over the full video, the question's pooled CLIP `[EOS]` embedding as the decoder query, cross-attention over all timesteps, and a single pooled output vector.

**Loss.** The reconstruction objective is a weighted sum of complementary terms, each optionally zero:

- **MSE** — anchors absolute magnitude.
- **cosine** (per-token) — anchors direction.
- **norm** (SmoothL1 on per-token L2 norm) — fixes scale collapse, where predictions get the direction right but land in a deflated region of the space.
- **soft-DTW** — a shift-tolerant alignment distance, the main reconstruction term once enabled, since point-wise MSE over-penalises correct-but-shifted sequences.
- **temporal InfoNCE** — contrastive term whose negatives are _same-video, time-shifted_ windows (hard negatives), forcing the model to delineate the action boundary rather than merely identify the right video. It works at `batch_size = 1` because negatives are intra-sample.

The mean model instead optimises the retrieval objective directly: in-batch InfoNCE on pooled vectors (each prediction vs. its own GT against every other sample's GT) plus a cosine anchor, optionally augmented with the same-video temporal negatives.

**Evaluation.** Beyond token-level reconstruction (`cos_token`, `cos_seq`, `norm_ratio`), the suite reports: **cross-sample retrieval** (recall@1/5, MRR) and a **query-usage ablation** that re-runs retrieval with shuffled questions: if the score barely drops, the model is ignoring the query.

## 5. Results and Discussion

Insert here the quantitative tables with the achieved results and compare your solution with the baseline. **Do not limit yourself to pasting numbers**, but comment on them:

- Why does model A perform better than model B?
- Are there classes where the model is particularly weak?
- Show qualitative examples (e.g., inserting correctly vs. incorrectly predicted images).\*

All retrieval metrics are cross-sample (rank each prediction's own GT moment against every other sample's GT). Higher is better; `chance` Recall@1 ≈ 0.023.

### 5.1 Mean model (retrieval ablation)

**Table 1**: Mean model vs. query-blind baselines (retrieval).

| Model                         | Recall@1 | Recall@5 |  MRR  |
| :---------------------------- | :------: | :------: | :---: |
| `full_video_mean` (baseline)  |  0.227   |  0.477   |   —   |
| `mean_token` (baseline)       |  0.205   |  0.455   |   —   |
| Mean model (InfoNCE + cosine) |  0.159   |  0.523   | 0.336 |

The trained mean model **does not beat the query-blind baselines on Recall@1** (0.159 vs. 0.205–0.227): predicting the global video mean already ranks the right moment surprisingly often, because the cross-video pool is dominated by _video identity_, not by the queried moment. It does improve Recall@5 (0.523), so the correct moment is usually in the top-5, but the top-1 decision is no better than ignoring the video content.

**Table 2**: Query-usage ablation — same model, each sample re-scored with another sample's (wrong) question.

| Query | Recall@1 | Recall@5 |  MRR   |
| :---- | :------: | :------: | :----: |
| Right |  0.1591  |  0.5227  | 0.3362 |
| Wrong |  0.1591  |  0.4773  | 0.3124 |

This is the key diagnostic. Swapping in the wrong question leaves **Recall@1 identical** and barely moves Recall@5 (−0.045) and MRR (−0.024). The model is therefore **largely ignoring the query** — it answers from the video context alone. This directly answers the project's central hypothesis: in the current setup the query does _not_ reliably retrieve the correct moment.

### 5.2 Full reconstructor (token reconstruction ladder)

**Table 3**: Reconstruction vs. retrieval across the loss ladder (`cos_seq` = sequence cosine, higher=better; Recall@1 = cross-sample retrieval).

| Model                      | cos_seq | Recall@1 |
| :------------------------- | :-----: | :------: |
| `mean_token` (baseline)    |  0.692  |  0.205   |
| `center_window` (baseline) |  0.493  |  0.205   |
| `mse_only`                 |  0.681  |  0.068   |
| `mse_cosine`               |  0.693  |  0.068   |
| `+ norm`                   |  0.679  |  0.080   |
| `+ infonce`                |  0.682  |  0.045   |
| `infonce_full`             |  0.661  |  0.023   |
| `small_full`               |  0.649  |  0.045   |

Two facts stand out:

- **Reconstruction looks fine but is uninformative.** Every variant reaches `cos_seq ≈ 0.65–0.69`, on par with the `mean_token` baseline (0.692). Cosine similarity _saturates_ in this token space — even predicting the average token scores ~0.69 — so it cannot tell whether the model discriminates the right moment.
- **Retrieval collapses.** Despite the good reconstruction numbers, **every trained variant scores far below the query-blind baselines** (0.045–0.080 vs. 0.205), and the most heavily-supervised one (`infonce_full`) lands at chance (0.023). This is **mode collapse**: the reconstruction objective drives predictions toward the global mean token, which minimises the loss while destroying the moment-level discrimination retrieval needs. Adding the contrastive/norm terms did not, on this data scale, pull the model out of the collapsed region.

### Summary

Across both model families the honest (retrieval) metric tells the same story the saturated reconstruction metric hides: the models do not yet use the query to localise the moment, and they fail to beat trivial query-blind baselines on Recall@1. The mean-model ablation isolates the cause (the query is ignored); the full-reconstructor ladder shows reconstruction loss alone collapses to the mean. Both are consistent with the very small usable dataset (see §6).

## 6. Conclusion and Limitations

_Summarize the project's outcome. What are the current limitations (e.g., requires too much memory, fails in low-light conditions)? If you had more time, what future experiments would you run?_

The project fell short of strong results mainly due to **memory constraints**: storing the visual tokens is expensive, so only a small fraction of the dataset could be used. The restricted quota also prevented saving model checkpoints regularly, which made it hard to produce the intermediate plots needed to explain the training dynamics and the results.

A recurring failure mode was **mode collapse**: reconstruction-only objectives push the model toward the global mean token, which scores well on cosine similarity (that metric saturates) but yields near-chance retrieval. The contrastive and span terms were added precisely to counter this, and retrieval — not reconstruction error — proved to be the honest metric. With more compute we would scale the training set, checkpoint on retrieval/localization from the start, and run a fuller sweep of the loss weights.

### 7.2 Use of Artificial Intelligence

_Declare here the possible use of tools like Copilot or ChatGPT, specifying in which phases they helped you (e.g., writing boilerplate, debugging, documentation), keeping in mind that the architectural design and the responsibility for the result are yours._

I used Claude Code to write the boilerplate code and for debugging. To verify the AI-written models, I did not re-read the code line by line; instead, each time I opened a fresh, clean session and used it to cross-check whether the models behaved as I expected. The architectural design and responsibility for the results are my own.
