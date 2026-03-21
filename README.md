# PlayerBERT: Player Similarity via Event Sequences

This project builds a player similarity model that compares **how players play** (process) rather than only what they produce (box scores). It treats each player’s event history as a sequence and learns a **PlayerBERT** model over those event embeddings.

## Model Architecture

### EventEncoder V2 (token‑level / “word”)
**Goal:** map a single event + its 360 context into a fixed‑dimensional vector `E_i ∈ R^128` that captures both the action and the tactical scene around it.

**Inputs**
- **Event attributes (tabular):** flattened dot‑keys with categorical values and bucketized numerics (e.g., `type.name`, `pass.length_bucket`, `shot.xg_bucket`, `location_bucket.label`).
- **360 freeze‑frame:** variable‑length list of visible players with locations and teammate/keeper flags.

**Event attribute encoding**
- The dataset is **flattened** (dot‑keys). The training notebook derives `EVENT_FEATURES` directly from the flattened JSONL, excluding IDs and lists.
- Each event feature is a **token**: `(feature_name, feature_value)`.
- **Per‑feature vocabularies** are built from the dataset (each feature has its own lookup table).
- **Value embedding:** `Embedding(|V_f|, d)` for each feature `f`.
- **Feature embedding:** learned embedding for feature identity.
- Token representation: `token_f = value_embed_f + feature_embed_f`.
- A learned event query token and the event feature tokens go through a **TransformerEncoder**.
- The contextualized query token becomes the event summary that conditions scene retrieval.

**360 scene encoding**
- Each visible player is converted into a per‑player vector:
  - `dx`, `dy`: relative to event actor
  - `dist`, `angle`
  - `is_teammate`, `is_keeper`
- Player tokens are projected into a shared space and passed through stacked **relational self-attention** blocks.
- The relational attention uses **geometric pairwise bias** so players can attend to one another as a structured scene instead of being collapsed by averaging.

**Event-conditioned fusion**
- The event summary acts as a **query** over the encoded scene/player tokens.
- Cross-attention produces a scene readout tailored to the event semantics.
- The final event embedding is built from the event summary plus this conditional scene context.

**Output**
- `E_i ∈ R^128` for each event.

---

### PlayerBERT (sequence‑level / “sentence”)
**Goal:** model a player’s ordered sequence of events within a match and learn player‑style representations.

**Inputs**
- Ordered event embeddings `[E_1, …, E_n]` for a single player in a single match.
- Positional embeddings for event order.

**Architecture**
- Learned **positional embeddings** (max_len default 256).
- **TransformerEncoder** over event embeddings (2 layers, 4 heads by default).

**Training objective: Masked Event Modeling**
- Randomly mask a subset of events in the sequence.
- Replace masked positions with a learnable `[MASK]` vector.
- Predict the original event embeddings at masked positions.
- Loss: **MSE** on masked positions.

**Initialization / training strategy**
- **EventEncoder pretraining (Masked Attribute Modeling):**
  - 15% of feature tokens are masked.
  - 80% replaced with `[MASK]`, 10% replaced with random value, 10% kept (BERT‑style).
  - Cross‑entropy loss per feature head, averaged over features.
- **PlayerBERT training:**
  - EventEncoder outputs are used as the **embedding initializer** for sequence tokens.
  - Current notebook keeps EventEncoder **frozen** for simplicity; end‑to‑end fine‑tuning is the next step.

## Pipeline

1. **Preprocess & join events + 360**
   - Script: `preprocess_360_events.py`
2. **Data cleaning & flattening**
   - Notebook: `data_processing.ipynb`
   - Removes unique IDs, fills missing values, bucketizes numeric features,
     drops rare features, and **flattens** event attributes into dot‑keys.
3. **EventEncoder pretraining**
   - Baseline notebook: `train_event_encoder.ipynb` (Colab)
   - Current Python trainer: `train_event_encoder_v2.py`
   - Masked Attribute Modeling on event features, masked player-token reconstruction, and contrastive learning.
4. **PlayerBERT training**
   - Notebook: `train_playerbert.ipynb` (Colab)
   - Masked Event Modeling on event embeddings.
5. **Inference / similarity search**
   - Notebook: `infer_playerbert.ipynb` (Colab)
   - Builds and caches player embeddings; supports nearest‑neighbor search.

## Notes & Assumptions

- Event sequences are ordered **within each match** by `period`, `minute`, `second`, `timestamp`, `index`.
- PlayerBERT trains **per‑match player sequences** (no cross‑match leakage).
- The current PlayerBERT training uses **frozen** EventEncoder embeddings for simplicity.
  End‑to‑end fine‑tuning can be enabled later.

## Outputs

Saved weights (example paths used in Colab):
- EventEncoder: `models/event_encoder_mam.pt`
- PlayerBERT: `models/playerbert_mam.pt`
- Player embeddings: `models/player_embeddings.pt`

## Baseline vs Current V2

The older notebook-based baseline used:
- mean pooling over event tokens,
- mean pooling over freeze-frame players,
- a static gate to mix the two summaries.

The current V2 encoder in this repo replaces that with a relational, event-conditioned design:

- **File:** `event_encoder_v2.py`
- **Trainer scaffold:** `train_event_encoder_v2.py`

### What changed in V2

- **Event-conditioned scene encoding:** an `[EV]` query token fuses event and frame context.
- **Relational 360 modeling:** freeze-frame players are encoded with relational self-attention and geometric pairwise bias.
- **Cross-attention fusion:** the event summary queries scene tokens directly instead of using a static gate.
- **Multi-task SSL pretraining:** masked event-attribute modeling + masked player-token reconstruction + contrastive objective.
