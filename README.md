# PlayerBERT: Player Similarity via Event Sequences

This project builds a player similarity model that compares **how players play** (process) rather than only what they produce (box scores). It treats each player’s event history as a sequence and learns a **PlayerBERT** model over those event embeddings.

## Model Architecture

### EventEncoder (token‑level / “word”)
**Goal:** map a single event + its 360 context into a fixed‑dimensional vector `E_i ∈ R^128`.

**Inputs**
- **Event attributes (tabular):** flattened dot‑keys with categorical values and bucketized numerics (e.g., `type.name`, `pass.length_bucket`, `shot.xg_bucket`, `location_bucket.label`).
- **360 freeze‑frame:** variable‑length list of visible players with locations and teammate/keeper flags.

**Event attribute encoding (EventTransformer)**
- The dataset is **flattened** (dot‑keys). The training notebook derives `EVENT_FEATURES` directly from the flattened JSONL, excluding IDs and lists.
- Each event feature is a **token**: `(feature_name, feature_value)`.
- **Per‑feature vocabularies** are built from the dataset (each feature has its own lookup table).
- **Value embedding:** `Embedding(|V_f|, d)` for each feature `f`.
- **Feature embedding:** learned embedding for feature identity.
- Token representation: `token_f = value_embed_f + feature_embed_f`.
- All feature tokens go through a **TransformerEncoder** (2 layers, 4 heads by default).
- The event representation is the **mean** of token outputs: `z_event ∈ R^128`.

**360 frame encoding (SetEncoder)**
- Each visible player is converted into a per‑player vector:
  - `dx`, `dy`: relative to event actor
  - `dist`, `angle`
  - `is_teammate`, `is_keeper`
- A shared **MLP** maps each player vector to an embedding.
- **Mean pooling** over players yields `z_frame`.

**Gated fusion**
- Combine `z_event` and `z_frame` with a learned gate:
  - `g = sigmoid(W [z_event ; z_frame])`
  - `E_i = g ⊙ z_event + (1 − g) ⊙ z_frame`

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
   - Notebook: `train_event_encoder.ipynb` (Colab)
   - Masked Attribute Modeling on event features.
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

