//! Documentation content for the Docs tab.

pub struct DocSection {
    pub id: &'static str,
    pub title: &'static str,
    pub body: &'static str,
    pub last_updated: &'static str,
}

pub fn sections() -> &'static [DocSection] { SECTIONS }

pub fn export_all_as_markdown() -> String {
    let mut output = String::new();
    output.push_str("# NeuralCabin Documentation\n\n");
    output.push_str("Generated: 2026-05-10\n\n");
    output.push_str("---\n\n");

    for (idx, section) in SECTIONS.iter().enumerate() {
        output.push_str(&format!("## {}\n\n", section.title));
        output.push_str(&format!("**Last Updated:** {}\n\n", section.last_updated));
        output.push_str(&format!("{}\n\n", section.body));
        if idx < SECTIONS.len() - 1 {
            output.push_str("---\n\n");
        }
    }

    output
}

const SECTIONS: &[DocSection] = &[
    DocSection {
        id: "overview",
        title: "Overview",
        last_updated: "2026-05-10",
        body: "\
NeuralCabin is a single-binary neural-network workbench written entirely in \
pure Rust. There are no external numerical or ML crates: tensors, autograd, \
layers, optimisers, and dataset utilities all live in this repository.

The workbench is organised into seven tabs:

- Docs — this guide (copy, download, or export all docs).
- Networks — list, create, and configure networks.
- Corpus — build or load the training data.
- Vocab — manage the token vocabulary for text networks.
- Training — train the active network with a live loss plot.
- Inference — run the network and inspect outputs.
- Plugins — install third-party network types.

The selector at the top of the window works like LM Studio's model picker: \
switching networks instantly loads the new network's state into every tab. \
All saved networks are loaded automatically at startup — no manual file picking \
required.

-------

What can you build?

NeuralCabin is suitable for a wide range of machine learning tasks:

- Binary classification (XOR, simple logic gates)
- Multi-class classification (iris, handwritten digits, text categorisation)
- Regression (sine waves, polynomial fitting, trend prediction)
- Next-token text generation (language models, code completion)
- Sequence prediction (time series, music generation)
- Autoregressive generation (GPT-style text, story writing)
- Custom networks via plugins (experimental layer types, novel architectures)

Every network trains on CPU with full gradient computation and support for \
multiple optimisers. The UI is responsive even during training thanks to \
background threading.",
    },
    DocSection {
        id: "engine",
        title: "How the engine works",
        last_updated: "2026-05-10",
        body: "\
Every tensor is a contiguous Vec<f32> with an associated shape. Forward and \
backward passes are recorded on an autograd tape: each op (matmul, add, ReLU, \
sigmoid, tanh, softmax + cross-entropy, MSE) appends a node, then a single \
reverse traversal accumulates gradients for every tracked leaf.

Optimisers consume the gradient set and update parameters in place. Training \
runs on a background thread so the UI stays responsive; you can pause, resume, \
adjust the learning rate, or stop training from the Training tab.

-------

Tensor representation and operations:

Tensors are multi-dimensional arrays stored as contiguous floating-point vectors. \
Operations include:

- Reshape — change dimensions without copying data
- Transpose — swap axes; computed on-the-fly in forward pass
- MatMul — batched matrix multiplication with shape checking
- Elementwise ops (Add, Mul, Sub, Div) — broadcast-compatible
- Reductions — sum, mean, max along axes
- Indexing — slice and gather operations

All operations record themselves on the autograd tape for backpropagation.

-------

Automatic differentiation:

The autograd system uses reverse-mode differentiation (backprop). Each operation \
records its dependencies and gradient rules. During backward pass:

1. Start with loss = 1.0
2. Traverse the tape in reverse order
3. Each node applies its local gradient rule and accumulates gradients to inputs
4. Leaf gradients accumulate across the full computational graph

This enables efficient training of arbitrarily deep networks.

-------

Available optimisers:

- SGD — stochastic gradient descent with optional momentum (classical, stable).
- Adam — adaptive moment estimation; a good first choice for most tasks.
- AdamW — Adam with decoupled weight decay (Loshchilov & Hutter, 2017). \
Weight decay is applied directly to the parameter, not mixed into the gradient, \
which prevents the decay from being offset by the adaptive scaling. Use this \
whenever you want regularisation without sacrificing Adam's convergence speed.
- LAMB — Layer-wise Adaptive Moments for Batch training (You et al., 2019). \
Scales each layer's update by the ratio of the parameter norm to the update \
norm, enabling stable training with very large batches. Useful for large \
next-token-gen networks where batch sizes of 256+ are common.

-------

Loss functions:

- MSE (Mean Squared Error) — for regression; measures average squared difference.
- Cross-Entropy — for classification; measures divergence between predicted \
and true probability distributions. Softmax is applied automatically for \
multi-class problems.",
    },
    DocSection {
        id: "networks",
        title: "Networks",
        last_updated: "2026-05-10",
        body: "\
A network is a named bundle of (kind, layer-stack, hyperparameters, weights). \
Use the Networks tab to:

- Click 'Create Network' to spawn a new one. The dialog asks for a name and \
kind: Simplex (standard MLP), Next-Token Generation (text), or any plugin kind.
- Select a network from the left sidebar to make it active. Its weights are \
loaded into memory and the top-bar dropdown reflects the selection.
- Configure the layer architecture and init seed on the right panel.

Networks are saved and loaded automatically. Every saved network reappears \
in the sidebar on the next launch — no manual loading required. Hitting \
'Save' serialises the full network state: architecture, trained weights, \
corpus text, vocabulary, embedding settings, and hyperparameters.

-------

Layer types:

Linear — dense fully-connected layer with learnable weights and biases.
  Config: input dimension, output dimension.
  Parameters: weight matrix (out_dim × in_dim) + bias vector (out_dim).

Activation — non-linearity applied element-wise.
  Options: ReLU, Sigmoid, Tanh, Softmax (output layer), Identity (no-op).

-------

Network storage:

Networks are stored in your platform's appdata folder. Each network is a \
single JSON file containing:

- Architecture specification (layer stack, dims, init seed)
- Trained weights and biases (full precision)
- Loss and optimiser configuration
- Corpus and vocabulary (for text networks)
- Embedding settings

The JSON format is human-readable and version-aware. You can diff two \
checkpoints to track how weights evolved, or hand-edit metadata.",
    },
    DocSection {
        id: "next-token-gen",
        title: "Next-Token Generation — full guide",
        last_updated: "2026-05-10",
        body: "\
Next-Token Generation (NTG) networks predict the most probable next token in \
a sequence. This is the architecture behind GPT, text predictors, code \
completers, chatbots, and autoregressive music generators.

-------

How it works:

1. The corpus text is tokenised into a stream of integer IDs.
2. A sliding context window of the last N tokens is encoded into a fixed-size \
input vector.
3. The MLP maps that vector to a logit score for every token in the vocabulary.
4. Cross-entropy loss teaches the network to assign high probability to the \
correct next token.
5. At inference time the network is applied autoregressively: the generated \
token is appended to the context and the next step uses the updated window.

-------

Step-by-step setup:

1. Create a network — choose 'Next-Token Generation' in the dialog.
2. Go to Corpus — paste text directly or click 'Browse…' to upload files. \
   Aim for at least a few KB; more is better.
3. Go to Vocab — pick a tokenisation mode and click one of the 'Auto-generate' \
   buttons. Per-character is the safest starting point.
4. Still on Vocab — choose an embedding type (see below).
5. Click 'Build training set' in the Vocab tab.
6. Back on Networks — click 'Build / Reset Model'. The input and output dims \
   are set automatically and are locked (they are driven by vocab × context).
7. Go to Training — start training. Watch the loss fall.
8. Go to Inference — type a prompt and click Generate.

-------

Tokenisation modes:

- Per-character — one token per character. Smallest vocab, great for learning \
spelling and structure. Works on any language or code without configuration.
- Per-word — one token per whitespace-delimited word. Larger vocab but the \
network learns word-level patterns faster.
- Per-subword — fragments of 1-4 characters from each word. Balances vocab \
size against granularity; useful when the corpus mixes rare words and common \
roots.
- Per-sentence — one token per sentence. Tiny vocab, very coarse; useful for \
high-level outline prediction.
- Bulk (all, deduped) — combines char, word, and sentence tokens. Largest \
vocab; slowest to train but most expressive.
- Custom (manual) — add tokens one by one for specialised domains.

-------

Embedding types — how tokens become numbers:

One-Hot (default):
Each token position becomes a sparse vector of length vocab_size. Bit \
i = 1 if the token at that position is token i, else 0. Simple and transparent. \
Input dim = context_size × vocab_size.

TF-IDF:
Like One-Hot but each 1 is replaced by the token's inverse document frequency: \
  weight = ln(1 + total_tokens / count_of_token)
Rare tokens that carry more meaning get higher weights; common filler tokens \
are downweighted. No extra parameters; input dim = context_size × vocab_size.

FastText / Word2Vec style:
Each token maps to a dense vector of embed_dim floats (configurable; default \
32). The embedding table is seeded deterministically from the network seed, so \
the same table is always used for both training and inference. Input dim = \
context_size × embed_dim. Dense embeddings let the network generalise across \
similar tokens and often train faster than one-hot on large vocabularies.

Transformer (positional):
Same as FastText but adds a sinusoidal positional encoding to each position's \
embedding:
  PE(pos, 2i)   = sin(pos / 10000^(2i/d))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
This lets the network distinguish 'cat sat' from 'sat cat' even when the \
context window is large.

All embedding types are implemented in pure Rust — no external model downloads, \
no Python, no internet connection required.

-------

Architecture tips:

- Start small. A single hidden layer of 64–128 units with Tanh activation \
is enough to overfit a short corpus and verify your pipeline works.
- Set context_size = 1 for your first experiment (bigram model). Increase \
gradually as the network learns.
- Use AdamW or Adam. Learning rate 0.01–0.05 with batch size 32 works for \
most small corpora.
- Character-level vocab with context 4–8, one hidden layer of 128 units, \
and ~10 000 training samples is a solid 'tiny GPT' baseline.
- For word-level prediction, embed_dim 32–64 with Transformer embedding and \
context_size 4–8 gives noticeably more coherent output.

-------

What you can build:

GPT-style text generator:
  Corpus: novels, articles, Wikipedia dumps, your own writing.
  Vocab: Per-character or Per-word.
  Arch: Input → Linear(512) → Tanh → Linear(256) → Tanh → Linear(vocab) → Identity
  Embedding: Transformer, embed_dim 64.
  Context: 8–16.

Code autocompleter:
  Corpus: source code in your language (Python, Rust, JS…).
  Vocab: Per-character (code has a small character set).
  Arch: Input → Linear(256) → Tanh → Linear(128) → Tanh → Linear(vocab) → Identity
  Context: 4–8. Temperature 0.6 for less random output.

Chatbot response predictor:
  Corpus: conversation transcripts formatted as 'User: ... Bot: ...'.
  Vocab: Per-word.
  Arch: Input → Linear(128) → Tanh → Linear(vocab) → Identity
  Context: 4. The model learns simple response patterns.

Next-word predictor (like phone keyboard):
  Corpus: SMS logs, chat history, emails.
  Vocab: Per-word.
  Arch: Input → Linear(64) → ReLU → Linear(vocab) → Identity
  Context: 2–3. Fast to train, immediately useful.

Music note predictor:
  Corpus: MIDI note sequences encoded as text (e.g. 'C4 D4 E4 …').
  Vocab: Per-word (one token per note).
  Arch: small MLP, context 4–8.

-------

Temperature and sampling:

Temperature controls how 'random' the output is:
- < 0.5 — deterministic, repeats the most common patterns.
- 0.7–0.9 — creative but coherent (recommended starting point).
- > 1.2 — very random, generates surprising but often nonsensical output.

Set Max tokens to control generation length.",
    },
    DocSection {
        id: "kinds",
        title: "Network kinds",
        last_updated: "2026-05-10",
        body: "\
Simplex — dense MLPs (multi-layer perceptrons):
  A stack of Linear and Activation layers trained with MSE or Cross-Entropy. \
  They are the right tool for XOR, sine regression, spirals, small CSV \
  classification, and general supervised learning on tabular data.

  Input/output dimensions: You control both. Input matches your dataset \
  features; output matches the number of classes (or 1 for regression).

  Best for: Quick experiments, tabular data, binary logic, regression.

Next-Token-Generation (NTG):
  Predicts the next token in a sequence. The Vocab and Corpus tabs are \
  required; input and output dimensions are locked and driven automatically \
  by vocab size and context length.

  Input dimension = context_size × embedding_dimension.
  Output dimension = vocabulary_size.

  See the 'Next-Token Generation — full guide' section for a complete walkthrough.

  Best for: Language models, text generation, code completion, music, \
  any sequential / autoregressive task.

Plugin networks:
  Come from third-party plugins installed via the Plugins tab. The plugin \
  manifest decides whether the network type takes over the Vocab and Inference \
  tabs; networks without a manifest opt-in fall back to the Simplex UI.

  Best for: Experimental architectures, domain-specific layer types, \
  custom inference logic.",
    },
    DocSection {
        id: "corpus",
        title: "Corpus",
        last_updated: "2026-05-10",
        body: "\
The Corpus tab defines the data a network trains on.

For Simplex / plugin networks, pick a template:

- XOR — the canonical 4-row binary problem (2 inputs, 1 output, 4 training samples).
- Sine — noisy 1-D regression (x → sin(x) with noise; 200 samples).
- Spirals — interleaved 2-D multi-class scatter (two interlocking spirals; \
  100 samples × 3 classes).
- CSV — load a comma-separated file; the last column is the label. CSV loader \
  handles numeric columns directly; use one-hot encoding in the vocab for \
  categorical columns.
- Hand-rolled — type rows directly into the text editor. Each line is a \
  comma-separated row; last column is the label.

CSV format:

  feature1, feature2, ..., featureN, label
  1.5, 2.3, ..., 0.9, 0
  2.1, 3.4, ..., 1.2, 1

The last column is always treated as the label (class or regression target). \
Numeric columns are used as-is. If num_classes is set in Training, the label \
is one-hot encoded.

For Next-Token-Generation networks the template is always 'Text'. Paste \
corpus text directly, or click 'Browse…' (or drag a file onto the window) \
to append text files. Multiple files accumulate into one corpus. \
'Re-tokenise' rebuilds the token stream from the current corpus + vocab. \
'Build training set' (on the Vocab tab) slices the stream into context windows \
and labels them for training.

-------

Tips for dataset preparation:

- Normalisation: Simplex networks train better on inputs in [0, 1] or [-1, 1]. \
  Divide by max or subtract mean then divide by std.
- Balance: For classification, try to have roughly equal numbers of samples \
  per class. If imbalanced, consider weighting loss or oversampling.
- Size: 100–1000 samples is a good starting point. Smaller datasets overfit \
  quickly (which can be useful for debugging); larger datasets need bigger networks.
- Noise: Real data is noisy. Small synthetic corpora work well for testing the \
  pipeline but are very clean — add some random noise if needed.",
    },
    DocSection {
        id: "vocab",
        title: "Vocabulary",
        last_updated: "2026-05-10",
        body: "\
The Vocab tab is available only for Next-Token Generation networks (or plugin \
networks whose manifest opts in).

A vocabulary is a mapping of tokens (integers 0..) to meanings. Token 0 is \
reserved as <unk> (unknown); unknown tokens during encoding fall back to 0.

-------

Embedding type — how tokens become vectors:

Select how tokens are converted to numeric vectors before the MLP. The options:

One-Hot (default):
Each token position becomes a sparse vector of length vocab_size. Bit \
i = 1 if the token at that position is token i, else 0. Simple and transparent. \
Input dim = context_size × vocab_size. No learnable parameters.

TF-IDF:
Like One-Hot but each 1 is replaced by the token's inverse document frequency: \
  weight = ln(1 + total_tokens / count_of_token)
Rare tokens that carry more meaning get higher weights; common filler tokens \
are downweighted. No extra parameters; input dim = context_size × vocab_size.

FastText / Word2Vec style:
Each token maps to a dense vector of embed_dim floats (configurable; default \
32). The embedding table is seeded deterministically from the network seed, so \
the same table is always used for both training and inference. Input dim = \
context_size × embed_dim. Dense embeddings let the network generalise across \
similar tokens and often train faster than one-hot on large vocabularies.

Transformer (positional):
Same as FastText but adds a sinusoidal positional encoding to each position's \
embedding:
  PE(pos, 2i)   = sin(pos / 10000^(2i/d))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
This lets the network distinguish 'cat sat' from 'sat cat' even when the \
context window is large. Most expressive for sequence prediction.

-------

Tokenisation modes — building the vocab from corpus:

- Wipe — reset to just the <unk> placeholder.
- Add — manually add a single token.
- Upload — load a newline-separated token list from a file (Browse… or drag).
- Auto-generate — derive a vocabulary from the corpus text.

Auto-generate options:

- Per-character — one token per unique character. Smallest vocab, great for \
  learning spelling and structure. Works on any language or code without config.
- Per-word — one token per whitespace-delimited word. Larger vocab but the \
  network learns word-level patterns faster.
- Per-subword — fragments of 1–4 characters from each word. Balances vocab \
  size against granularity; useful when the corpus mixes rare words and common roots.
- Per-sentence — one token per sentence. Tiny vocab, very coarse; useful for \
  high-level outline prediction.
- Bulk (all, deduped) — combines char, word, and sentence tokens. Largest \
  vocab; slowest to train but most expressive.
- Custom (manual) — add tokens one by one for specialised domains.

-------

Workflow:

1. Create a Next-Token-Gen network.
2. Paste or upload corpus text (Corpus tab).
3. Come back to Vocab, click an Auto-generate button (Per-character is safest).
4. (Optional) Tweak the embedding type (One-Hot is fine; FastText is faster).
5. Click 'Build training set' to slice the corpus into context windows.
6. Go to Networks, click 'Build / Reset Model'. Dims are locked and auto-set.
7. Train away.",
    },
    DocSection {
        id: "training",
        title: "Training",
        last_updated: "2026-05-10",
        body: "\
The Training tab runs the active network against its corpus.

Main controls:

- Loss function: MSE (regression), Cross-Entropy (classification / NTG).
- Optimiser: SGD, Adam, AdamW, LAMB. See 'How the engine works' for details.
- Learning rate (α) — can be changed live while training is running. Typical \
  values: 0.01–0.1 for SGD, 0.001–0.01 for Adam.
- Momentum (SGD) — 0.9 is standard; accumulates velocity.
- β₁, β₂ (Adam family) — first and second moment decay. Defaults (0.9, 0.999) \
  work for most tasks.
- Weight decay (AdamW / LAMB) — regularisation strength; 0.01 is a good default \
  to prevent overfitting.
- Epochs — number of passes through the full training set.
- Batch size — samples processed before a gradient update. Larger batches = \
  fewer updates per epoch but more stable gradients.
- Validation fraction — hold out this fraction of data for validation (0–1). \
  If > 0, loss plot shows both train (blue) and validation (orange) losses.

-------

What to watch during training:

Loss curve:
- Should generally decrease over time.
- If loss increases, learning rate is too high; turn it down.
- If loss plateaus, try a larger network or longer training.
- If validation loss > train loss and diverging, you're overfitting.

Validation loss:
- If you set a validation fraction > 0, this shows on the plot.
- If validation loss stops falling while train loss keeps dropping, reduce \
  network size or increase weight decay.

Accuracy (for classification):
- Displayed alongside loss if applicable.
- Should rise as loss falls.

-------

Controls:

A live loss and accuracy plot updates after each epoch. Training runs on a \
background thread; click Pause / Resume / Stop at any time. You can adjust \
learning rate while training is paused or running (changes take effect on the \
next step).

The pulsing dot in the top bar indicates active training.",
    },
    DocSection {
        id: "inference",
        title: "Inference",
        last_updated: "2026-05-10",
        body: "\
The Inference tab is shaped by the active network's kind.

Simplex (general-purpose networks):

- One drag-value slider per input dimension. Adjust them to try different inputs.
- 'Real-time' toggle: when on, re-runs the network every time an input \
changes — no button click needed. Useful for exploring the model's behaviour.
- 'Predict' button: manually run inference once (only when Real-time is off).
- 'Clear Input' resets all inputs to zero; 'Clear Output' hides the last result.
- Output is shown as-is (numeric for regression) or as probabilities (for \
classification; argmax class is highlighted).
- When the corpus has a 2-D input (XOR, Spirals) a scatter plot is shown with \
the current inference point highlighted as a crosshair. The crosshair colour \
shows the predicted class (useful for visualising decision boundaries).

Next-Token Generation (text models):

- Type or paste a prompt in the text box.
- Click 'Generate' to autoregressively produce the next N tokens.
- Temperature (0.05 – 4.0) controls randomness of sampling:
  - 0.1–0.5: deterministic, repeats common patterns (good for code).
  - 0.7–0.9: creative but coherent (recommended starting point).
  - 1.0: neutral (uniform sampling from logits).
  - 1.2–2.0: very random, often nonsensical (for exploration).
- Max tokens sets the maximum generation length.
- 'Clear Input' wipes the prompt; 'Clear Output' clears the generated text.
- The current embedding type, vocab size, and context length are shown in \
the header for reference.
- Generated text is appended to the prompt, so you can see the full output \
and continue generating.

Plugin (custom network types):

- If the plugin's manifest has `manages_inference = true`, the plugin \
controls this entire tab with custom UI.
- Otherwise falls back to the Simplex numeric setup.",
    },
    DocSection {
        id: "plugins",
        title: "Plugins",
        last_updated: "2026-05-10",
        body: "\
Plugins extend NeuralCabin with new network types and custom UI for \
experimental architectures.

Installation:
-----------

Click Browse… (or drag a file / folder) in the Plugins tab. Three source types \
are accepted:

- A folder containing a manifest.json.
- A bare .json manifest file.
- A .zip archive whose manifest.json entry is stored uncompressed.

Manifest format:
---------------

```
{
  \"id\": \"my-org.my-plugin\",
  \"name\": \"My Plugin\",
  \"version\": \"0.1.0\",
  \"author\": \"You\",
  \"description\": \"What it does.\",
  \"network_types\": [\"diffusion\", \"transformer-tiny\"],
  \"manages_vocab\": false,
  \"manages_inference\": false
}
```

`network_types` — list of kinds (strings) your plugin contributes. When a \
network of one of these kinds is active, the plugin is loaded.

`manages_vocab` — if true, the Vocab tab is enabled and plugin-controlled.

`manages_inference` — if true, the Inference tab is plugin-managed. Otherwise \
plugin networks behave like Simplex (numeric drag-values).

Configuration:
--------------

Each plugin has a free-form JSON settings blob you can edit in the Plugins tab. \
The blob is persisted and available to plugin-managed UIs so they can read user \
preferences without baking them into the manifest.

The settings are not automatically used by the core engine — plugin UIs must \
read the config and apply it themselves.

Removal:
-------

Select a plugin in the list and click the ✕ button. NeuralCabin removes the \
registry entry and cleans up its settings. The on-disk source archive is left \
untouched.

-------

Plugin development:

Plugins are currently a specification and UI affordance. A full plugin system \
with WASM or dynamic loading is on the roadmap. For now, use plugins to \
document experimental network types and configurations (e.g. a diffusion \
architecture spec, a custom layer type, or research ideas).",
    },
    DocSection {
        id: "persistence",
        title: "Saving & loading",
        last_updated: "2026-05-10",
        body: "\
Auto-loading:

All saved networks are reloaded automatically at startup. NeuralCabin scans the \
platform appdata folder and loads every valid network file. No manual file \
picking required.

Manual saves:

Click the 'Save' button on the Networks tab to checkpoint the active network. \
This serialises the full state to a JSON file:

- Architecture (layer stack, input dim, seed).
- Trained weights and biases (full precision).
- Loss function and optimiser settings.
- Corpus text body and uploaded file paths.
- Vocabulary tokens, tokenisation mode, and embedding config.
- Embedding type, embed_dim, and context_size.
- Network kind (Simplex, NextTokenGen, Plugin).

The on-disk format is intentionally human-readable so you can:

- Diff weight checkpoints between training runs.
- Hand-edit metadata without retraining.
- Inspect gradients and parameter statistics.
- Version-control networks in git.

File locations:

- Linux: ~/.local/share/neuralcabin/
- macOS: ~/Library/Application Support/neuralcabin/
- Windows: %APPDATA%\\neuralcabin\\

Version compatibility:

Format version 1 files from older builds are loaded cleanly; newer optional \
fields default to sensible values. Networks are forward-compatible within a \
major version.",
    },
    DocSection {
        id: "shortcuts",
        title: "Tips & tricks",
        last_updated: "2026-05-10",
        body: "\
Quick wins — get your network training in seconds:

XOR task:
  Arch: Linear(2) → Tanh → Linear(8) → Tanh → Linear(1) → Sigmoid
  Loss: MSE
  Opt: Adam, lr=0.01
  Batch: 4 (all samples)
  Should converge to ~0.01 loss in 500 epochs.

Spirals task:
  Arch: Linear(2) → ReLU → Linear(64) → ReLU → Linear(3) → Identity
  Loss: Cross-Entropy
  Opt: Adam, lr=0.01
  Batch: 16
  Validation: 0.2
  Should reach >95% accuracy in 1000 epochs.

Sine regression:
  Arch: Linear(1) → ReLU → Linear(32) → ReLU → Linear(1) → Identity
  Loss: MSE
  Opt: Adam, lr=0.01
  Batch: 16
  Should fit quickly, showing smooth sine curve in Inference.

-------

Next-Token Generation tips:

Start minimal:
- Tokenisation: Per-character
- Context: 1 (bigram model)
- Embedding: One-Hot
- Arch: Linear(hidden) → Tanh → Linear(vocab) → Identity
- Hidden size: 32–64 for a 10–50 char corpus
- Even a tiny bigram model will overfit a short corpus in seconds, verifying \
  the pipeline end-to-end.

Scale up gradually:
- Increase context_size to 4–8 for more coherent output.
- Switch to Transformer embedding (embed_dim 32–64) for richer patterns.
- Add a second hidden layer if loss plateaus.
- Use AdamW with weight_decay=0.01 to prevent overfitting on small corpora.

Interpreting quality:
- If all outputs are repetitive, model is underfitting or learning spuriously.
- If output starts coherent then degrades, temperature is too high or \
  context is too short.
- If validation loss stops falling, increase weight decay or reduce network size.

Generation control:
- Temperature 0.6–0.8: Wikipedia-style, predictable, good summaries.
- Temperature 1.0: neutral, balanced randomness.
- Temperature 1.2–1.5: creative but often coherent on larger models.
- Temperature > 2.0: experimental, usually nonsensical.

-------

Optimiser selection:

Quick rule of thumb:

- For most tasks (especially if in doubt): Adam with lr=0.001–0.01.
- For large batches (>128): LAMB with lr=0.01–0.1 and weight_decay=0.01.
- For fine-tuning: AdamW with weight_decay=0.01–0.05.
- For convex problems or shallow nets: SGD with momentum=0.9, lr=0.01–0.1.

Momentum vs. adaptive:

- SGD momentum accumulates velocity; good on smooth landscapes, essential \
  for momentum-based methods.
- Adam adapts per-parameter learning rate; handles sparse gradients and \
  noisy data well.
- AdamW decouples weight decay from adaptive scaling; slightly better generalisation.

-------

Debugging training:

Loss is NaN:
  → Learning rate is too high. Halve it and retry. Also check for bad data.

Loss oscillates wildly:
  → Batch size too small or learning rate too aggressive. Increase batch or \
    reduce lr.

Loss plateaus early:
  → Network too small. Add more hidden units or layers. Try a different activation.

Validation loss >> train loss:
  → Overfitting. Reduce network size, increase weight decay (AdamW/LAMB), or \
    get more data.

Training is slow:
  → Normal on CPU. Batch size affects convergence speed; larger batches = \
    fewer gradient steps per epoch. Try training for more epochs with a larger \
    batch size if wall time matters.

-------

UI notes:

- The pulsing dot (◯) in the top bar indicates active training.
- You can pause/resume/stop training at any time. Pausing lets you inspect \
  weights or adjust hyperparameters.
- The monochrome colour scheme is intentional — all emphasis comes from \
  typography, layout, and motion rather than colour.
- Drag-values in inference are keyboard-friendly: click then use arrow keys.
- Copy, download, and export docs via the buttons on the Docs tab.",
    },
];
