//! Documentation content for the Docs tab.

pub struct DocSection {
    pub id: &'static str,
    pub title: &'static str,
    pub body: &'static str,
}

pub fn sections() -> &'static [DocSection] { SECTIONS }

const SECTIONS: &[DocSection] = &[
    DocSection {
        id: "overview",
        title: "Overview",
        body: "\
NeuralCabin is a single-binary neural-network workbench written entirely in \
pure Rust. There are no external numerical or ML crates: tensors, autograd, \
layers, optimisers, and dataset utilities all live in this repository.

The workbench is organised into seven tabs:

- Docs — this guide.
- Networks — list, create, and configure networks.
- Corpus — build or load the training data.
- Vocab — manage the token vocabulary for text networks.
- Training — train the active network with a live loss plot.
- Inference — run the network and inspect outputs.
- Plugins — install third-party network types.

The selector at the top of the window works like LM Studio's model picker: \
switching networks instantly loads the new network's state into every tab. \
All saved networks are loaded automatically at startup — no manual file picking \
required.",
    },
    DocSection {
        id: "engine",
        title: "How the engine works",
        body: "\
Every tensor is a contiguous Vec<f32> with an associated shape. Forward and \
backward passes are recorded on an autograd tape: each op (matmul, add, ReLU, \
sigmoid, tanh, softmax + cross-entropy, MSE) appends a node, then a single \
reverse traversal accumulates gradients for every tracked leaf.

Optimisers consume the gradient set and update parameters in place. Training \
runs on a background thread so the UI stays responsive; you can pause, resume, \
adjust the learning rate, or stop training from the Training tab.

Available optimisers:

- SGD — stochastic gradient descent with optional momentum.
- Adam — adaptive moment estimation; a good first choice for most tasks.
- AdamW — Adam with decoupled weight decay (Loshchilov & Hutter, 2017). \
Weight decay is applied directly to the parameter, not mixed into the gradient, \
which prevents the decay from being offset by the adaptive scaling. Use this \
whenever you want regularisation without sacrificing Adam's convergence speed.
- LAMB — Layer-wise Adaptive Moments for Batch training (You et al., 2019). \
Scales each layer's update by the ratio of the parameter norm to the update \
norm, enabling stable training with very large batches. Useful for large \
next-token-gen networks where batch sizes of 256+ are common.",
    },
    DocSection {
        id: "networks",
        title: "Networks",
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
corpus text, vocabulary, embedding settings, and hyperparameters.",
    },
    DocSection {
        id: "next-token-gen",
        title: "Next-Token Generation — full guide",
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
        body: "\
Simplex networks are dense MLPs: a stack of Linear and Activation layers \
trained with MSE or Cross-Entropy. They are the right tool for XOR, sine \
regression, spirals, and small CSV problems.

Next-Token-Generation networks predict the next token in a sequence. The \
Vocab and Corpus tabs are required; input and output dimensions are locked \
and driven automatically by vocab size and context length. See the \
'Next-Token Generation — full guide' section for a complete walkthrough.

Plugin networks come from third-party plugins installed via the Plugins tab. \
The plugin manifest decides whether the network type takes over the Vocab and \
Inference tabs; networks without a manifest opt-in fall back to the Simplex UI.",
    },
    DocSection {
        id: "corpus",
        title: "Corpus",
        body: "\
The Corpus tab defines the data a network trains on.

For Simplex / plugin networks, pick a template:

- XOR — the canonical 4-row binary problem.
- Sine — noisy 1-D regression.
- Spirals — interleaved 2-D multi-class scatter.
- CSV — load a comma-separated file; the last column is the label.
- Hand-rolled — type rows directly into the text editor.

For Next-Token-Generation networks the template is always 'Text'. Paste \
corpus text directly, or click 'Browse…' (or drag a file onto the window) \
to append text files. Multiple files accumulate into one corpus. \
'Re-tokenise' rebuilds the token stream from the current corpus + vocab. \
'Build training set' (on the Vocab tab) slices the stream into context windows \
and labels them for training.",
    },
    DocSection {
        id: "vocab",
        title: "Vocabulary",
        body: "\
The Vocab tab is available only for Next-Token Generation networks (or plugin \
networks whose manifest opts in).

Embedding type:
Select how tokens are converted to numeric vectors before the MLP. The options \
are One-Hot, TF-IDF, FastText / Word2Vec, and Transformer (positional). See \
the 'Next-Token Generation' guide for a detailed comparison.

Tokenisation:
- Wipe — reset to just the <unk> placeholder.
- Add — manually add a single token.
- Upload — load a newline-separated token list from a file (Browse… or drag).
- Auto-generate — derive a vocabulary from the corpus text.

Token id 0 is always <unk>; encoding falls back to <unk> for unknown tokens.",
    },
    DocSection {
        id: "training",
        title: "Training",
        body: "\
The Training tab runs the active network against its corpus.

Controls:
- Loss function: MSE (regression), Cross-Entropy (classification / NTG).
- Optimiser: SGD, Adam, AdamW, LAMB. See 'How the engine works' for details.
- Learning rate — can be changed live while training is running.
- Momentum (SGD) / β₁, β₂ (Adam family).
- Weight decay (AdamW / LAMB) — regularisation strength; 0.01 is a good default.
- Epochs, batch size, validation fraction.

A live loss and accuracy plot updates after each epoch. Training runs on a \
background thread; click Pause / Resume / Stop at any time.",
    },
    DocSection {
        id: "inference",
        title: "Inference",
        body: "\
The Inference tab is shaped by the active network's kind.

Simplex:
- One drag-value per input dimension. 'Real-time' re-runs the network \
every time an input changes — no button click needed.
- 'Clear Input' resets all inputs to zero; 'Clear Output' hides the last result.
- When the corpus has a 2-D input (XOR, Spirals) a scatter plot is shown with \
the current inference point highlighted as a crosshair. The crosshair colour \
shows the predicted class.

Next-Token Generation:
- Type a prompt in the text box, then click Generate.
- Temperature (0.05 – 4.0) controls randomness of sampling.
- Max tokens sets the generation length.
- 'Clear Input' wipes the prompt; 'Clear Output' clears the generated text.
- The current embedding type and vocab size are shown in the header.

Plugin:
- If `manages_inference = true`, the plugin controls this tab.
- Otherwise falls back to the Simplex numeric setup.",
    },
    DocSection {
        id: "plugins",
        title: "Plugins",
        body: "\
Plugins extend NeuralCabin with new network types.

Installation:
------------

Click Browse… (or drag a file / folder) in the Plugins tab. Three source \
types are accepted:

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

`network_types` lists the kinds your plugin contributes. If `manages_vocab` \
is true, the Vocab tab is enabled for plugin networks; if `manages_inference` \
is true the Inference tab is plugin-managed. Otherwise plugin networks behave \
like Simplex.

Configuration:
-------------

Each plugin has a free-form JSON settings blob you can edit in the Plugins \
tab. The blob is available to plugin-managed UIs so they can persist user \
preferences without baking them into the manifest.

Removal:
-------

Select a plugin in the list and click the ✕ button. NeuralCabin removes the \
registry entry; the on-disk source archive is left untouched.",
    },
    DocSection {
        id: "persistence",
        title: "Saving & loading",
        body: "\
Saving a network ('Save' button on the Networks tab) serialises the full \
network state to a JSON file in the platform appdata folder:

- Architecture (layer stack, input dim, seed).
- Trained weights.
- Loss function and optimiser settings.
- Corpus text body and uploaded file paths.
- Vocabulary tokens and tokenisation mode.
- Embedding type and embed_dim.
- Network kind (Simplex, NextTokenGen, Plugin).

All saved networks are reloaded automatically at startup. The on-disk format \
is intentionally human-readable so you can diff weight checkpoints or \
hand-edit metadata.

Format version 1 files from older builds are loaded cleanly; the new optional \
fields default to sensible values.",
    },
    DocSection {
        id: "shortcuts",
        title: "Tips & tricks",
        body: "\
General:
- For tiny binary tasks like XOR, Linear → Tanh → Linear → Sigmoid + MSE works well.
- For classification (Spirals, CSV), end with Linear → Identity and use \
Cross-Entropy. Softmax is applied internally; the Inference tab reports \
probabilities and highlights the argmax class.
- The Real-time toggle in Simplex inference re-runs the model on every input \
change and is especially useful with the 2-D scatter preview.

Next-Token Generation:
- Start with Per-character tokenisation, context_size = 1, one hidden layer of \
32–64 units. Even a bigram model will overfit a short corpus quickly so you can \
verify the pipeline end-to-end in seconds.
- For richer output, increase context_size to 4–8 and switch to Transformer \
embedding with embed_dim 32–64.
- Keep the loss plot open while training. If validation loss stops falling but \
train loss keeps dropping, you're overfitting — try a smaller network or add \
weight decay (AdamW/LAMB).
- Lower temperature (0.6–0.8) for more predictable, 'Wikipedia-style' output; \
raise it (1.0–1.5) for creative or exploratory generation.

Optimisers:
- Adam or AdamW are the best defaults for nearly everything.
- Use AdamW instead of Adam when you want implicit regularisation without \
manually tuning the architecture size.
- Try LAMB when using batch sizes above 128; it keeps training stable at \
scales where Adam diverges.
- SGD with momentum 0.9 is still the gold standard for convex problems and \
very shallow networks.

UI:
- The pulsing dot in the top bar shows training is active.
- The monochrome theme is intentional — all emphasis comes from typography and \
motion rather than colour.",
    },
];
