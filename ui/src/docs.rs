//! Documentation content for the Docs tab.
//!
//! Sections are plain Markdown-ish text; the renderer is intentionally
//! minimal — egui has no markdown widget by default and the content is small
//! enough that a hand-rolled renderer is simpler than pulling in a crate.

pub struct DocSection {
    pub id: &'static str,
    pub title: &'static str,
    pub body: &'static str,
}

pub fn sections() -> &'static [DocSection] {
    SECTIONS
}

const SECTIONS: &[DocSection] = &[
    DocSection {
        id: "overview",
        title: "Overview",
        body: "\
NeuralCabin is a single-binary neural-network workbench, written from scratch \
in pure Rust. There are no external numerical or machine-learning crates: the \
tensor, autograd tape, layers, optimisers and dataset utilities all live in \
this repository.

The workbench is organised into seven tabs:

- Docs — this guide.
- Networks — list, create and select networks.
- Corpus — pick or build the training data for a network.
- Vocab — manage the vocabulary for next-token-generation networks.
- Training — train the active network.
- Inference — run the active network and inspect its outputs.
- Plugins — install third-party network types.

The selector at the top of the window is the same idea as LM Studio's model \
picker: switching networks loads the new network's weights into memory and \
makes every tab reflect the new selection.",
    },
    DocSection {
        id: "engine",
        title: "How the engine works",
        body: "\
Every tensor is a contiguous Vec<f32> with an associated shape. Forward and \
backward passes are recorded by the autograd tape: each op (matmul, add, ReLU, \
sigmoid, tanh, softmax+cross-entropy, MSE) appends a node, and a single \
reverse traversal computes gradients for every tracked leaf.

Optimisers (SGD, Adam) consume the gradient set and step the parameter \
tensors in place. Training runs on a background thread so the UI can sample \
its state every frame without blocking input. The trainer can be paused, \
resumed, learning-rate-tweaked, or stopped from the Training tab.",
    },
    DocSection {
        id: "networks",
        title: "Networks",
        body: "\
A network is a named instance of (kind, layer-stack, hyperparameters, weights). \
Use the Networks tab to:

- Click 'Create Network' to spawn a new one. The dialog asks which kind: \
Simplex (a regular MLP), Next-Token Generation (text), or any plugin-provided \
kind.
- Select a network from the list on the left to make it active. Its weights \
are immediately loaded into memory and the top-bar dropdown reflects the \
selection.
- Configure the architecture (input dimension, layers, init seed) on the right.

The active network is shared across the Corpus, Vocab, Training and Inference \
tabs.",
    },
    DocSection {
        id: "kinds",
        title: "Network kinds",
        body: "\
Simplex networks are dense MLPs: a stack of Linear and Activation layers \
trained with MSE or Cross-Entropy. They are the right tool for XOR, sine \
regression, spirals and small CSV problems.

Next-Token-Generation networks predict the next token in a sequence. The \
network's input is a flattened one-hot encoding of the last `context_size` \
tokens; the output is a logit vector over the vocabulary, trained with \
Cross-Entropy. The Vocab tab is enabled and required for this kind.

Plugin networks come from third-party plugins. The plugin's manifest decides \
whether its network type takes over the Vocab and Inference tabs; networks \
without a manifest opt-in fall back to the Simplex setup.",
    },
    DocSection {
        id: "corpus",
        title: "Corpus",
        body: "\
The Corpus tab defines the data a network is trained on.

For numeric networks (Simplex / fall-back plugins), pick a template:

- XOR — the canonical 4-row binary problem.
- Sine — noisy 1-D regression.
- Spirals — interleaved 2-D classification arms.
- CSV — load a comma-separated file from disk; last column is the label.
- Hand-rolled — type rows directly into the editor.

For Next-Token-Generation networks, the template is automatically 'Text'. \
You can paste a body of text and/or upload one or more text files (chats, \
docs, source code…); they all get appended to a single corpus. The 'Build \
training set' button slices the encoded token stream into context windows.",
    },
    DocSection {
        id: "vocab",
        title: "Vocabulary",
        body: "\
The Vocab tab is greyed out unless the active network is a Next-Token \
Generation network (or a plugin network whose manifest opts into vocab \
management).

Available actions:

- Wipe — reset to just the <unk> token.
- Add — paste tokens manually, one per click.
- Upload — load a newline-separated token file.
- Auto-generate — derive a vocab from the corpus text using one of: \
per-character, per-subword, per-word, per-sentence, or 'bulk' (all of the \
above, deduplicated).

Token id 0 is always <unk>; encoding falls back to <unk> when a token cannot \
be matched.",
    },
    DocSection {
        id: "training",
        title: "Training",
        body: "\
The Training tab runs the active network against its corpus. Choose the \
loss function, the optimiser (SGD or Adam), the learning rate and \
optimiser-specific hyperparameters, the number of epochs, the batch size, \
and the validation fraction.

A live loss plot updates after every epoch. The training loop can be paused, \
resumed, stopped, or live-tuned (the learning rate slider takes effect \
immediately).",
    },
    DocSection {
        id: "inference",
        title: "Inference",
        body: "\
The Inference tab is shaped by the active network's kind:

- Simplex — one numeric input box per dimension. There is a 'Real-time' \
toggle: when enabled the network is re-evaluated as soon as you change an \
input value, with no Predict button needed.
- Next-Token Generation — a text-input box plus temperature and \
max-tokens sliders. The model autoregressively generates tokens after the \
prompt.
- Plugin — if the plugin manifest sets `manages_inference = true`, the tab \
displays plugin-controlled fields. Otherwise it falls back to the Simplex \
real-time setup.",
    },
    DocSection {
        id: "plugins",
        title: "Plugins",
        body: "\
Plugins extend NeuralCabin with new network types.

Installation
------------

The Plugins tab accepts three sources:

- A folder containing a `manifest.json`.
- A bare `.json` manifest file.
- A `.zip` archive whose `manifest.json` entry is stored uncompressed (the \
other archive entries may be compressed).

Manifest format
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

`network_types` is the list of kinds your plugin contributes — they show up \
in the Create Network dialog. If `manages_vocab` is true, the Vocab tab is \
enabled and labelled as plugin-managed; if `manages_inference` is true, the \
Inference tab is plugin-managed; otherwise plugin networks behave like \
Simplex.

Configuration
-------------

Each installed plugin has a free-form JSON settings blob you can edit from \
the Plugins tab. The blob is passed through to plugin-managed UIs so they \
can persist user preferences without baking them into the manifest.

Removal
-------

Select a plugin in the list and click 'Delete'. NeuralCabin only removes \
the registry entry; the on-disk source archive is left untouched.",
    },
    DocSection {
        id: "persistence",
        title: "Saving & loading",
        body: "\
Every network has a 'Save' / 'Load' pair on the Networks tab. The on-disk \
format is a small JSON envelope (`format_version`, model, loss, optimiser, \
notes). The format is intentionally human-readable so you can diff weight \
checkpoints or hand-edit metadata.",
    },
    DocSection {
        id: "shortcuts",
        title: "Tips & conventions",
        body: "\
- For tiny binary tasks like XOR, a stack of Tanh + Sigmoid + MSE works well.
- For classification (Spirals, MNIST-ish CSVs), end with Linear → Identity \
and use Cross-Entropy. Softmax is applied internally for numerical stability \
and reported on the Inference tab as probabilities.
- For Next-Token Generation, start small: char-level vocab, context_size 1, \
a single hidden layer of 32–64 units. Even bigram-ish setups will overfit a \
short corpus quickly.
- The whole UI is monochrome by design. Status is communicated by typography \
and motion (the pulsing dot in the top bar lights up when training is live).",
    },
];
