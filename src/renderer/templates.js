// NeuralCity templates. Exposed on window.NC_TEMPLATES.

window.NC_TEMPLATES = [
  {
    id: 'xor',
    name: 'XOR Classifier',
    kind: 'classifier',
    desc: 'Tiny 2-in, 2-class network that learns XOR. Proves the engine works end-to-end in seconds.',
    arch: {
      kind: 'classifier',
      inputDim: 2,
      outputDim: 2,
      hidden: [8, 8],
      activation: 'relu',
      dropout: 0,
      classes: ['false', 'true']
    },
    training: { optimizer: 'adam', learningRate: 0.05, batchSize: 4, epochs: 200, seed: 42 },
    trainingData: {
      samples: [
        { input: [0, 0], label: 0 },
        { input: [0, 1], label: 1 },
        { input: [1, 0], label: 1 },
        { input: [1, 1], label: 0 }
      ]
    }
  },
  {
    id: 'spiral',
    name: '2D Spiral Classifier',
    kind: 'classifier',
    desc: 'Classic two-spiral benchmark. Demonstrates a nonlinear decision boundary.',
    arch: {
      kind: 'classifier',
      inputDim: 2, outputDim: 2,
      hidden: [16, 16], activation: 'tanh', dropout: 0,
      classes: ['inner', 'outer']
    },
    training: { optimizer: 'adam', learningRate: 0.01, batchSize: 32, epochs: 120, seed: 7 },
    trainingData: { samples: generateSpiral(200) }
  },
  {
    id: 'regressor',
    name: 'Sine Regressor',
    kind: 'regressor',
    desc: 'Learns y = sin(x) from samples. A simple regression showcase.',
    arch: {
      kind: 'regressor',
      inputDim: 1, outputDim: 1,
      hidden: [32, 32], activation: 'tanh', dropout: 0
    },
    training: { optimizer: 'adam', learningRate: 0.005, batchSize: 16, epochs: 200, seed: 42 },
    trainingData: { samples: generateSineSamples(200) }
  },
  {
    id: 'slm',
    name: 'Tiny Character LM',
    kind: 'charLM',
    desc: 'A small language model that learns character sequences. Use the training corpus to teach it any text.',
    arch: {
      kind: 'charLM',
      vocabSize: 0,
      embDim: 24,
      contextLen: 16,
      hidden: [64, 64],
      activation: 'gelu',
      dropout: 0.1
    },
    training: { optimizer: 'adam', learningRate: 0.003, batchSize: 32, epochs: 30, seed: 42, workers: 0 },
    trainingData: { text: DEFAULT_CORPUS() }
  },
  {
    id: 'code_predictor',
    name: 'Code Predictor (char)',
    kind: 'charLM',
    desc: 'Small LM primed on JavaScript-like text. Starting point for a toy coding agent.',
    arch: { kind: 'charLM', vocabSize: 0, embDim: 32, contextLen: 24, hidden: [96, 96], activation: 'gelu', dropout: 0.1 },
    training: { optimizer: 'adam', learningRate: 0.002, batchSize: 32, epochs: 40, seed: 11, workers: 0 },
    trainingData: { text: DEFAULT_CODE_CORPUS() }
  },
  {
    id: 'chat',
    name: 'Chat Assistant (pairs)',
    kind: 'charLM',
    desc: 'Train on JSON pairs like {"user":"…","assistant":"…"}. The app auto-wraps each turn with role tags so the model learns to respond.',
    arch: { kind: 'charLM', vocabSize: 0, embDim: 32, contextLen: 128, hidden: [96, 96], activation: 'gelu', dropout: 0.1, isChat: true },
    training: { optimizer: 'adam', learningRate: 0.002, batchSize: 32, epochs: 60, seed: 7, workers: 4 },
    trainingData: { samples: DEFAULT_CHAT_PAIRS() }
  },
  {
    id: 'coder',
    name: 'Coding Assistant (pairs)',
    kind: 'charLM',
    desc: 'An language model meant to be an agent designated for coding. Settings made to already be hard to train but strong in the long run.',
    arch: { kind: 'charLM', vocabSize: 0, embDim: 32, contextLen: 128, hidden: [128, 96, 64], activation: 'gelu', dropout: 0.1, isChat: true },
    training: { optimizer: 'adam', learningRate: 0.01, batchSize: 256, epochs: 30, seed: 22, workers: 4 },
    trainingData: { samples: DEFAULT_CODING_PAIRS() }
  }
];

function generateSpiral(n) {
  const samples = [];
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < n / 2; i++) {
      const r = i / (n / 2) * 5;
      const t = 1.75 * i / (n / 2) * 2 * Math.PI + c * Math.PI;
      const x = r * Math.sin(t) + (Math.random() - 0.5) * 0.2;
      const y = r * Math.cos(t) + (Math.random() - 0.5) * 0.2;
      samples.push({ input: [x, y], label: c });
    }
  }
  return samples;
}

function generateSineSamples(n) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const x = (Math.random() * 2 - 1) * Math.PI;
    out.push({ input: [x], output: [Math.sin(x)] });
  }
  return out;
}

function DEFAULT_CORPUS() {
  return [
    "the quick brown fox jumps over the lazy dog.",
    "neural networks learn patterns from data.",
    "a small model can still be useful.",
    "training takes time but inference is fast.",
    "build from the ground up to understand every layer.",
    "each weight carries a fragment of the pattern.",
    "data shapes the mind of the machine.",
    "the future is written one token at a time.",
    "learn the math, own the magic.",
    "neuralcity builds models that you can actually hold in your head."
  ].join("\n").repeat(8);
}

function DEFAULT_CHAT_PAIRS() {
  // Tiny seed corpus. Replace with your own pairs in the editor.
  const seed = [
    { user: "Hello, I need help", assistant: "Of course, I'm here, what's up?" },
    { user: "Hi there", assistant: "Hey! How can I help today?" },
    { user: "What's your name?", assistant: "I'm a small NeuralCity model." },
    { user: "Thanks", assistant: "Happy to help!" },
    { user: "Can you help me?", assistant: "Yes — tell me what you need." },
    { user: "Tell me a fact", assistant: "Neural networks learn by adjusting weights." },
    { user: "Goodbye", assistant: "Take care!" },
    { user: "Are you smart?", assistant: "I'm small, but I try my best." },
    { user: "What can you do?", assistant: "I can chat about anything you teach me." },
    { user: "How do you work?", assistant: "I predict the next character using a tiny neural network." }
  ];
  // Repeat to give the network enough signal at this tiny scale.
  const out = [];
  for (let i = 0; i < 12; i++) for (const s of seed) out.push(s);
  return out;
}

function DEFAULT_CODING_PAIRS() {
  const seed = [
    {
      user: "Write a python script that prints hello world",
      assistant: "print(\"Hello World\")"
    },
    {
      user: "Make a python function that adds two numbers",
      assistant: "def add(a, b):\\n    return a + b"
    },
    {
      user: "Create a python loop that prints numbers 1 to 10",
      assistant: "for i in range(1, 11):\\n    print(i)"
    },
    {
      user: "Write a python script to read a file",
      assistant: "with open(\"file.txt\", \"r\") as f:\\n    data = f.read()\\n    print(data)"
    },
    {
      user: "Make a python class for a car",
      assistant: "class Car:\\n    def __init__(self, brand):\\n        self.brand = brand\\n\\n    def drive(self):\\n        print(self.brand + \" is driving\")"
    },
    {
      user: "Write python code to check if a number is even",
      assistant: "def is_even(n):\\n    return n % 2 == 0"
    },
    {
      user: "Create a python list and loop through it",
      assistant: "items = [1, 2, 3, 4, 5]\\nfor item in items:\\n    print(item)"
    },
    {
      user: "Make a simple calculator in python",
      assistant: "def calc(a, b, op):\\n    if op == '+':\\n        return a + b\\n    elif op == '-':\\n        return a - b\\n    elif op == '*':\\n        return a * b\\n    elif op == '/':\\n        return a / b"
    },
    {
      user: "Write python code to generate random number",
      assistant: "import random\\nprint(random.randint(1, 100))"
    },
    {
      user: "Create a python dictionary example",
      assistant: "person = {\\n    \"name\": \"John\",\\n    \"age\": 25,\\n    \"city\": \"NY\"\\n}\\nprint(person)"
    },
    {
      user: "Make python code to reverse a string",
      assistant: "def reverse_string(s):\\n    return s[::-1]"
    },
    {
      user: "Write python code for fibonacci sequence",
      assistant: "def fib(n):\\n    a, b = 0, 1\\n    for _ in range(n):\\n        print(a)\\n        a, b = b, a + b"
    }
  ];

  // Expand dataset for stronger training signal
  const out = [];
  for (let i = 0; i < 20; i++) {
    for (const s of seed) {
      out.push(s);
    }
  }
  return out;
}

function DEFAULT_CODE_CORPUS() {
  return [
    "fn add(a, b) do return a + b end",
    "fn mul(a, b) do return a * b end",
    "let x = 1",
    "let y = 2",
    "print add(x, y)",
    "if x > 0 do print x end",
    "for i = 0 to 10 do print i end",
    "while x < 100 do set x = x + 1 end",
    "fn square(n) do return n * n end",
    "print square(5)",
    "let list = [1, 2, 3, 4, 5]",
    "for i = 0 to 4 do print list[i] end"
  ].join("\n").repeat(10);
}
