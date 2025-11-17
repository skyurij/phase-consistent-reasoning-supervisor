# Phase-Consistent Reasoning Supervisor

![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange.svg)

### Detecting semantic drift, abstraction jumps, and phase transitions in dialogue with LLMs
This repository contains a working prototype of a system that analyzes reasoning coherence in human–LLM interaction. It detects when the model silently drifts away from the user’s intended meaning, even while staying superficially coherent.

The system is based on several months of experiments using OpenAI tools (ChatGPT, Assistants API, embeddings). Experimental results match the theoretical phase-dynamics model described in docs/theoretical_basis.md.

✨ Key Features

✔ Semantic Divergence Detection ✔ Abstraction-Level Shift Detection ✔ Meaning Substitution Detection ✔ Episode Segmentation (Attractor → Spiral → Attractor) ✔ Phase-Tension Metric T ✔ Episodic Memory of Meaning States ✔ Prediction of Coherence Failures ✔ Optional Joystick-of-Thought interface for visualization

📁 Repository Structure phase-consistent-reasoning-supervisor/ │ ├── analyzer.py ├── predictor.py ├── episode_memory.py │ ├── examples/ │ ├── dialog.txt │ ├── report.json │ └── docs/ └── theoretical_basis.md 🚀 Quick Start

Install dependencies:

pip install numpy scipy sklearn

Run analysis:

from analyzer import PhaseAnalyzer

analyzer = PhaseAnalyzer() dialog = open("examples/dialog.txt").read().split("\n")

report = analyzer.analyze(dialog) print(report)

🧠 Theory (short summary)

A human–LLM dialogue behaves like a dynamical system.

Stable meaning states correspond to attractors.

Transitions between attractors behave like spirals in phase-space.

Semantic drift and abstraction jumps are measurable as phase mismatches.

A combined metric:

[ T = w_1 D ;+; w_2 J ;+; w_3 E ]

where:

D = semantic divergence
J = abstraction-level jump
E = meaning substitution error
predicts upcoming coherence breakdowns.

Full theory → docs/theoretical_basis.md.

📬 Contact

This research is ongoing. For discussion or collaboration:

Yuri Skomorovsky Israel skyurij@gmail.com

📎 Attached Proposal

The full proposal summarized in Phase_Consistent_Reasoning_Proposal.pdf accompanies this repository.
