<div align="center">
 <img src="./docs/figures/intellagent_logo.png" alt="IntellAgent Logo" width="600">
 
 <p><i>Uncover Your Agent's Blind Spots — Now with DeepSeek Support</i></p>

 [![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/plurai-ai/intellagent/blob/main/LICENSE)
 [![Upstream](https://img.shields.io/badge/Upstream-plurai--ai%2FintellAgent-blue)](https://github.com/plurai-ai/intellagent)

 [Original Documentation](https://intellagent-doc.plurai.ai/) |
 [Quick Start](#fire-quickstart) |
 [What's Different](#whats-different-in-this-fork) |
 [Paper](https://arxiv.org/pdf/2501.11067)
</div>

---

> **This is a maintained fork of [IntellAgent](https://github.com/plurai-ai/intellagent) by [Collectiwise](https://collectiwise.ai).**
> Fork maintainer: [johannes@collectiwise.ai](mailto:johannes@collectiwise.ai)

---

## Why This Fork Exists

At [Collectiwise](https://collectiwise.ai), we build AI data analytics agents that help organizations unlock value from their data warehouses. Our agents handle complex multi-turn conversations — querying databases, building visualizations, interpreting results — and we need to know they work reliably before every release.

IntellAgent, created by the team at [Plurai](https://plurai.ai), is the best open-source framework we've found for this. It generates thousands of realistic edge-case scenarios, simulates diverse user interactions, and provides the kind of granular, policy-level diagnostics that let us pinpoint exactly where our agents fall short. We run IntellAgent as part of our CI/CD pipeline so that every release is stress-tested before it reaches a single client.

There was just one problem: **cost**. IntellAgent was built around OpenAI's API, and building a policy graph for a complex agent can require tens of thousands of LLM calls. On GPT-4o, a single evaluation run was costing us $20–50. That's fine for a one-off benchmark, but unsustainable when you want to run it on every commit.

This fork adds **native DeepSeek support**, bringing the cost of a full evaluation run down to roughly **$1–3** — a 10–20x reduction. We share these results transparently with our clients through performance dashboards that track agent quality across releases.

## What's Different in This Fork

### Native DeepSeek LLM Provider

DeepSeek's API is OpenAI-compatible but has three key differences that this fork handles natively:

| Issue | OpenAI | DeepSeek | This Fork's Fix |
|---|---|---|---|
| **Structured output** | `json_schema` mode | Only `json_object` mode | `set_llm_chain()` auto-selects `json_mode` for DeepSeek |
| **Prompt requirements** | No constraint | Requires "json" in prompt text | Auto-injects JSON instruction into system messages |
| **Field naming** | Follows schema exactly | May return `likelihood_score` instead of `score` | `Rank` model accepts common field name variants |

### Additional Improvements

- **Zero-edge protection** in `extract_graph()` — prevents `ZeroDivisionError` when edge extraction has a high failure rate
- **Clean `type: 'deepseek'` config** — no monkey-patching, no environment variable hacks. Just set `type: 'deepseek'` in your config and go

### Cost Comparison

| Phase | GPT-4o | DeepSeek | Savings |
|---|---|---|---|
| Policy graph (one-time) | ~$18 | ~$0.50–1.00 | **95%** |
| Dataset generation (5 scenarios) | ~$2–5 | ~$0.10–0.25 | **95%** |
| Simulation + critique | ~$5–10 | ~$0.25–0.50 | **95%** |
| **Full run** | **$25–35** | **$1–3** | **90–95%** |

At 500 scenarios (production scale), the difference is even starker: **~$750–1,500** on GPT-4o vs. **$10–25** on DeepSeek.

---

## :fire: Quickstart

IntellAgent requires `python >= 3.9`

#### Step 1 — Download and install

```bash
git clone git@github.com:johannescastner/nicer_intellagent.git
cd nicer_intellagent
pip install -r requirements.txt
```

#### Step 2 — Set your API key

Edit `config/llm_env.yml`:

```yaml
# Option A: DeepSeek (recommended for cost)
deepseek:
  DEEPSEEK_API_KEY: "your-deepseek-key"

# Option B: OpenAI (still fully supported)
openai:
  OPENAI_API_KEY: "your-openai-key"
```

#### Step 3 — Configure your LLM provider

In your config file (e.g. `config/config_education.yml`), set the provider:

```yaml
# For DeepSeek (~95% cheaper):
llm_intellagent:
  type: 'deepseek'
  name: 'deepseek-chat'

# For OpenAI (original behavior):
llm_intellagent:
  type: 'openai'
  name: 'gpt-4o'

# Azure, Anthropic, Google, etc. all still work as before
```

#### Step 4 — Run

```bash
# Simple environment (no database)
python run.py --output_path results/education --config_path ./config/config_education.yml

# Complex environment (with database)
python run.py --output_path results/airline --config_path ./config/config_airline.yml
```

#### Step 5 — Visualize

```bash
streamlit run simulator/visualization/Simulator_Visualizer.py
```

> **Troubleshooting**
> - **Rate limit messages** → Decrease `num_workers` in your config file
> - **Frequent timeout errors** → Increase `timeout` values
> - **DeepSeek field errors** → Already handled by this fork's `Rank` model aliases

---

## How It Works

![simulator_recording](./docs/figures/overview.gif)

IntellAgent operates in three stages:

1. **Policy Graph Construction** — Decomposes your agent's system prompt into individual policies, scores their complexity, and builds a weighted co-occurrence graph
2. **Scenario Simulation** — Samples policy combinations via graph walks, generates realistic user-agent dialogues, and monitors policy compliance in real time
3. **Critique & Analysis** — Evaluates each conversation for policy violations, coherence, and task completion, producing granular diagnostics

For a deeper understanding, see the [system overview guide](https://intellagent-doc.plurai.ai/How_it_Works/how-it-works/).

---

## How Collectiwise Uses IntellAgent

We integrate IntellAgent into our Cloud Build CI/CD pipeline. On every build:

1. **Unit & integration tests** validate our agent's core modules
2. **IntellAgent generates scenarios** tailored to our agent's policies and tools
3. **Simulated conversations** stress-test the agent across complexity levels
4. **Results flow into BigQuery**, where they power Superset dashboards that track quality over time

For major releases, these dashboards are shared directly with our clients — giving them transparent, data-driven confidence that our agents are improving with every version.

---

## Acknowledgments

This fork would not exist without the excellent work of **Elad Levi** and **Ilan Kadar** at [Plurai](https://plurai.ai), who created IntellAgent and open-sourced it under the Apache 2.0 license. Their framework is genuinely innovative — the policy graph approach to scenario generation is elegant, and the correlation with τ-bench results (Pearson 0.98 on airline, 0.92 on retail) speaks for itself.

We encourage you to:
- Read the [original paper](https://arxiv.org/pdf/2501.11067)
- Star the [upstream repository](https://github.com/plurai-ai/intellagent)
- Join the [Plurai Discord community](https://discord.gg/YWbT87vAau)
- Subscribe to the [Plurai Newsletter](https://plurai.substack.com/)

## Citation

```bibtex
@misc{2501.11067,
  Author = {Elad Levi and Ilan Kadar},
  Title = {IntellAgent: A Multi-Agent Framework for Evaluating Conversational AI Systems},
  Year = {2025},
  Eprint = {arXiv:2501.11067},
}
```

## License

Apache 2.0 — same as upstream. See [LICENSE](./LICENSE).

## Contact

- **Fork maintainer:** Johannes Castner — [johannes@collectiwise.ai](mailto:johannes@collectiwise.ai)
- **Collectiwise:** [collectiwise.ai](https://collectiwise.ai)
- **Original authors:** [Plurai](https://plurai.ai/contact-us)
- **Issues:** [GitHub Issues](https://github.com/johannescastner/nicer_intellagent/issues)
