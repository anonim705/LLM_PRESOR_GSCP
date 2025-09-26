# 🌍 Seeing Sustainability Through the Eyes of AI: Large Language Model Biases Concerning Corporate Social Responsibility and Green Supply Chain

This repository contains **Python scripts** and **data** for comparing how different Large Language Models (LLMs) respond to:  

**Likert-scale surveys** on ethics, social responsibility, and environmental collaboration, in both **default settings (no cultural framing)** and **organizational culture personas** (Clan, Adhocracy, Market, Hierarchy)  


It supports both **API-based models** (OpenAI, Anthropic, DeepSeek) and **local Hugging Face models** (Mistral, LLaMA).  

---

## 📂 Repository Structure
```
├── data/
│   ├── PRESOR_gpt-4o_clan.xlsx
│   ├── GSCP_claude_hierarchy.xlsx
│   └── ...
├── scripts/
│   └── surveys_runner.py
├── requirements.txt
└── README.md
```

- `data/` → Collected results (Excel format)  
- `scripts/` → Survey runner scripts  
- `requirements.txt` → Python dependencies  
- `README.md` → Project documentation  

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/anon0101-llm/llm-sustainability-culture.git
cd llm-sustainability-culture
```

### 2. Install dependencies
We recommend Python **3.10+**.  
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
DEEPSEEK_API_KEY=your_deepseek_key
```

> Local models (Mistral, LLaMA) do not require API keys but need GPUs with sufficient memory.  

---

## Organizational Culture Personas

The framework supports two main modes of response:  

- **Default** → No cultural framing is given. LLMs answer *only* with Likert-scale responses.  
- **Culture personas** → Based on the Competing Values Framework. The LLM is instructed to adopt a perspective aligned with one of four organizational cultures:  
  - **Clan** → Cooperation, teamwork, trust, family-like commitment.  
  - **Adhocracy** → Innovation, risk-taking, autonomy, entrepreneurship.  
  - **Market** → Competition, achievement, productivity, results orientation.  
  - **Hierarchy** → Structure, rules, stability, efficiency, control.  

This design enables comparisons between **baseline (default)** responses and **culture-shaped** responses.  

---

## 📊 Running Surveys

### Available Surveys
- **PRESOR** – Perceived Role of Ethics and Social Responsibility  
- **GSCP** – Green Supply Chain Partnerships (suppliers & customers)  

### Example Commands
Run PRESOR with GPT-4o, Adhocracy persona, 2 runs:
```bash
python scripts/surveys_runner.py --model gpt-4o --culture adhocracy --survey PRESOR --runs 2
```

Run GSCP with local LLaMA-3.3-70B, Market persona:
```bash
python scripts/surveys_runner.py --model meta-llama/Llama-3.3-70B-Instruct --culture market --survey GSCP --runs 1
```

Run PRESOR with Mistral-Large, Default persona (no culture):
```bash
python scripts/surveys_runner.py --model mistralai/Mistral-Large-Instruct-2407 --culture default --survey PRESOR
```

Results are saved as Excel files, e.g.:
```
gpt_PRESOR_adhocracy_2runs.xlsx
llama_GSCP_market_1runs.xlsx
mistral_PRESOR_default_1runs.xlsx
```

---

## 📜 License
This project is licensed under the MIT License.  

