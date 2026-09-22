# 🌤️ Weather-Advisory Support Bot

A LangGraph-based chatbot that answers outdoor activity safety questions using live weather data, grounded in a set of Standard Operating Procedures (SOPs). Every answer either cites a specific SOP or explicitly states that none applies — no invented advice, ever.

---

## Architecture

The bot is built as a directed LangGraph graph. Each user turn flows through a fixed pipeline of nodes:

```
User message
     │
     ▼
┌─────────────────┐
│  extract_intent │  LLM call — pulls location, activity, vulnerable group
│                 │  from the message + conversation history
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  fetch_weather  │  HTTP call to Open-Meteo (no API key needed)
│                 │  Geocodes city → fetches current conditions
└────────┬────────┘
         │
    weather_error?
    ┌────┴────┐
   Yes       No
    │         │
    ▼         ▼
┌───────────┐  ┌──────────────┐
│ handle_   │  │  match_sop   │  Phase 1: deterministic numeric checks
│ weather_  │  │              │  Phase 2: LLM adjudicates fuzzy SOPs +
│ failure   │  │              │           activity relevance + severity ranking
└─────┬─────┘  └──────┬───────┘
      │                │
      ▼                ▼
     END       ┌──────────────┐
               │ compose_     │  LLM writes final response, grounded in
               │ answer       │  actual weather numbers + matched SOP advice
               └──────┬───────┘
                      │
                     END
```

### Node details

| Node | Type | What it does |
|------|------|-------------|
| `extract_intent` | LLM | Extracts location, activity, vulnerable group, detects follow-ups |
| `fetch_weather` | Deterministic | Geocodes city via Open-Meteo, fetches 9 current weather fields |
| `match_sop` | Hybrid | Numeric conditions checked first; LLM resolves fuzzy SOPs and picks highest-severity match |
| `compose_answer` | LLM | Writes a warm, grounded response citing exact weather numbers and the SOP |
| `handle_weather_failure` | LLM | Honest failure message; never invents conditions or gives safety advice |

### Multi-SOP conflict resolution

When multiple SOPs are numerically triggered simultaneously, the selection policy is:

1. **Highest severity wins** — `critical > high > moderate > low`
2. **Tie-break: most activity-specific** — a cycling-specific SOP beats a general outdoor SOP if the user is asking about cycling
3. The LLM documents all applicable SOPs in `all_applicable` for auditability

---

## SOP Design

SOPs are defined entirely in `sops/sops.yaml`. The Python code in `src/sops_loader.py` reads and interprets them — **no code changes are needed to add, modify, or retire a policy**.

**Why YAML?**
- Non-engineers can read and propose changes to SOPs without touching Python
- Diffs are human-readable in code review
- Easy to validate with a YAML linter or JSON Schema
- Supports both numeric-threshold SOPs and fuzzy (LLM-evaluated) SOPs in the same format

**SOP schema:**

```yaml
- id: "OE-001"               # Unique identifier
  category: "outdoor_exercise"  # outdoor_exercise | travel | vulnerable_groups | general_outdoor | water_activities
  title: "Human-readable name"
  severity: "high"           # low | moderate | high | critical
  trigger_description: "..."  # Plain-language description for documentation
  conditions:                # List of numeric conditions (all must be true)
    - field: "wind_speed_10m"
      op: "gte"              # gte | gt | lte | lt | eq
      value: 40
  advice: "..."              # Exact advice text injected into the LLM prompt
  fuzzy: false               # true = no numeric threshold; LLM evaluates holistically
```

**Adding a new SOP:** Add a new entry to `sops/sops.yaml` and restart the app. No code changes required.

---

## Setup

### Prerequisites
- Python 3.11+
- An OpenAI API key (uses `gpt-4o-mini`)

### Installation

```bash
# 1. Clone the repo
git clone <repo-url>
cd weather-advisory-bot

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
cp .env.example .env
# Edit .env and paste your OpenAI API key
```

### Running the app

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

**Example questions to try:**
- `Is it safe to cycle in Mumbai today?`
- `My kids want to play outside in Jaipur this afternoon`
- `Should I drive to Pune right now?`
- `Is today a good day for a picnic in Bangalore?`

---

## Running the eval suite

```bash
python eval/eval_suite.py
```

The suite runs 8 tests:

| # | Test | Type |
|---|------|------|
| 1 | Cycling in strong wind (Delhi) | Mocked — wind 55 km/h |
| 2 | Jogging in heat (Chennai) | Mocked — temp 39°C |
| 3 | Children at playground (Jaipur) | Mocked — UV 9 |
| 4 | Bike ride in Bhopal | **Live API call** |
| 5 | Stargazing in mild conditions (Pune) | Mocked — no SOP should fire |
| 6 | Weather API failure (Paris) | Mocked httpx exception |
| 7 | Adversarial prompt injection | Mocked — wind 55 km/h |
| 8 | Picnic in comfortable conditions (Bangalore) | Mocked — fuzzy SOP |

**Note on Test 4 (live API):** This test makes a real call to Open-Meteo and the matched SOP (or NONE) will vary with actual weather conditions in Bhopal. The test passes as long as real weather numbers are returned and the response is not an error message — it does not assert a specific SOP ID, since current conditions may not trigger any threshold.

---

## Project structure

```
weather-advisory-bot/
├── .env.example          # Copy to .env and add your API key
├── .gitignore
├── README.md
├── requirements.txt
├── sops/
│   └── sops.yaml         # All 12 SOPs — edit here to change policies
├── src/
│   ├── __init__.py
│   ├── sops_loader.py    # Loads/parses SOPs; numeric condition checker
│   ├── weather.py        # Open-Meteo geocoding + forecast API calls
│   ├── graph.py          # LangGraph graph + run_conversation()
│   └── prompts.py        # All LLM prompt templates
├── app.py                # Streamlit chat frontend
└── eval/
    └── eval_suite.py     # 8-test evaluation suite
```
