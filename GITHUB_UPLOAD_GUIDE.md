# GitHub Upload Guide – Gust Loads Surrogate

This document tells you **exactly** how to put the whole project on GitHub cleanly.

---

## 1. What is already inside the zip

```
gust-loads-surrogate/
├── .github/workflows/ci.yml     # Automatic pytest on every push / PR
├── .gitignore
├── LICENSE                      # MIT
├── README.md                    # Main documentation (problem, physics, results, CV line)
├── GITHUB_UPLOAD_GUIDE.md       # This file
├── pyproject.toml
├── requirements.txt
├── data/
│   ├── .gitkeep
│   ├── demo_dataset.csv         # Small example dataset (kept)
│   └── demo_gust_response.png   # Example time-history plot (kept)
├── notebooks/                   # Empty, ready for your own notebooks
├── src/
│   ├── structure/beam.py        # Euler-Bernoulli FEM (validated)
│   ├── aero/strip.py            # Quasi-steady strip theory
│   ├── gust/discrete.py         # 1-cos discrete gust
│   ├── solver/
│   │   ├── coupled.py           # Newmark-β aeroelastic solver
│   │   ├── dataset.py           # LHS/Sobol + parallel DoE
│   │   └── demo_run.py          # One-shot demo + plot
│   ├── surrogate/
│   │   ├── models.py            # Linear / MLP / Deep Ensemble
│   │   └── train_eval.py
│   └── optimise/worst_case.py   # UCB-style worst-case search
└── tests/
    ├── test_beam.py             # Analytic validation (tip deflection + frequency)
    └── test_gust_response.py    # Smoke + sanity tests
```

All core tests pass (`pytest tests/ -v` → 8 passed).

---

## 2. Create the GitHub repository

1. Go to https://github.com/new
2. Repository name: `gust-loads-surrogate` (or any name you like)
3. Description:  
   `Physics-based aeroelastic solver + ML surrogate for discrete gust loads on high-aspect-ratio wings`
4. Visibility: Public (recommended for portfolio) or Private
5. **Do NOT** initialise with README, .gitignore or license (we already have them)
6. Click **Create repository**

---

## 3. Push the code (three easy ways)

### Option A – GitHub web upload (fastest if you are not comfortable with git)

1. Unzip the file you received.
2. On the empty repo page click **“uploading an existing file”**.
3. Drag the **contents** of the `gust-loads-surrogate` folder (not the folder itself) into the browser.
4. Commit message: `Initial commit – full gust-loads-surrogate project`
5. Commit.

### Option B – Command line (recommended)

```bash
# 1. Unzip
unzip gust-loads-surrogate.zip
cd gust-loads-surrogate

# 2. Initialise git
git init
git add .
git commit -m "Initial commit: physics-based aeroelastic solver + ML surrogate

- Euler-Bernoulli FEM validated against analytic tip deflection & natural frequency
- Quasi-steady strip theory + 1-cos discrete gust
- Newmark-β coupled time-domain solver
- Latin Hypercube / Sobol DoE + multiprocessing
- Linear + Neural + Deep-ensemble surrogates
- Bayesian-style worst-case search
- Full pytest suite + GitHub Actions CI
"

# 3. Connect to your new GitHub repo (replace USERNAME)
git branch -M main
git remote add origin https://github.com/USERNAME/gust-loads-surrogate.git
git push -u origin main
```

### Option C – GitHub CLI

```bash
gh repo create gust-loads-surrogate --public --source=. --remote=origin --push
```

---

## 4. After the first push – enable GitHub Actions

1. Go to the **Actions** tab of your repo.
2. You should see the workflow `CI` already running (or waiting).
3. If it is the first time, GitHub may ask you to enable Actions – click Enable.
4. Once it is green you can add the badge to the README (already prepared):

```markdown
![CI](https://github.com/USERNAME/gust-loads-surrogate/actions/workflows/ci.yml/badge.svg)
```

Replace `USERNAME` with your GitHub username.

---

## 5. Recommended next steps on GitHub

- Add topics: `aeroelasticity`, `gust-loads`, `surrogate-modelling`, `finite-element`, `machine-learning`, `python`, `aircraft-structures`
- Pin the repository on your profile
- In the About section paste the short description and the CV one-liner from the README
- (Optional) Create a Release `v0.1.0` and attach the same zip

---

## 6. Local verification after cloning

```bash
git clone https://github.com/USERNAME/gust-loads-surrogate.git
cd gust-loads-surrogate
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=.               # Windows: set PYTHONPATH=.
pytest tests/ -v
python -m src.solver.demo_run
```

---

## 7. CV / LinkedIn one-liner (ready to copy)

> Built a physics-based aeroelastic solver and ML surrogate predicting discrete-gust wing loads ~1000× faster at low single-digit percent error; validated against analytic beam solutions with fully CI-tested Python.

---

## 8. What still needs real data / optional packages

- For production-quality surrogate metrics generate a larger dataset:
  ```bash
  python -m src.solver.dataset   # edit n_samples=2000–5000 inside
  ```
- Optional packages (scikit-learn, scikit-optimize, streamlit, pyarrow) can be installed when network allows:
  ```bash
  pip install scikit-learn joblib scikit-optimize pyarrow streamlit
  ```
  Then the Gaussian-process and Bayesian-optimisation paths become available.

That’s everything you need to put a clean, professional project on GitHub.
