# AI-Based Decision Support for Enhancing Software Development Team Productivity

## Overview

This project implements a machine-learning–based decision support system for analyzing software development workflows and detecting productivity bottlenecks in software teams.
The system uses issue-tracking data and developer activity logs to estimate productivity patterns and support data-driven project management decisions.

The repository accompanies an academic research project focused on improving software development team productivity through AI-assisted analytics.

---

## Objectives

* Analyze software development workflow data
* Detect bottlenecks in issue resolution processes
* Estimate productivity indicators using regression models
* Compare multiple machine learning algorithms
* Provide reproducible experimental results

---

## Dataset

This project uses issue-tracking datasets derived from Apache JIRA repositories.

Dataset components may include:

* Issues metadata
* Change logs
* Comments
* Issue links
* Developer activity records

You can download the dataset from the following sources:

Dataset links:

* Apache JIRA Issues Dataset:
  (PUT DATASET LINK HERE)

* Change Log Files:
  (PUT DATASET LINK HERE)

* Comments Dataset:
  (PUT DATASET LINK HERE)

If datasets are large, download them separately and place them in the `data/` directory.

Expected structure:

```
project/
│
├── data/
│   ├── issues.csv
│   ├── change_log.csv
│   ├── comments.csv
│   └── issuelinks.csv
```

---

## Requirements

Python version:

```
Python 3.10
```

Main libraries:

* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* XGBoost / LightGBM (if used)

Install dependencies using:

```
pip install -r requirements.txt
```

---

## Project Structure

```
project/
│
├── data/
├── models/
├── results/
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## How to Run

Step 1 — Install dependencies:

```
pip install -r requirements.txt
```

Step 2 — Run training:

```
python train.py
```

Step 3 — Run evaluation:

```
python evaluate.py
```

---

## Reproducibility

To reproduce the experimental results:

* Use the same dataset version referenced in this repository
* Use Python 3.10
* Install dependencies from `requirements.txt`
* Run the scripts in the order described above

Random seeds are fixed where applicable to ensure reproducibility.

---

## Release Information

Version: v1.0
This release corresponds to the experiments reported in the research paper.

---

## License

MIT License

---

## Citation

If you use this code in academic work, please cite:

```
AI-Based Decision Support for Enhancing Software Development Team Productivity, 2026.
```

---

## Contact

For questions regarding this repository or the research work:

Your Name
[your_email@university.edu](mailto:your_email@university.edu)
