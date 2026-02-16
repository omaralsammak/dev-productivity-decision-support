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
  ((https://storage.googleapis.com/kagglesdsdata/datasets/6883361/11049592/issues.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260213%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260213T181504Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=2987e4e62a6218da01a0b179af3f7d8585251b13cdc015da06458cb7c4663d04099c77d40faca7a6f23d7b64a8c001bee41cd58f4e8892855d50432f2353c39c324312087dcca5fd48ffa1569388961c8b9f128335fb99968d5f8933c641f55b8d872c45039209b1b0a996d3b4f849d61f0120299256c9eacaee14f370d060ae5876569602e07d694e3136346bd99fcce8e425d808cc1705034d08bd503aa4dc3ebb697de66fd8c85f1b7247cb939c789650ce506415ddc01edcbe7896a3c7275553add742a489b36baa8ec689d862a26fc68ffa7fc96df36db1ab291fadef35f2901f1494adf4b19c46df3a233146f966cbdff0fd4b3cdb86b0cf168e8e7c08))

* Change Log Files:
  (https://storage.googleapis.com/kagglesdsdata/datasets/6883361/11049592/changelog.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260213%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260213T181429Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=8feb3f97a76d6a8e7481e60ebaaa58f0a3c9784a62ef6085a485271f2b0eaafcecf2fb28dd2422ab3284fe241f2bf04deaf43941d343c56e8f3a674c54da79014c317ed4fe8906637d6bbe9867010b8a53401e5da5705c3c14fa6923bd55173dee21b2c0ba616a21751e42051c9fa1f83eb878e60f7dec115f63877c2b3f53489c39179586936d96fbf77c805063e1e2467971cb07187d0634ab633d87e6e2392a02187b768c6adc26129bde919966fe54e8eed4e7ef67f8655aed158fe80e193ccb5b7bc27ac309c83139a038fee3aa01c47e93386ba65478dd5f936202aff25096e6fd60186873b6be98392c4e015154048561bca3111451c32f8de7f43ed9)

* Comments Dataset:
  (https://storage.googleapis.com/kagglesdsdata/datasets/6883361/11049592/comments.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260213%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260213T181452Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4f58588e1cff18530a85e491cd860a8ca96f4f0f579441f512b7ca76f9995e03eb2533f6f9022da84126d9cd8ed141c775c7228297662e006013fa283142e3114be366cc22ce005596b4dc887c98c76d446570e1575b075b39f23b74a13250d027718b980f3d5a57a321dc704c7eb9d12f45374e0376c0990998371612a6414c450d774c94531865578ddaca07d63b99f4b7b3c1d03d282301eeb8fb8a75db258f27eba803b3afc3da593021a75bd1bcf290a852cc2c113e95248293a4d88e76a2f063f4f02fd88b9f9a959979c3e53090b0b974fc730c7b6c72353e0cf8df3179b1b8c7d52503564a9e0ba33262b18e7baeb55f0186bb6d6c605d53349b7964)

* issuelinks dataset
  ( https://storage.googleapis.com/kagglesdsdata/datasets/6883361/11049592/issuelinks.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260213%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260213T181458Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4ae3cd66eb28eb0b75eed539eb7d1c0d1395c9c8fdca846a9aa5b18b47d932f1f0bf86824e44fd3e0b78d527a60276e4d8791ea77a7e6a83974bab377f1bf53d825b9774563c7501984dbeed9a9a31228764827d67234eb253a310bc03809b539fa2c756f7461d6f1b1a97aacda46cc960f664f1ca17901ba94dc48499759177f39fd52e2d8da6db197f74dff2e23e2b511d5ebc9916ca97d5c501d4c0ac491886d09a72fe20a03d12c5052811eb0fb235a1526bc6a46adf71aa400301ef7f4c99c11650eb82368e7373e10d37be3680146463e6ce5e5f21a26cd2129f40ccfa0c474ea18257011c1069d5c5bd8fa30bc7986e5b11556913f220d3dcdcdcd386)
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


