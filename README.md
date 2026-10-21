# Low-label predicate detection: baseline MLP vs class-balanced RBF-SVM

This script compares two per-predicate classifiers on a digits-based stand-in for object predicates (scikit-learn `load_digits`, five binary predicates derived from the digit label).

## How to run

```
pip install -r requirements.txt
python analysis.py
```

The script writes `results.json` and `figures/accuracy_bars.png`. It runs on CPU in well under a minute.

## Method

- Data: `load_digits`, split once into train pool and test set (30%, stratified by digit, seed 42). The test set is used only for final evaluation.
- Labelled subsets: 1% and 5% of the training pool, stratified, seed 42.
- Baseline: per-predicate `MLPClassifier` (64 hidden units) on raw pixels, threshold 0.5.
- Improved: per-predicate class-balanced RBF-SVM on raw pixels, with C and gamma chosen by stratified cross-validated balanced accuracy on the labelled subset only.
- Seeds 42, 43, 44. Metric: mean predicate accuracy on the test set.

## Caveats

- The predicates are deterministic functions of the digit label, so they are much easier than real object attributes. Results are not comparable to the paper's numbers.
- With 12 to 62 labelled examples, CV-based selection is noisy.

## Results (latest run)

On the load_digits stand-in dataset with 12 labelled examples, a standard per-predicate MLP concept head reached 76.1% mean predicate accuracy versus 77.8% for a class-balanced cross-validated RBF-SVM, a small gain of 1.7 points, while worst-predicate accuracy fell from 74.1% to 71.1%.

| Metric | Value |
|---|---|
| baseline_score | 0.7606 |
| baseline_score_std | 0.002536 |
| improved_score | 0.7777 |
| improved_score_std | 0.009352 |
| improvement | 0.01704 |
| baseline_f1 | 0.7258 |
| improved_f1 | 0.7545 |
| baseline_min_acc | 0.7407 |
| improved_min_acc | 0.7111 |
| n_labelled | 12 |
| baseline_score_5pct | 0.8849 |
| baseline_score_std_5pct | 0.004194 |
| improved_score_5pct | 0.9152 |
| improved_score_std_5pct | 0.005238 |
| improvement_5pct | 0.03025 |
| baseline_f1_5pct | 0.8617 |
| improved_f1_5pct | 0.8963 |
| baseline_min_acc_5pct | 0.8654 |
| improved_min_acc_5pct | 0.9019 |
| n_labelled_5pct | 62 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in data: predicates are deterministic functions of the digit label, so they are far easier than real object attributes and the paper's 0.746 is not comparable. The labelled subsets are tiny (a few dozen examples at most), so CV choices are noisy. Unlabelled pool is unused by design. Improved model falls back to a constant predictor if a class has fewer than 3 labelled examples. The previous improved model (PCA + logistic regression + isotonic calibration + threshold search) was replaced because its threshold and calibration steps overfit the tiny labelled sets. Std is across the three seeds (population std). The test set is used once per model and seed; no selection on it.

## Reference

This project is motivated by the following paper:

> S Tiwari, B Finzel, G Schwalbe (2026). *Weakly Supervised Concept Learning for Object-centric Visual Reasoning*. arXiv preprint arXiv:2605.08201. https://arxiv.org/abs/2605.08201

Paper author: Prof. Gesina Schwalbe, Ulm University (gesina.schwalbe@uni-ulm.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
