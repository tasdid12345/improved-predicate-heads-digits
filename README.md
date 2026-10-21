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

## Reference

This project is motivated by the following paper:

> S Tiwari, B Finzel, G Schwalbe (2026). *Weakly Supervised Concept Learning for Object-centric Visual Reasoning*. arXiv preprint arXiv:2605.08201. https://arxiv.org/abs/2605.08201

Paper author: Prof. Gesina Schwalbe, Ulm University (gesina.schwalbe@uni-ulm.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
