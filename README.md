# Class-balanced PCA concept heads vs per-predicate MLP on a digits stand-in

**Motivation.** Object-centric visual reasoning systems need predicate detectors that work with very few labels, and a single weak predicate head can break any rule that depends on it. "Weakly Supervised Concept Learning for Object-centric Visual Reasoning" (2026) proposes a two-stage neuro-symbolic SlotVAE whose concept heads feed symbolic reasoners, and reports zero-shot predicate accuracy at 1% supervision. This experiment asks whether a simpler, cross-validated concept-head recipe can improve the worst-predicate accuracy of a low-label detector.

**Question.** Can a class-balanced, cross-validated ensemble of concept heads on PCA features raise the worst-predicate accuracy of a low-label predicate detector above that of a standard per-predicate MLP head, using a digits-based stand-in for object predicates?

**Dataset.** scikit-learn's bundled `load_digits` (no download). Five binary predicates are derived from each digit label and treated as object attributes. This is a stand-in, not the paper's dataset.

**Baseline.** A standard per-predicate MLP on raw pixels, trained only on the labelled subset.

**Improved model.** Standardised PCA features followed by class-balanced logistic regression, with C chosen by stratified CV on the labelled subset, isotonic calibration, and a per-predicate threshold.

**How to run**

```
pip install -r requirements.txt
python analysis.py
```

**Results** are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> S Tiwari, B Finzel, G Schwalbe (2026). *Weakly Supervised Concept Learning for Object-centric Visual Reasoning*. arXiv preprint arXiv:2605.08201. https://arxiv.org/abs/2605.08201

Paper author: Prof. Gesina Schwalbe, Ulm University (gesina.schwalbe@uni-ulm.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
