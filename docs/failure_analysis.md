# Failure Analysis

The evaluation results show that the system is functional, but performance is not uniform across all parts of the pipeline. The following are the five main failure modes identified from the evaluation results and observed examples.

## 1. Intent Classifier Over-Predicts `complaint_feedback`

The intent classifier achieved 69.54% accuracy on the 151-example labelled golden set. However, the performance was highly uneven across intents.

The classifier achieved 100% recall for `complaint_feedback`, while several other intents had 0% recall:

- `account_access`: 0%
- `billing_payment`: 0%
- `refund_cancellation`: 0%
- `other`: 0%

For example:

Customer message:

> I was charged twice for my order.

Predicted intent:

> complaint_feedback

Expected intent:

> billing_payment

### Hypothesis

The training dataset contains only 120 examples and is imbalanced across the eight intents. `complaint_feedback` has 36 training examples, while some minority intents have only 5–8 examples. TF-IDF with Logistic Regression therefore has insufficient examples to learn reliable boundaries for several minority classes.

### Improvement

Increase labelled training examples for minority intents and evaluate the classifier using a larger held-out dataset.

---

## 2. Escalation Logic Misses Many Escalation Cases

The escalation evaluation produced 51.66% accuracy on 151 labelled examples.

The most important issue is escalation recall:

- Expected escalations: 70
- Correctly detected escalations: 2
- Escalation recall: 3%

The current escalation system relies primarily on:

- Sensitive-issue keywords
- Account-access intent
- Retrieval availability
- Retrieval-distance threshold

This means an escalation case that does not contain one of the predefined keywords may be automatically handled.

### Hypothesis

The current rule-based approach does not capture the full range of situations that require human intervention. Customer-support messages can require escalation without explicitly containing words such as "fraud", "scam", or "hacked".

### Improvement

Build and evaluate a dedicated escalation classifier using the labelled golden set. Combine message features, intent, retrieval confidence, and potentially other safety signals instead of relying mainly on keywords.

---

## 3. Incorrect Retrieval Can Produce an Irrelevant Reply

The reply-quality evaluation included an example where the generated response did not address the customer's actual issue.

Customer message:

> you people are cheaters.. lightening deal is just for name sake..

Generated reply:

> Please be informed that the amount will be credited to the Bank account linked with Citi Credit or Debit Card latest by January 08, 2018.

The LLM judge gave this example:

- Relevance: 1/5
- Helpfulness: 1/5
- Overall: 1/5

Although the generated response was professionally written, it did not address the customer's Lightning Deal complaint.

### Hypothesis

The response generator depends on retrieved historical conversations for grounding. A semantically related but operationally incorrect historical conversation can therefore influence the generated response.

### Improvement

Improve retrieval evaluation and filtering. Use stronger similarity thresholds, retrieve multiple candidates, and add a verification step that checks whether the proposed response actually addresses the customer's intent and retrieved evidence.

---

## 4. Small Evaluation Sets Make Headline Numbers Unstable

The intent classifier produced different results on different evaluation sets:

- Original test set: 31 examples
- Test accuracy: 35.48%
- Golden set: 151 examples
- Golden-set accuracy: 69.54%

This difference demonstrates that the measured performance depends strongly on the evaluation sample.

The training dataset itself contains only 120 examples across eight intents.

### Hypothesis

The available labelled data is small relative to the number of intents, and the evaluation samples may contain different distributions and levels of difficulty.

### Improvement

Create a larger and more carefully stratified evaluation set, maintain a strict held-out test set, and report per-intent precision, recall, and F1 rather than relying only on overall accuracy.

---

## 5. LLM-Judge Validation Is Based on a Very Small Human Sample

The reply-quality evaluation produced an average overall score of 4.20/5 across 5 successfully evaluated examples.

A separate human comparison was performed on those 5 examples.

The results were:

- Overall human average: 4.44/5
- Overall LLM average: 4.52/5
- Overall MAE: 0.08
- Relevance exact agreement: 80%
- Helpfulness exact agreement: 80%
- Professionalism exact agreement: 100%

However, only five examples were used for the human-vs-LLM comparison.

### Hypothesis

The preliminary agreement is encouraging, but five examples are not enough to establish that the LLM judge reliably agrees with humans across the full range of customer-support scenarios.

### Improvement

Evaluate a substantially larger sample with independent human ratings before using the LLM judge as the primary quality-evaluation method.

---

# Summary

The main weaknesses of the current system are concentrated in intent-class imbalance, escalation detection, retrieval reliability, limited labelled evaluation data, and the small size of the human validation sample.

These failures do not invalidate the system. Instead, they identify where additional data, stronger evaluation methodology, and additional safeguards would provide the largest improvements.