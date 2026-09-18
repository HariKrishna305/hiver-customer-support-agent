# Hiver SDE Intern — Take-Home Assignment Report

## 1. Problem Framing

The goal of this project is to build an AI customer-support agent using real customer-support conversations from Twitter.

I selected Amazon as the target brand and built a pipeline that performs three main tasks:

1. Classifies incoming customer messages into support intents.
2. Retrieves historically similar Amazon customer-support conversations and uses them as grounding evidence to draft a reply.
3. Decides whether the conversation can be automatically handled or should be escalated to a human, with a reason.

### What Good Means

For this project, a good support agent should:

- Identify the customer's issue correctly.
- Use historical Amazon support conversations as evidence.
- Generate a relevant and professional response.
- Avoid unsupported information.
- Escalate sensitive or uncertain cases instead of confidently giving an unsupported response.

### What I Chose Not to Build

I focused on the core take-home requirements rather than building a complete production customer-support platform.

I did not build:

- A production Twitter/X integration.
- Real customer account authentication.
- Real order lookup.
- Real payment processing.
- Real-time human-agent routing.
- Production deployment and monitoring.

The system is an evaluation-focused prototype designed to demonstrate the complete AI-support workflow and how its quality can be measured.

---

## 2. Dataset and Data Preparation

The primary dataset is the Customer Support on Twitter dataset containing real customer-support conversations between customers and brands.

The raw dataset was transformed into customer-to-brand conversation pairs containing:

- Customer message
- Brand reply
- Customer tweet ID
- Brand tweet ID

I selected Amazon as the target brand and created an Amazon-specific dataset for the support agent.

The processed Amazon conversation dataset contains approximately 168,814 conversation records.

The pipeline removes unnecessary records and creates customer-message/brand-reply pairs so that historical support interactions can be used for retrieval and response generation.

---

## 3. Intent Classification

I defined eight support intents from the labelled data:

- `account_access`
- `billing_payment`
- `complaint_feedback`
- `delivery_shipping`
- `information_request`
- `other`
- `refund_cancellation`
- `technical_service_issue`

A TF-IDF + Logistic Regression classifier was used as the initial intent-classification model.

### Golden Evaluation

I created a 200-example golden evaluation set.

Of these, 151 examples were manually labelled and used for evaluation.

The intent classifier achieved:

**Accuracy: 69.54%**

The results show strong performance on several intents but weak performance on minority intents.

For example:

- `complaint_feedback` recall: 100%
- `delivery_shipping` recall: 74%
- `information_request` recall: 80%
- `technical_service_issue` recall: 86%

However, `account_access`, `billing_payment`, `refund_cancellation`, and `other` had 0% recall in this evaluation.

This indicates that the classifier is strongly affected by the limited and imbalanced labelled training data.

---

## 4. Baselines

Two baselines were implemented.

### Trivial Baseline

The trivial baseline always predicts the majority class.

The majority intent was:

`complaint_feedback`

Results on the 31-example test set:

**Accuracy: 29.03%**

### Keyword Baseline

A simple keyword-based intent classifier was also implemented.

Results on the same 31-example test set:

**Accuracy: 32.26%**

### Model Comparison

| Method | Accuracy |
|---|---:|
| Trivial majority baseline | 29.03% |
| Keyword baseline | 32.26% |
| TF-IDF + Logistic Regression | 35.48% |

The model therefore improves over both simple baselines on the original 31-example test set.

On the larger 151-example labelled golden set, the classifier achieved 69.54% accuracy.

The difference between these evaluation results is discussed in the failure-analysis section.

---

## 5. Historical Retrieval and Grounded Reply Generation

The response-generation pipeline uses historical Amazon support conversations.

The basic flow is:

Customer message

↓

Intent classification

↓

Semantic retrieval from historical Amazon conversations

↓

Retrieve similar customer-support examples

↓

Use historical replies as grounding evidence

↓

Generate a response

The historical conversations are stored in ChromaDB using embeddings.

The retrieval step searches for customer messages that are semantically similar to the incoming customer message.

This allows the response generator to use examples of how Amazon support historically responded to similar problems rather than generating an answer without supporting evidence.

---

## 6. Escalation Decision

The escalation component uses several signals:

- Sensitive keywords
- Account-access intent
- Availability of historical retrieval results
- Retrieval similarity/confidence

Sensitive issues such as fraud, scams, unauthorized activity, account hacking, and payment disputes are configured as escalation triggers.

Account-access issues are also escalated because they may require customer-specific verification.

If no useful historical conversation is retrieved, the system escalates instead of automatically generating a response.

If the best retrieval distance is above the configured threshold, the system also escalates because the historical evidence may not be sufficiently similar.

Otherwise, the system allows automatic handling.

Each escalation decision includes a reason so that the decision is explainable.

---

## 7. Escalation Evaluation

The escalation system was evaluated on the 151 labelled golden examples.

Results:

**Accuracy: 51.66%**

The evaluation showed:

- Expected escalations: 70
- Correctly detected escalations: 2
- Escalation recall: 3%

The result shows that the current rule-based escalation logic is conservative in some areas but does not capture many cases labelled for escalation.

This is an important limitation of the current prototype.

A future version should use a dedicated escalation classifier trained on the labelled golden dataset and combine intent, retrieval confidence, and safety signals.

---

## 8. Reply Quality Evaluation

An LLM-as-judge evaluation was implemented to evaluate generated replies on five dimensions:

- Relevance
- Grounding
- Helpfulness
- Professionalism
- Hallucination

Each dimension is scored from 1 to 5.

The initial evaluation successfully evaluated 5 examples because the Gemini API free-tier request quota was reached during the larger evaluation attempt.

Results from the 5 successfully evaluated examples:

| Metric | Average |
|---|---:|
| Relevance | 4.00/5 |
| Grounding | 5.00/5 |
| Helpfulness | 4.00/5 |
| Professionalism | 4.60/5 |
| Hallucination | 5.00/5 |
| Overall | 4.20/5 |

These results indicate that the evaluated responses were generally grounded and professional, but the sample size is too small to treat the score as representative of the complete system.

---

## 9. Human vs LLM-Judge Agreement

Five generated responses were independently evaluated by a human using the same scoring dimensions.

The comparison produced:

- Human average: 4.44/5
- LLM average: 4.52/5
- Overall MAE: 0.08
- Relevance exact agreement: 80%
- Helpfulness exact agreement: 80%
- Professionalism exact agreement: 100%
- Grounding exact agreement: 100%
- Hallucination exact agreement: 100%

The results show close agreement on this small sample.

However, only five examples were used, so the agreement should be treated as preliminary rather than conclusive.

---

## 10. Failure Analysis

### Failure 1: Intent Classifier Over-Predicts Complaint Feedback

The classifier frequently predicts `complaint_feedback`.

On the golden set, `complaint_feedback` had 100% recall, while several minority intents had 0% recall.

### Hypothesis

The labelled training data is small and imbalanced. The classifier has substantially more examples for `complaint_feedback` than some minority classes.

### Improvement

Increase labelled examples for minority intents and use a larger held-out evaluation set.

---

### Failure 2: Escalation Logic Misses Many Escalation Cases

The escalation classifier achieved only 3% recall for expected escalations.

### Hypothesis

The current system relies heavily on predefined keywords and simple rules. Many real customer-support situations require escalation without containing the predefined keywords.

### Improvement

Train a dedicated escalation model using the golden set and combine message features, intent, retrieval confidence, and safety signals.

---

### Failure 3: Incorrect Retrieval Can Produce an Irrelevant Reply

One evaluated example received an overall score of 1/5.

The customer complained about a Lightning Deal, but the generated response discussed a credit/debit-card amount being credited.

### Hypothesis

A retrieved historical conversation can be semantically similar but operationally different from the current customer problem.

### Improvement

Retrieve multiple candidates, improve similarity filtering, and add a response-verification step before automatic handling.

---

### Failure 4: Small Evaluation Sets Produce Unstable Headline Metrics

The original test set contained only 31 examples and produced 35.48% accuracy.

The larger 151-example golden set produced 69.54% accuracy.

### Hypothesis

The evaluation samples have different distributions and difficulty levels, while the labelled dataset itself is small.

### Improvement

Use a larger stratified evaluation dataset and maintain a strict held-out test set.

---

### Failure 5: LLM-Judge Validation Uses Only Five Human Examples

The human-vs-LLM evaluation used only five examples.

### Hypothesis

The small sample does not provide enough evidence to establish judge reliability across many types of customer-support responses.

### Improvement

Increase human-labelled judge-validation examples and measure agreement across a wider range of support scenarios.

---

## 11. What Is Misleading About My Headline Number?

The headline intent accuracy of 69.54% on the 151-example golden set does not mean that the classifier performs equally well across all intents.

The classifier performs strongly on several intents but completely misses some minority classes.

The original 31-example test set produced only 35.48% accuracy, showing that the measured performance depends substantially on the evaluation sample.

The reply-quality score of 4.20/5 is also based on only five successfully evaluated examples. One of those examples received an overall score of 1/5.

The escalation accuracy of 51.66% also hides an important weakness: escalation recall was only 3%.

Therefore, the headline numbers demonstrate that the system is functional, but they should not be interpreted as evidence that the agent is ready for unrestricted automatic customer support.

---

## 12. What I Would Do With One More Week

With another week, I would:

1. Increase labelled examples for minority intents.
2. Improve intent classification using better class balancing and evaluation.
3. Build a dedicated escalation classifier.
4. Improve retrieval filtering and confidence thresholds.
5. Add response verification before auto-handling.
6. Expand human evaluation beyond five examples.
7. Validate the LLM judge on a larger sample.
8. Add adversarial and ambiguous customer-support examples.
9. Perform larger end-to-end evaluation.
10. Improve safety rules for unsupported or uncertain responses.

---

## 13. Conclusion

This project demonstrates an end-to-end AI customer-support workflow using real customer-support data.

The system performs intent classification, historical semantic retrieval, grounded reply generation, and escalation decisions.

The evaluation harness compares the system with simple baselines and measures intent accuracy, escalation performance, reply quality, and human-vs-LLM judge agreement.

The evaluation also identifies important limitations, particularly minority-intent classification, escalation recall, retrieval errors, and limited evaluation sample sizes.

The main focus of the project is therefore not only building the AI system, but also measuring where it works and where it should not yet be trusted.