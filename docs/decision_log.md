# Decision Log

## 1. Selected Amazon as the Target Brand

I selected Amazon as the target brand from the Customer Support on Twitter dataset.

Why:
- Amazon has a large number of customer-support conversations.
- The conversations provide useful historical examples for retrieval and reply generation.
- A sufficiently large dataset makes it possible to build and evaluate a brand-specific support agent.

---

## 2. Converted Raw Tweets into Customer-Brand Conversation Pairs

I transformed the raw Twitter dataset into customer-message and brand-reply pairs.

Why:
- The raw dataset contains individual tweets and thread relationships.
- The support agent needs to learn from customer issues and how the brand responded.
- Pairing customer messages with brand replies provides useful retrieval and grounding evidence.

---

## 3. Removed Duplicate Conversation Pairs

Duplicate customer-message and brand-reply pairs were removed during preprocessing.

Why:
- Duplicate examples can unnecessarily increase the influence of repeated conversations.
- Removing duplicates provides cleaner historical data for retrieval and evaluation.

---

## 4. Created a Small Set of Support Intents

I defined eight intents:

- account_access
- billing_payment
- complaint_feedback
- delivery_shipping
- information_request
- other
- refund_cancellation
- technical_service_issue

Why:
- A small intent set is easier to evaluate and understand.
- The categories were derived from the customer-support data.
- The categories cover common support scenarios without creating an unnecessarily large classification problem.

---

## 5. Used TF-IDF + Logistic Regression for Intent Classification

I selected TF-IDF with Logistic Regression as the initial classifier.

Why:
- It is simple and fast to train.
- It provides a strong baseline for text classification.
- It is easy to reproduce and explain during a technical interview.
- It does not require a large GPU or expensive model.

---

## 6. Created a 200-Example Golden Evaluation Set

I created a golden evaluation set containing 200 examples.

Of these, 151 examples were manually labelled and used for evaluation.

Why:
- The assignment specifically requires a 150–250 example hand-labelled golden set.
- A manually labelled set provides an independent evaluation source.
- Keeping the remaining examples available provides room for additional labelling and validation.

---

## 7. Kept the Golden Set Separate from the Training Data

The manually labelled golden examples were evaluated separately from the training data.

Why:
- Evaluation data should not simply be treated as training data.
- Separating evaluation examples provides a more meaningful measurement of system performance.

---

## 8. Implemented Two Baselines

I implemented:
- A majority-class trivial baseline.
- A keyword-based simple baseline.

Why:
- The assignment requires comparison against at least two baselines.
- The trivial baseline shows the performance of simply predicting the most common intent.
- The keyword baseline provides a simple rule-based comparison.

---

## 9. Used ChromaDB for Historical Conversation Retrieval

I used ChromaDB to store embeddings of historical Amazon conversations.

Why:
- The response generator needs to find semantically similar historical conversations.
- Vector retrieval allows similar messages to be found even when they do not use exactly the same words.
- ChromaDB is lightweight and suitable for a local prototype.

---

## 10. Used Historical Brand Replies as Grounding Evidence

The response generator uses retrieved historical Amazon conversations and their replies as grounding evidence.

Why:
- The assignment requires replies to be grounded in how the selected brand historically resolved similar issues.
- Using historical responses reduces the need for the model to generate unsupported responses from general knowledge.

---

## 11. Added an Escalation Layer

I implemented a separate escalation decision component instead of automatically replying to every customer message.

Why:
- Some customer issues require human intervention.
- Sensitive issues such as fraud, unauthorized activity, account hacking, and payment disputes should not be handled automatically without additional verification.
- The escalation component also considers retrieval availability and similarity.

---

## 12. Added Explainable Escalation Reasons

Each escalation decision returns a reason.

Why:
- The assignment requires the system to state why a message is escalated.
- An explicit reason makes the system easier to debug and evaluate.
- It also makes the decision easier for a human support agent to understand.

---

## 13. Added Retrieval Confidence to the Escalation Decision

The escalation logic checks the distance of the best retrieved historical conversation.

Why:
- A response should not be automatically generated when there is insufficient historical evidence.
- Low retrieval similarity can indicate that the current issue is different from the available historical examples.
- Escalating uncertain cases is safer than generating an unsupported response.

---

## 14. Used an LLM-as-Judge for Reply Quality

I evaluated generated replies using five dimensions:

- Relevance
- Grounding
- Helpfulness
- Professionalism
- Hallucination

Why:
- Reply quality cannot be measured only by intent accuracy.
- The assignment specifically requires an LLM-as-judge evaluation.
- Multiple dimensions provide a more detailed view of response quality.

---

## 15. Compared the LLM Judge Against Human Evaluation

I manually evaluated five generated responses using the same scoring dimensions and compared them with the LLM judge.

Why:
- An LLM judge should not automatically be assumed to be reliable.
- Human comparison provides evidence about whether the judge's scores are reasonably aligned with human assessment.
- The small sample is reported as a limitation rather than treated as definitive evidence.