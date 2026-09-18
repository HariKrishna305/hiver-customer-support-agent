# Hiver SDE Intern Take-Home Assignment

## Amazon Customer Support AI Agent

An AI customer-support agent built using real customer-support conversations from the Customer Support on Twitter dataset.

The system performs three main tasks:

1. Classifies customer messages into support intents.
2. Retrieves historically similar Amazon conversations and generates a grounded reply.
3. Decides whether the message should be automatically handled or escalated to a human, with a reason.

---

# 1. Problem Framing

The goal is to build a brand-specific customer-support assistant for Amazon using historical customer-support conversations.

A good system should:

- Identify the customer's intent correctly.
- Generate replies based on how Amazon historically handled similar issues.
- Avoid unsupported responses.
- Escalate sensitive or uncertain cases to a human.
- Provide an explanation for escalation decisions.
- Be measurable through an independent evaluation set.

## What I Chose Not to Build

This project focuses on a prototype support-agent pipeline rather than a production customer-support platform.

I did not build:

- A production Twitter integration.
- Real-time customer authentication.
- Actual order/account lookup.
- Automatic refunds or account changes.
- A production human-agent dashboard.
- Fine-tuning of a large language model.
- Full multi-turn conversation state management.

These features would require production infrastructure, authentication, business-system integrations, and additional safety controls.

---

# 2. Architecture

```text
                         Raw Twitter Dataset
                                  |
                                  v
                           Data Cleaning
                                  |
                                  v
                    Customer-Brand Conversation Pairs
                                  |
                     +------------+------------+
                     |                         |
                     v                         v
             Intent Classification     Historical Conversations
                     |                         |
                     |                         v
                     |                  Embeddings + ChromaDB
                     |                         |
                     +------------+------------+
                                  |
                                  v
                         Retrieved Evidence
                                  |
                                  v
                          Reply Generator
                                  |
                                  v
                        Escalation Decision
                                  |
                       +----------+----------+
                       |                     |
                       v                     v
                  Auto Handle        Human Escalation