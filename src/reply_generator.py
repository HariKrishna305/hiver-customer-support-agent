import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


class AmazonReplyGenerator:

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")

        self.client = genai.Client(api_key=api_key)

        self.model = "gemini-3.6-flash"

    def generate_reply(self, customer_message, retrieved_conversations):

        examples = []

        for i, item in enumerate(retrieved_conversations, start=1):
            examples.append(
                f"""
Example {i}

Customer:
{item['customer_message']}

Amazon historical reply:
{item['brand_reply']}
"""
            )

        examples_text = "\n".join(examples)

        prompt = f"""
You are an Amazon customer support reply assistant.

Your task is to draft a short and professional reply to the customer.

CUSTOMER MESSAGE:
{customer_message}

Here are historical Amazon customer-support conversations that are
similar to the current message:

{examples_text}

Instructions:
1. Use the historical replies as grounding evidence.
2. Do not invent policies, refunds, delivery dates, account information,
   or actions that are not supported by the examples.
3. Do not expose internal IDs, tweet IDs, or agent initials.
4. Do not copy URLs from examples unless they are clearly relevant.
5. Be concise and professional.
6. If the issue requires account-specific investigation, ask the customer
   to contact Amazon support or provide the appropriate next step.
7. Return ONLY the proposed customer-facing reply.
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return response.text.strip()