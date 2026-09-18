from src.retriever import AmazonRetriever
from src.reply_generator import AmazonReplyGenerator
from src.escalation import EscalationDecider


class AmazonSupportPipeline:

    def __init__(self):

        print("Initializing Amazon Support Pipeline...")

        # 1. Retriever
        self.retriever = AmazonRetriever()

        # 2. Reply Generator
        self.reply_generator = AmazonReplyGenerator()

        # 3. Escalation Decider
        self.escalation_decider = EscalationDecider()

        print("Pipeline initialized successfully.")

    def process(self, customer_message, intent):

        print("\nProcessing customer message...")

        # Step 1: Retrieve similar conversations
        print("Retrieving similar conversations...")

        retrieved_conversations = self.retriever.search(
            customer_message,
            top_k=5
        )

        # Step 2: Decide escalation
        print("Checking escalation...")

        escalation_result = self.escalation_decider.decide(
            customer_message,
            intent,
            retrieved_conversations
        )

        # Step 3: Generate reply only if safe to auto-handle
        if escalation_result["decision"] == "auto_handle":

            print("Generating Amazon support reply...")

            reply = self.reply_generator.generate_reply(
                customer_message,
                retrieved_conversations
            )

        else:

            reply = None

        return {
            "customer_message": customer_message,
            "intent": intent,
            "retrieved_conversations": retrieved_conversations,
            "escalation": escalation_result,
            "reply": reply
        }