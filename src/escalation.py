class EscalationDecider:

    def decide(self, customer_message, intent, retrieved_conversations):

        message = customer_message.lower()

        # High-risk or sensitive issues
        escalation_keywords = [
            "fraud",
            "scam",
            "stolen",
            "unauthorized",
            "charged twice",
            "wrong charge",
            "account hacked",
            "hack",
            "security",
            "payment dispute",
        ]

        for keyword in escalation_keywords:
            if keyword in message:
                return {
                    "decision": "escalate",
                    "reason": f"Customer message contains a sensitive issue: '{keyword}'."
                }

        # Account-specific issues
        if intent == "account_access":
            return {
                "decision": "escalate",
                "reason": "Account access issues may require customer-specific verification."
            }

        # No retrieval results
        if not retrieved_conversations:
            return {
                "decision": "escalate",
                "reason": "No similar historical Amazon conversations were retrieved."
            }

        # Check retrieval confidence
        best_distance = retrieved_conversations[0]["distance"]

        if best_distance > 0.75:
            return {
                "decision": "escalate",
                "reason": "The retrieved historical examples are not sufficiently similar to the customer message."
            }

        # Automatic handling
        return {
            "decision": "auto_handle",
            "reason": "A sufficiently similar historical Amazon support pattern was found and no sensitive escalation trigger was detected."
        }