"""
Interactive Negotiation Agent (LangChain + Mock NLP)

File: negotiation_agent_manual.py

In this version:
- Seller messages are typed manually by YOU (the user acts as the seller).
- The agent (AI) negotiates toward the target price using negotiation policy + NLP.
- LLM (or mock) is used to classify seller messages and compose polite offers.

Run:
    python negotiation_agent_manual.py
"""

import os
import time
from typing import Optional, Dict, Any

# === Config ===
PROVIDER = os.environ.get("NEG_AGENT_PROVIDER", "mock")  # "mock" or "openai"

# === Mock NLP / LLM Response ===
def mock_classify(text: str) -> str:
    text = text.lower()
    if any(x in text for x in ["can't", "no", "fixed", "not"]):
        return "decline"
    if any(x in text for x in ["do", "final", "take", "offer"]):
        return "counter_offer"
    if "?" in text:
        return "question"
    return "other"

def mock_generate_offer_message(price: int) -> str:
    return f"I understand. How about ₹{price}? I can pick it up today."

# === Negotiation Policy ===
class NegotiationPolicy:
    def __init__(self, user_target: int, max_budget: int, anchor_factor: float = 0.6):
        self.user_target = user_target
        self.max_budget = max_budget
        self.anchor_factor = anchor_factor
        self.current_offer = None
        self.round_no = 0

    def initial_offer(self, listing_price: int) -> int:
        anchor = int(self.user_target * self.anchor_factor)
        self.current_offer = min(anchor, listing_price)
        self.round_no = 1
        return self.current_offer

    def next_offer(self, seller_offer: Optional[int]) -> int:
        self.round_no += 1
        if seller_offer:
            next_offer = (seller_offer + self.user_target) // 2
        else:
            next_offer = self.current_offer + 500
        if next_offer > self.max_budget:
            next_offer = self.max_budget
        self.current_offer = next_offer
        return self.current_offer

    def should_accept(self, seller_offer: int) -> bool:
        return seller_offer <= self.user_target

# === Agent ===
class NegotiationAgent:
    def __init__(self, target_price: int, max_budget: int, provider="mock"):
        self.policy = NegotiationPolicy(target_price, max_budget)
        self.provider = provider

    def process_seller_message(self, message: str) -> Dict[str, Any]:
        intent = mock_classify(message)  # replace with LLM if using LangChain
        return {"intent": intent, "text": message}

    def make_offer(self, seller_offer: Optional[int], listing_price: int) -> str:
        if self.policy.round_no == 0:
            offer = self.policy.initial_offer(listing_price)
        elif seller_offer and self.policy.should_accept(seller_offer):
            return f"✅ Great, I accept ₹{seller_offer}."
        else:
            offer = self.policy.next_offer(seller_offer)

        return mock_generate_offer_message(offer)

# === Interactive Loop ===
def interactive_session():
    print("\n=== Negotiation Agent (Manual Seller Input) ===")
    listing_price = int(input("Enter seller's listing price (₹): "))
    target_price = int(input("Enter your target price (₹): "))
    budget = int(input("Enter your max budget (₹): "))

    agent = NegotiationAgent(target_price, budget)

    print("\nYou are the SELLER. Type seller replies manually.")
    print("Type 'exit' to end the session.\n")

    seller_offer = None
    while True:
        seller_msg = input("Seller: ")
        if seller_msg.lower() in ["exit", "quit"]:
            print("[Session Ended]")
            break

        # Parse seller input (check for number/offer)
        words = seller_msg.replace(",", "").split()
        seller_offer = None
        for w in words:
            if w.isdigit():
                seller_offer = int(w)

        # Agent reply
        response = agent.make_offer(seller_offer, listing_price)
        print(f"Agent: {response}\n")

if __name__ == "__main__":
    interactive_session()