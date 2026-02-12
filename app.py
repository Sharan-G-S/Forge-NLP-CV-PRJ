# app.py
import streamlit as st
import random
import builtins
from typing import Dict, Any

# -----------------------------
# Mock Seller Personas
# -----------------------------
class SellerSimulator:
    def __init__(self, persona: str, listing_price: int):
        self.persona = persona
        self.listing_price = listing_price
        self.current_price = listing_price

    def respond(self, offer: int) -> Dict[str, Any]:
        if self.persona == "strict":
            # Hardly moves
            if offer >= self.listing_price:
                return {"accepted": True, "price": offer, "message": "Okay, deal."}
            elif offer >= self.listing_price - 1000:
                return {"accepted": False, "price": self.listing_price - 500, "message": "Lowest I can go is just a bit down."}
            else:
                return {"accepted": False, "price": self.listing_price, "message": "No, price is fixed."}

        elif self.persona == "flexible":
            # Moves gradually
            if offer >= self.listing_price - 2000:
                return {"accepted": True, "price": offer, "message": "Alright, let’s close it."}
            else:
                counter = max(offer + 1000, self.listing_price - 2000)
                return {"accepted": False, "price": counter, "message": f"I can do {counter}."}

        else:  # desperate
            if offer >= self.listing_price - 4000:
                return {"accepted": True, "price": offer, "message": "Deal, I really need to sell this."}
            else:
                counter = max(offer + 500, self.listing_price - 4000)
                return {"accepted": False, "price": counter, "message": f"Please, at least {counter}."}

# -----------------------------
# Negotiation Agent
# -----------------------------
class NegotiationAgent:
    def __init__(self, target_price: int, budget: int):
        self.target_price = target_price
        self.budget = budget
        self.last_offer = None

    def make_offer(self, seller_response: Dict[str, Any]) -> int:
        if seller_response is None:
            # Start low (anchoring)
            self.last_offer = max(0, self.target_price - 3000)
        elif not seller_response["accepted"]:
            # Counter logic
            seller_price = seller_response["price"]
            new_offer = (seller_price + self.target_price) // 2
            if new_offer > self.budget:
                new_offer = self.budget
            self.last_offer = new_offer
        return self.last_offer

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Negotiation Agent", page_icon="🤖", layout="centered")

st.title("🤖 AI Negotiation Agent for Online Marketplaces")
st.markdown("This demo shows how an AI agent can negotiate with a seller (simulated).")

# Inputs
col1, col2 = st.columns(2)
with col1:
    listing_price = st.number_input("Seller Listing Price (₹)", value=20000, step=500)
    target_price = st.number_input("Your Target Price (₹)", value=15000, step=500)
with col2:
    budget = st.number_input("Your Maximum Budget (₹)", value=16000, step=500)
    persona = st.selectbox("Seller Persona", ["strict", "flexible", "desperate"])

if st.button("Start Negotiation"):
    st.session_state.history = []
    st.session_state.seller = SellerSimulator(persona, listing_price)
    st.session_state.agent = NegotiationAgent(target_price, budget)
    st.session_state.round = 0
    st.success("Negotiation started!")

# Run negotiation rounds
if "history" in st.session_state:
    seller = st.session_state.seller
    agent = st.session_state.agent

    # Interactive chat-like UI
    for round_idx in range(6):
        st.session_state.round += 1

        # Agent makes an offer
        if round_idx == 0:
            offer = agent.make_offer(None)
        else:
            offer = agent.make_offer(seller_response)

        with st.chat_message("assistant"):
            st.write(f"My offer is ₹{offer}")

        # Seller responds
        seller_response = seller.respond(offer)
        with st.chat_message("user"):
            st.write(f"Seller: {seller_response['message']} (Counter: ₹{seller_response['price']})")

        if seller_response["accepted"]:
            with st.chat_message("assistant"):
                st.success(f"✅ Deal closed at ₹{seller_response['price']} after {round_idx+1} rounds.")
            break
    else:
        with st.chat_message("assistant"):
            st.warning("⚠️ Negotiation ended without a deal.")