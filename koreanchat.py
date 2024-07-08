import streamlit as st
from openai import OpenAI
#import json

# OpenAI API key
API_KEY = '-'

client = OpenAI(api_key=API_KEY)

# Set page configuration
st.set_page_config(page_title="Korean Language Learning Chatbot")

st.title("Korean Language Learning Chatbot")
st.write("Practice your Korean through conversation with a friendly chatbot.")
st.write("Situation: You are at a Korean restaurant. You need to order food, ask for recommendations, understand the menu, place your order, and ask for the bill. Engage in a natural conversation based on this situation.")

# Input area for user messages
user_input = st.text_input("You:", key="user_input")

# Display chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "progress" not in st.session_state:
    st.session_state["progress"] = {
        "fluency_level": "beginner",
        "vocabulary_learned": [],
        "areas_of_improvement": [],
        "conversation_ended": False
    }

def update_progress(new_data):
    st.session_state["progress"].update(new_data)

def get_response(messages):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].message.content

if st.button("Send") and not st.session_state["progress"]["conversation_ended"]:
    if user_input:
        # Prepare messages for API call
        system_prompt = (f"""You are a Korean language learning assistant. 
                Your role is to help users practice their Korean by simulating natural conversations. 
                You have two personas, divided with two names: 'speaking_partner' and 'tutor'. 
                The 'speaking_partner' engages in friendly conversation, acting as a Korean waiter in a restaurant, using appropriate vocabulary and grammar based on the user's fluency level.
                Common phrases include greeting the customer, taking the customers to their seats, taking orders, offering recommendations, and providing the bill.
                The 'speaking_partner' should move the conversation naturally from greeting, to seating, to taking orders, to offering recommendations, to providing the bill.
                The 'tutor' provides guidance and corrections subtly when necessary. Apply the concept of [scaffolding] to assist the user. 
                The 'tutor' should flexibly jump in and provide explanations or corrections when the user's message indicates a need for help or when the conversation context becomes too complex.
                As the system, decide when the tutor should intervene based on the user's fluency level and the complexity of their message.
                For example, if the user uses '반말', which is only used between friends, instead of '존댓말', use the 'tutor' to correct the user in a friendly manner.
                If the user makes a grammatical error in their response or is using some phrases wrong, intervene as needed and provide understandable feedback.
                Ensure that your interventions are helpful and do not interrupt the natural flow of the conversation too much.
                Always consider the user's fluency level and reinforce previously learned concepts.
                Track the user's progress and suggest new vocabulary or grammar naturally during conversations."""
        )

        user_context = ("""Situation: You are at a Korean restaurant.
            You need to order food, ask for recommendations, understand the menu, place your order, and ask for the bill.            
            Engage in a natural conversation based on this situation.
        """)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_context},
        ]

        for message in st.session_state["chat_history"]:
            if message.startswith("You:"):
                messages.append({"role": "user", "content": message[4:]})
            else:
                role, content = message.split(": ", 1)
                role_name = "speaking_partner" if role == "Speaking Partner" else "Tutor"
                messages.append({"role": "assistant", "content": content, "name": role_name})

        messages.append({"role": "user", "content": user_input})

        # Get response from speaking partner
        speaking_partner_response = get_response(messages)
        
        # Append speaking partner response to messages
        st.session_state["chat_history"].append(f"You: {user_input}")
        st.session_state["chat_history"].append(f"Speaking Partner: {speaking_partner_response}")
        messages.append({"role": "assistant", "content": speaking_partner_response, "name": "speaking_partner"})

        # Check if tutor intervention is needed
        tutor_prompt = (
            "You are a Korean language learning tutor. "
            "Your role is to assist users in learning Korean by providing guidance and corrections. "
            "You should subtly correct the user's mistakes, explain complex phrases, and introduce new vocabulary or grammar points as needed. "
            "Ensure that your interventions are helpful and do not interrupt the natural flow of the conversation too much. "
            "Always consider the user's fluency level and reinforce previously learned concepts. "
            "Decide if the user needs help based on the previous conversation."
            "The level of politeness doesn't have to be business level. Remember that the conversation is casual, and everyday-life like."
            "Provide guidance or correction if the waiter's response is too complex compared to the user's fluency level or if the user's response is grammatically or contextually wrong. Otherwise, do not interrupt at all."
        )

        messages.append({"role": "system", "content": tutor_prompt})
        tutor_decision = get_response(messages)
        
        # Tutor intervention based on decision
        if "yes" in tutor_decision.lower():
            tutor_response = get_response(messages)
            st.session_state["chat_history"].append(f"Tutor: {tutor_response}")
        
        # Check for end of conversation
        if any(phrase in speaking_partner_response for phrase in ["안녕히 가세요", "수고하세요", "다음에 또 오세요", "감사합니다"]):
            st.session_state["chat_history"].append("Conversation ended. Thank you for practicing!")
            st.session_state["progress"]["conversation_ended"] = True
            st.write("Conversation ended. Thank you for practicing!")


# Display updated chat history
for message in st.session_state["chat_history"]:
    st.write(message)

if not st.session_state["progress"]["conversation_ended"]:
    st.write("Your fluency level is:", st.session_state["progress"]["fluency_level"])
else:
    st.write("Your practice session has ended. Please refresh the page to start a new session.")