import streamlit as st 
from openai import OpenAI 
import json 
from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext
from llama_index.core import Settings 
from llama_index.llms.openai import OpenAI as OpenAI_llama
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

client = OpenAI()
model_name = "xlm-roberta-base"
Settings.llm = OpenAI_llama(model='gpt-4o')
Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)

DECIDER = """
You are a decider. Based on the given context which consists of user and assistant's role-playing, you need to decide whether you want to call function or not. The function you can call is 'call_explainer'. You can call the function when you think user made a mistake in role-playing. Followings are some examples of specific situation where you will need to call function:
#################### 
1. If the user uses '반말', which is only used between friends, call the function in order to kindly correct the user to use '존댓말' which is used in formal situations. 
2. If user don't follow appropriate sentence structure, call the function in order to provide detailed explanation about the sentence structure.
####################

Only call the function when user made a significant mistake in role-playing. For example, when ordering at a restaurant, if the user says, “불고기 일인분 주세요”, that's polite enough. You should never call a function that says they should be more polite. Remember, Don't just call function arbitrarily. You should only call function when you think user made a mistake in role-playing.

If you think user made a mistake in role-playing, you can call the function by providing the following parameters:
- mistake: Detailed explanation of the mistake user made in the role-playing. Must be as specific as possible and be written **in English**.
- correct_sentence: Correct sentence that user should have used in the role-playing. Must be written **in Korean**.
- user_sentence: User sentence that contains the mistake. Must be written **in Korean**.

You must provide mistake made by user in the middle of role-playing. You should **NEVER** make up the mistake to call function. You should only call function when you think user made a mistake in role-playing.
""".strip()

storage_context = StorageContext.from_defaults(persist_dir='./example_index')
index = load_index_from_storage(storage_context)

retriever = VectorIndexRetriever(
  index=index,
  similarity_top_k=3
)

def call_explainer(mistake, correct_sentence, user_sentence):
  query = mistake
  
  nodes = retriever.retrieve(query)
  list_of_past_mistakes = []
  for node in nodes:
    past_mistake = node.text
    past_correct_sentence = node.metadata['correct_sentence']
    past_user_sentence = node.metadata['user_sentence']
    list_of_past_mistakes.append(f"""
    Wrong sentence : {past_user_sentence}
    Correct sentence : {past_correct_sentence}
    Detailed explanation of the mistake : {past_mistake}
    """)
  
  mistakes_made_by_user_in_past = '\n'.join(list_of_past_mistakes)
  
  prompt = f"""
  User is an English native speaker who is trying to learn Korean. To do so, user is role-playing with an AI assistant. In the middle of role-playing, user made a mistake in using Korean. The user's mistake is as follows:
  ---
  [A MISTAKE USER MADE **IN CURRENT ROLE-PLAYING**]
  Wrong sentence : {user_sentence}
  Correct sentence : {correct_sentence}
  Detailed explanation of the mistake : {mistake}
  ---
  
  Some of the similar mistakes that user made in the past are as follows:
  ---
  [MISTAKES USER MADE **IN PAST**] : {mistakes_made_by_user_in_past}
  ---
  
  Taking into account the mistakes user made in the past, provide additional explanation about user's inappropriate usage of Korean in simulation. You have to try your best to **SCAFFOLD** user which means to provide just enough help to allow user to continue role-playing without giving the correct answer directly. To sum it up, you must follow the following instructions:
  [INSTRUCTIONS]
  1. Must **SCAFFOLD** user by providing just enough help to allow user to continue role-playing without giving the correct answer directly.
  2. Must take into account the mistakes user made in the past to connect past learning experiences with the current learning experience.
  3. It's true that connecting past learning experiences with the current learning experience is important, however, You MUST remember that past mistakes are only for reference. Your focus should rather be on the current mistake user made.
  4. Your response must be written **in English**.
  """
  
  hint = client.chat.completions.create(
    model='gpt-4o',
    messages=[
      {'role': 'system', 'content': prompt}
    ],
  ).choices[0].message.content
  
  print(f"User's mistake: {user_sentence}")
  print(f"Correct sentence: {correct_sentence}")
  print(f"Explanation of the mistake: {mistake}")
  print(f"Past mistakes: {mistakes_made_by_user_in_past}")
  print(f"Hint: {hint}")
  
  return hint

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'call_explainer',
            'description': "Provide additional explanation about user's inappropriate usage of Korean in simulation.",
            'parameters': {
                'type': 'object',
                'properties': {
                    'mistake': {
                          'type': 'string',
                          'description': 'Detailed explanation of the mistake user made in the role-playing. Must be as specific as possible and be written **in English**.'
                        },
                    'correct_sentence': {
                        'type': 'string',
                        'description': 'Correct sentence that user should have used in the role-playing. Must be written **in Korean**.'
                    },
                    'user_sentence': {
                        'type': 'string',
                        'description': 'User sentence that contains the mistake. Must be written **in Korean**.'
                    }
                },
                'required': [
                    'mistake',
                    'correct_sentence',
                    'user_sentence'
                ]
            }
        }
    },
]

SCENARIO_WRITER = """
Your role as an assistant is to create an appropriate scenario to help a user who wants to learn Korean. You have to create scenario of user visiting restaurant. Followings are the Steps you need to follow.


[STEP1: Gather Information] 
In this step, you have to gather information from user to tailor the learning scenario for them. Followings are things you should do and don't do in this step.
---
**Do:**
1. Ask about the learner’s experience level in Korean.
  - "What is your experience level in Korean? Are you a beginner, intermediate, or advanced learner? This helps me tailor the scenario for you."

2. Ask about the learner’s favorite Korean food.
  - "Great! What is your favorite Korean food? This will help make the scenario more engaging for you."
---

---
**Don't:**
- Ask more than one question at a time.
- Mention the steps during your interaction with the user.
---

[STEP2: Set up the role play]
In this step, you have to design learner scenario choices and ask the user to select one out of them. Followings are things you should do and don't do in this step.
---
**Do:**
1. Suggest seven types of restaurant scenarios based on the learner’s information and ask them to pick one.
  - "Here are some restaurant scenarios you can choose from:
    1. **Ordering 엽기떡볶이 for the First Time:** Asking about spiciness levels, main ingredients, sauces, and side dishes.
    2. **Adjusting the Spiciness Level:** Inquiring about different spiciness levels and confirming preferences.
    3. **Adding Toppings to 엽기떡볶이:** Asking about and adding various toppings to the dish.
    4. **Ordering Set Menu at 엽기떡볶이:** Choosing a set menu and confirming its components.
    5. **Ordering 엽기떡볶이 To-Go:** Asking about packaging, delivery options, and times.
    6. **Discussing Spiciness with Friends:** Negotiating spiciness levels and menu choices with friends.
  
    Please type the number corresponding to the scenario you want to practice."
---
---
**Don't:**
- Ask more than one question at a time.
- Overcomplicate the scenario.
- Mention the steps during your interaction with the user.
---

[STEP3: Set up the Scene]
In this step, you have to construct a role-play scenario based on the user's choice. Followings are things you should do in this step.
---
**Do:**
1. Once the learner chooses the type of scenario, construct a role-play scenario based on it.
2. Once you reach STEP3, value of 'in_progress' in your response should be False, and your value of 'content' in your response should be '<<<GENRATING ROLE-PLAY>>>'. Also, you should start the role-play with the user. To do so, your value of 'scenario' should look something like the following:
###
"BEGIN ROLE PLAY: Based on your choice, here is the scenario setup. [Set up of the Scenario.] 

You are a foreigner visiting Korea for the first time. You are at a restaurant and want to order some food. The waiter approaches you and asks...

Waiter: "무엇을 주문하시겠어요?" 

Pronounciation : 'mueos-eul jumunhasigess-eoyo'


Meaning in English: What would you like to order?
###
---

Your response must follow the JSON structure that looks like this:
{
  'in_progress': '[True if you are in the middle of creating a scenario, False if you are done creating a scenario]',
  'content': '[Your response to the user]',
  'scenario': '[The scenario created for the user. This field should be created only when you are done creating a scenario.]'
}
"""

ROLE_PLAYER = """
Your role is the counterpart of the user's role-playing. As a first input, you will be given the scenario that user will be using to role-play. As the user's counterpart, you should lead the conversation with the user. 

Followings are GOAL, PERSONA, NARRATIVE you should follow in this role-play:
---
### **GOAL:** 
This is a role-playing scenario in which the user (learner) practices and learns **how to understand and use vocabulary related to different levels of spiciness in Korean** through interactive dialogues and receives **scaffolded** feedback on their practice.

### **PERSONA:**
In this scenario, you play AI-Tutor, a friendly and practical language tutor.

### **NARRATIVE:**
The learner is introduced to AI-Tutor, is asked initial questions that guide the scenario setup, plays through the dialogue, and gets feedback following the interaction. The interaction leverages the scaffolding method of language learning and incorporates the Korean romanization for better understanding.
---


Since the user is a foreigner unfamiliar with Korean, they may need additional guidance or hints. Follow the instructions below to fulfill your role:
---
### **INSTRUCTIONS:**
1. start a conversation based on the role-play situation presented as input. For example, if the situation is a first-time order at a restaurant, start the conversation by asking, "May I help you, sir?" 
2. Name yourself appropriately for the situation.
For example, if the situation is about ordering at a restaurant, you would call yourself waiter and label your utterances as [waiter: 'response'].
3. Provide a description of the situation in English, a description of the assistant's utterances, and an outline of the user's response in order to keep the role-playing as realistic as possible and at an appropriate level of difficulty. 
4. Your response should include the description of the scene, the waiter's utterance, the pronunciation of the waiter's utterance, the meaning of the waiter's utterance in English, and the hint for user's response.
5. When providing a hint, you should guide the user to the right expression without giving the correct answer directly. This is very important, don't just throw correct answer to user.
6. Once the scene is over, value of 'in_progress' in your response should be False, and your value of 'content' in your response should be '<<<ROLE-PLAY ENDED>>>'. 
7. Value of each field in your response must not be JSON object. It must be a string.
---

**Include the following confirmations in the role play:**

1. the amount of 엽기떡볶이 serves 3-4 people, if you are ordering for 2 people, please mention the 2 person menu with lesser amount.
2. You need to receive information about the spiciness. The spiciness levels are as follows:
  - Mild: It's not spicier than Shin Ramyun, do you mind?
  - Beginner: Spicier than Shin Ramyun, do you mind?
  - Less spicy: Spicier than buldak fried noodles, do you like it?
  - Original: 5 times spicier than buldak fried noodles, do you like it?
  - Spicy: 10 times spicier than the buldak fried noodles, do you like it?
3. Choose the main ingredient: rice cake, oden, half-and-half, or denominator.
4. Sauce type: Ikki (original), Rose (less spicy), Jajang, Mara.
5. Remind diners what is included when ordering a set menu.
6. Beverage and pickled radish options available: CoolPeace, cider, cola.
7. Mention the availability of additional toppings: cheese, bacon, quail egg, vermicelli, etc.

Value of 'content' in your response must follow the structure of following examples:

---

[Example1] 
Summary : User is a foreigner visiting Korea for the first time. User is at a restaurant and wants to order some food. The waiter approaches user and asks...
---


Waiter: "무엇을 주문하시겠어요?" 


Pronounciation : 'mueos-eul jumunhasigess-eoyo'


Meaning in English: What would you like to order?


---


Hint: You should respond with the name of the dish you want to order. Don't forget to use '존댓말'(honorofics) when speaking to the waiter.


[Example2]
Summary: Asking for an additional order in the middle of the meal.

---


Waiter: "추가 주문하시겠어요?"


Pronounciation : 'chuga jumunhasigess-eoyo'


Meaning in English: Would you like to order more?


---


Hint: You should respond with the name of the dish you want to order additionally. Don't forget to use '존댓말'(honorofics) when speaking to the waiter.

[Example3]
Summary: Making a payment after a meal.

---


Waiter: "음식은 괜찮으셨나요?"


Pronounciation : 'eumsig-eun gwaenchanh-eusyeoss-eossnayo'


Meaning in English: Did you enjoy your meal?


---


Hint: You should respond with [Yes, it was delicious.]. Don't forget to use '존댓말'(honorofics) when speaking to the waiter.

---

############################################

Your response must follow the JSON structure that looks like this:
{
  'in_progress': '[True if you are in the middle of the role-play, False if you are done with the role-play]',
  'content': '[Your response to the user]',
}

Keep in mind that content should follow the structure of the examples provided above. Summary must be a **SUMMARY**. You should not just copy and paste the past roleplaying converations. You have to summarize them very shortly and provide the hint for the next role-playing.

############################################
""".strip()

st.header("Role Play Development 1")
if 'messages' not in st.session_state:
  st.session_state.messages = []
  
if 'feedback' not in st.session_state:
  st.session_state.feedback = []

if 'messages_create' not in st.session_state:
  st.session_state.messages_create = []

if 'messages_roleplay' not in st.session_state:
  st.session_state.messages_roleplay = []

if 'progress_roleplay' not in st.session_state:
  st.session_state.progress_roleplay = False
  
if 'progress_create' not in st.session_state:
  st.session_state.progress_create = False

if 'initialize' not in st.session_state:
  st.session_state.initialize = False

if 'summarize' not in st.session_state:
  st.session_state.summarize = False

def starter():
  st.session_state.progress_create = True
  st.session_state.initialize = True

st.button('Try Roleplaying', on_click=starter, key='try_roleplaying')
if not st.session_state.initialize:
  first_message = client.chat.completions.create(
  model='gpt-4o',
  messages=[
    {'role': 'system', 'content': SCENARIO_WRITER,}
  ],
  response_format={'type': 'json_object'}
  ).choices[0].message.content

  first_message = json.loads(first_message)

  st.session_state.messages_create.append({'role': 'assistant', 'content': first_message['content']})
  st.session_state.messages.append({'role': 'assistant', 'content': first_message['content']})

if st.session_state.initialize:
  for message in st.session_state.messages:
    with st.chat_message(message['role']):
      st.markdown(message['content'])
  if st.session_state.progress_create:
    user_input = st.chat_input('You: ', key='user_input_1')
    st.write('Current Status: Creating Scenario')
    if user_input:
      st.session_state.messages_create.append({'role': 'user', 'content': user_input})
      st.session_state.messages.append({'role': 'user', 'content': user_input})
      with st.chat_message('user'):
        st.write(user_input)
      response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{'role': 'system', 'content': SCENARIO_WRITER}] + [msg for msg in st.session_state.messages_create],
        response_format={'type': 'json_object'}
      ).choices[0].message.content
      response = json.loads(response)
      with st.chat_message('assistant'):
        st.markdown(response['content'])
        if not response['in_progress']:
            st.markdown(response['scenario'])
        st.markdown(response['in_progress'])
        
      st.session_state.messages_create.append({'role': 'assistant', 'content': response['content']})
      st.session_state.messages.append({'role': 'assistant', 'content': response['content']})

      if response['in_progress'] == False:
        st.session_state.progress_create = False
        st.session_state.progress_roleplay = True
        scenario = response['scenario']
        st.session_state.messages_roleplay.append({'role': 'assistant', 'content': scenario})
        st.session_state.messages.append({'role': 'assistant', 'content': scenario})
  
  if st.session_state.progress_roleplay:
    user_prompt = st.chat_input('You: ', key='user_prompt_2')
    if user_prompt:
      with st.chat_message('user'):
        st.write(user_prompt)
      st.session_state.messages_roleplay.append({'role': 'user', 'content': user_prompt})
      st.session_state.messages.append({'role': 'user', 'content': user_prompt})
      st.session_state.feedback.append({'role': 'user', 'content': user_prompt})
      
      decision = client.chat.completions.create(
        model='gpt-4o',
        messages=[{'role': 'system', 'content': DECIDER}] + [msg for msg in st.session_state.feedback],
        tools=tools,
        tool_choice='auto',
      ).choices[0].message.tool_calls
      
      if decision == None:
        response = client.chat.completions.create(
          model='gpt-4o',
          messages=[{'role': 'system', 'content': ROLE_PLAYER}] + [msg for msg in st.session_state.messages_roleplay],
          response_format={'type': 'json_object'}
        ).choices[0].message.content
        response = json.loads(response)
        with st.chat_message('assistant'):
          st.markdown(response['content'])
        st.session_state.messages_roleplay.append({'role': 'assistant', 'content': response['content']})
        st.session_state.messages.append({'role': 'assistant', 'content': response['content']})
        st.session_state.feedback.append({'role': 'assistant', 'content': response['content']})
        if not response['in_progress']:
          st.session_state.progress_roleplay = False
          st.session_state.summarize = True
          with st.chat_message('assistant'):
            st.write('If you want the summarization of role-playing, type "summarize".')
      else:
        if decision[0].function.name == 'call_explainer':
          parameters = decision[0].function.arguments
          params_dict = json.loads(parameters)
          response = call_explainer(params_dict['mistake'], params_dict['correct_sentence'], params_dict['user_sentence'])
          explainer_response = response
          with st.chat_message('assistant'):
            st.markdown(explainer_response)
          
          st.session_state.feedback.append({'role': 'assistant', 'content': explainer_response})
          st.session_state.messages.append({'role': 'assistant', 'content': explainer_response})
  if st.session_state.summarize:
    def summarizer():
      summary = client.chat.completions.create(
        model='gpt-4o',
        messages=[{'role': 'system', 'content': 'summarize'}] + [msg for msg in st.session_state.messages],
      )
      # feed_summary
      # dialogue_summary
    st.button('summarize', on_click=summarizer)
