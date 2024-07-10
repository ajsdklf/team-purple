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

client = OpenAI()
Settings.llm = OpenAI_llama(model='gpt-4o')

DECIDER = """
You are a decider. Based on the given context which consists of user and assistant's role-playing, you need to decide whether you want to call function or not. The function you can call is 'call_explainer'. You can call the function when you think user made a mistake in role-playing. Followings are some examples of specific situation where you will need to call function:
#################### 
1. if the user uses '반말', which is only used between friends, call the function in order to kindly correct the user to use '존댓말' which is used in formal situations. 
2. If user don't follow appropriate sentence structure, call the function in order to provide detailed explanation about the sentence structure.
####################

When calling a function, be sure that your topic argument is the specific topic that needs to be explained more in detail and the context argument is the specific context related to the topic that needs to be explained more in detail. Also be careful not to just call function everytime. You should only call function when you think user needs additional explanation about the topic's context.
"""

storage_context = StorageContext.from_defaults(persist_dir='../index_example1')
index = load_index_from_storage(storage_context)

retriever = VectorIndexRetriever(
  index=index,
  similarity_top_k=3
)

response_synthesizer = get_response_synthesizer(
  response_mode=ResponseMode.COMPACT
)

query_engine = RetrieverQueryEngine(
  retriever=retriever,
  response_synthesizer=response_synthesizer
)

def call_explainer(context, topic):
  query = f"""
  User is having a conversation with their counterpart in the role-playing. During the conversation, user made a mistake. Followings are the detailed context of conversation, and mistake user made in the conversation: {context}
  
  Given the context, user needs to be provided explanation about {topic}. Provide user with moderate hint to guide them to a right expression. Remember, you should not provide the correct answer directly to the user.
  """
  
  nodes = retriever.retrieve(query)
  response = query_engine.query(query)
  
  return {
      "response": response.response,
      "nodes": [node.node for node in nodes]
  }

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'call_explainer',
            'description': "Provide additional explanation about user's usage of Korean in simulation.",
            'parameters': {
                'type': 'object',
                'properties': {
                    'context': {
                          'type': 'string',
                          'description': 'Description of the context in which you and your counterpart are having the conversation. Context should also include information about utterances of user and counterpart, and the mistake user made in the conversation.'
                        },
                    'topic': {
                        'type': 'string',
                        'description': 'Specific topic that needs to be explained more in detail to guide user to a right expression.'
                    },
                },
                'required': [
                    'context',
                    'topic',
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
     1. **Entering and Ordering:** Greeting the waiter, asking for the menu, asking for recommendations, placing an order with special requests.
     2. **Asking Questions about the Menu:** Inquiring about ingredients, portion sizes, and suggestions based on preferences.
     3. **During the Meal:** Asking for additional items, requesting refills, making complaints or adjustments.
     4. **After Finishing Your Meal:** Complimenting the food, asking for the bill, inquiring about payment methods, tipping customs.
     5. **Special Requests:** Making a reservation, requesting specific tables, asking for special seating arrangements.
     6. **Dealing with Issues:** Reporting problems with food, asking to speak with the manager, handling billing errors.
     7. **Takeout or Delivery:** Ordering food to go, asking about delivery options and times, clarifying packaging or pickup instructions.
  
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
Your role is the counterpart of the user's role-playing. As a first input, you will be given the scenario that user will be using to role-play. As the user's counterpart, you should lead the conversation with the user. Since the user is a foreigner unfamiliar with Korean, they may need additional guidance or hints. Follow the instructions below to fulfill your role:
---
[Instructions]
1. start a conversation based on the role-play situation presented as input. For example, if the situation is a first-time order at a restaurant, start the conversation by asking, "May I help you, sir?" 
2. Name yourself appropriately for the situation.
For example, if the situation is about ordering at a restaurant, you would call yourself waiter and label your utterances as [waiter: 'response'].
3. Provide a description of the situation in English, a description of the assistant's utterances, and an outline of the user's response in order to keep the role-playing as realistic as possible and at an appropriate level of difficulty. 
4. Your response should include the description of the scene, the waiter's utterance, the pronunciation of the waiter's utterance, the meaning of the waiter's utterance in English, and the hint for user's response.
5. When providing a hint, you should guide the user to the right expression without giving the correct answer directly. This is very important, don't just throw correct answer to user.
6. Once the scene is over, value of 'in_progress' in your response should be False, and your value of 'content' in your response should be '<<<ROLE-PLAY ENDED>>>'. 
7. Value of each field in your response must not be JSON object. It must be a string.
---

Value of 'content' in your response must follow the structure of following examples:

---

[Example1] 
Scene: Making an order in Korean restaurant.

[detailed explanation of the scene.]

---

Waiter: "무엇을 주문하시겠어요?" 

Pronounciation : 'mueos-eul jumunhasigess-eoyo'

Meaning in English: What would you like to order?

---

Hint: You should respond with the name of the dish you want to order. Don't forget to use '존댓말'(honorofics) when speaking to the waiter.

[Example2]
Scene: Asking for an additional order in the middle of the meal.

[detailed explanation of the scene.]

---

Waiter: "추가 주문하시겠어요?"

Pronounciation : 'chuga jumunhasigess-eoyo'

Meaning in English: Would you like to order more?

---

Hint: You should respond with the name of the dish you want to order additionally. Don't forget to use '존댓말'(honorofics) when speaking to the waiter.

[Example3]
Scene: Making a payment after a meal.

[detailed explanation of the scene.]

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

############################################
"""

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
      print(response)

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
          response = call_explainer(params_dict['topic'], params_dict['context'])
          explainer_response = response['response']
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
