import streamlit as st
import openai
import random

# OpenAI API 키 설정
openai.api_key = 'api-key'

# 대화 상황 설정
situations = [
    {"situation": "당신은 카페에서 커피를 주문하고 있습니다.", "role": "바리스타", "initial_prompt": "안녕하세요! 주문하시겠어요?", "formality": "존댓말"},
    {"situation": "오랜만에 친구를 만났습니다.", "role": "친구", "initial_prompt": "안녕! 오랜만이다! 그동안 어떻게 지냈어?", "formality": "반말"}
]

# 상황 랜덤 선택
def get_random_situation():
    return random.choice(situations)

# AI 응답 생성
def get_ai_response(messages, formality):
    if formality == "반말":
        messages.insert(0, {"role": "system", "content": "You are a close friend that always uses casual language in Korean."})
    else:
        messages.insert(0, {"role": "system", "content": "You are a cafe worker that always uses formal language in Korean."})

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].message['content'].strip()

# 피드백 제공 함수
def give_feedback(user_input):
    feedback_prompt = "Provide feedback on only the grammatical errors in the following Korean sentence. Do not suggest more polite forms of expression."

    feedback_response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": feedback_prompt},
            {"role": "user", "content": f"사용자의 문장: {user_input}\n어떤 문법적인 오류가 있는지 피드백을 주세요."}
        ],
        max_tokens=150,
        temperature=0.7
    )
    feedback = feedback_response.choices[0].message['content'].strip()
    if "문법적으로 올바른 문장입니다" in feedback or "문법적으로 오류가 없습니다" or "문법적인 오류가 없습니다" in feedback:
        return None
    feedback += " 이를 바탕으로 다시 한 번 말해보세요!"
    return feedback

# Streamlit 앱
def main():
    st.title("한국어 학습 챗봇")

    if 'situation' not in st.session_state:
        st.session_state.situation = get_random_situation()
        st.session_state.feedback_count = 0
        st.session_state.previous_feedback = None
        st.session_state.messages = [
            {"role": "user", "content": st.session_state.situation['initial_prompt']}
        ]

    situation = st.session_state.situation
    st.write(f"상황: {situation['situation']} ({situation['formality']})")
    st.write(f"AI ({situation['role']}): {situation['initial_prompt']}")

    # 사용자 입력
    user_input = st.text_input("당신의 응답:", key="user_input")


    while(user_input):
        # 메시지 업데이트
        st.session_state.messages.append({"role": "user", "content": user_input})

        # AI 응답 생성
        ai_response = get_ai_response(st.session_state.messages, situation['formality'])
        st.write(f"AI ({situation['role']}): {ai_response}")

        # 피드백 제공
        feedback = give_feedback(user_input)

        if feedback:
            st.session_state.feedback_count += 1
            st.session_state.previous_feedback = feedback  # 현재 피드백을 저장
            st.warning(feedback)
            # 피드백 후 사용자 입력을 받을 수 있도록 새로운 입력란 제공
            new_user_input = st.text_input("피드백을 바탕으로 다시 시도해보세요:", key="retry_input")

            if new_user_input:
                # 새로운 입력에 대해 피드백 제공
                st.session_state.messages.append({"role": "user", "content": new_user_input})
                ai_response = get_ai_response(st.session_state.messages, situation['formality'])
                st.write(f"AI ({situation['role']}): {ai_response}")
                user_input = new_user_input

                feedback = give_feedback(user_input)
        else:
            st.success("잘 했어요!")
            st.session_state.feedback_count = 0
            st.session_state.previous_feedback = None  # 올바른 문장이 입력되면 피드백 초기화
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            # 올바른 문장 입력 후 새 대화를 위한 입력란 추가
            user_input = st.text_input("새로운 대화를 이어나가세요:", key="new_user_input")

    # 피드백 3번 이후 올바른 표현 보기 버튼 생성
    if st.session_state.feedback_count >= 3:
        if st.button("올바른 표현 보기"):
            correct_response = st.session_state.messages[-1]['content']
            st.write(f"올바른 표현: {correct_response}")

if __name__ == "__main__":
    main()