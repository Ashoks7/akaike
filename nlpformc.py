from transformers import T5ForConditionalGeneration, T5Tokenizer
import random
import streamlit as st

question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')

def get_question(sentence, answer, mdl, tknizer):
    text = "context: {} answer: {}".format(sentence, answer)
    max_len = 256
    encoding = tknizer.encode_plus(
        text,
        max_length=max_len,
        pad_to_max_length=False,
        truncation=True,
        return_tensors="pt"
    )

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = mdl.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        early_stopping=True,
        num_beams=5,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        max_length=300
    )

    dec = [tknizer.decode(ids, skip_special_tokens=True) for ids in outs]

    ques = dec[0].replace("question:", "")
    ques = ques.strip()
    return ques

def generate_distractors(answer, num_distractors=3):
    distractors = set()

    # Swap characters within the answer to create distractors
    for _ in range(num_distractors):
        distractor = list(answer)
        idx1, idx2 = random.sample(range(len(answer)), 2)
        distractor[idx1], distractor[idx2] = distractor[idx2], distractor[idx1]
        distractors.add("".join(distractor))

    # If we still need more distractors, add some random characters to the answer
    while len(distractors) < num_distractors:
        distractor = list(answer)
        for _ in range(len(answer)):  # Change all characters
            idx = random.randint(0, len(answer) - 1)
            distractor[idx] = chr(random.randint(48, 57))  # Replace with a random digit character
        distractors.add("".join(distractor))

    # If the correct answer is already in distractors, remove it and generate a new distractor
    while answer in distractors:
        new_distractor = list(answer)
        for _ in range(len(answer)):  # Change all characters
            idx = random.randint(0, len(answer) - 1)
            new_distractor[idx] = chr(random.randint(48, 57))  # Replace with a random digit character
        distractors.remove(answer)
        distractors.add("".join(new_distractor))

    return list(distractors)

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://wallpapers.com/images/hd/question-mark-background-tghys36vjksf02r6.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

st.title("Did you want to create MCQs?")

user_sentence = st.text_input("Enter or paste the para:")

ans = st.text_input("Enter the answers which you want to create MCQs (separated by commo): ")


if st.button("Generate MCQs"):
    context = user_sentence

    inp = ans
    user_list = inp.split(',')
    a = len(user_list)

    for i in range(a):
        answer = user_list[i]

        ques = get_question(context, answer, question_model, question_tokenizer)
        distractors = generate_distractors(answer, num_distractors=3)

        st.write(f"{i + 1}) Question: {ques}")
        st.write("a)", answer, end=" ")
        options = distractors
        random.shuffle(options)
        for idx, option in enumerate(options, start=1):
            st.write(f"{chr(97 + idx)}) {option}", end=" ")
        st.write("\nCorrect answer:", answer)

        i += 1

        #try this code in vscode
        #in new terminal - "python -m streamlit run nlpformc.py"