import streamlit as st
from translate import translated_english_to_hindi
def translate_text(input_text):
    # Add your translation logic here
    # translated_text = input_text.upper()  # Example translation: convert input to uppercase
    result = translated_english_to_hindi(input_text)
    return result

def main():
    st.title("English to hindi project")

    input_text = st.text_input("Enter text to translate")
    translated_text = st.empty()

    if st.button("Translate"):
        translated_text.text(translate_text(input_text))

if __name__ == "__main__":
    main()