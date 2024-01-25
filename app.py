import google.generativeai as genai
import os
import streamlit as st
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key = gemini_api_key)
model = genai.GenerativeModel('gemini-pro')
pinecone_api_key=os.getenv("PINECONE_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("data")


def query_function(question):
    encoded_query=genai.embed_content(
        model="models/embedding-001",
        content=question,
        task_type="retrieval_query")

    res=index.query(vector=encoded_query["embedding"],top_k=10,include_metadata=True)
    paragraphs=[]
    final_paragraph=""
    for passage in res['matches']:
        final_paragraph+=passage['metadata']['content']+"/n"
    print(final_paragraph)
    response = model.generate_content(
        f"""You are provided with some supporting passages from a book along with a question. Answer in detail the user's question from the supporting passages of the book.
        Strictly provide answer from the book and not from outside sources.

        Question: {question}

        Supporting passages from book:
       
        {final_paragraph}
    
    """)
    print(response.text)
    return response.text

def main():
    
    st.title("The 48 Laws Of Power (Question Answering Bot)")

   
    user_question = st.text_input("Ask your question:")

    if st.button("Get Answer"):
        if user_question:
            with st.spinner('Your answer is loading...'):
                answer = query_function(user_question)
            st.text("Answer:")
            st.write(answer)
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
