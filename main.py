import streamlit as st
import streamlit as st
from dotenv import load_dotenv
import pickle
import os
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# Hide Hamburger menu and Streamlit footer watermark
hide_menu_style ='''
<style>
MainMenu {visibility: hidden; }
footer{visibility: hidden;}
</style>
'''
st.markdown (hide_menu_style, unsafe_allow_html=True)

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 AI powered App...')
    add_vertical_space(1)
    st.write('Made with ❤️ by LangLabs')
    add_vertical_space(1)
    OPENAI_API_KEY = st.text_input("Enter OpenAI API Key:", placeholder="sk-ndcyu3ye2...")
if OPENAI_API_KEY=="":
    st.sidebar.error("Please enter your OpenAI API Key")        
#load_dotenv()
#def main():
else:
    try:
        llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
        st.header("Chat with PDF 💬")
        pdf = st.file_uploader("Upload your PDF", type='pdf')
        
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = text_splitter.split_text(text=text)
                        
            # embeddings
            file_name = pdf.name[:-4]
            # st.write(f'{file_name}')
        
            if os.path.exists(f"{file_name}.pkl"):
                with open(f"{file_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
                st.write('Embeddings Loaded from the Disk')
            else:
                st.spinner('your file is being processed...')
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{file_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)
            st.write('Type exit to end this session.')
            while True:
                # Accept user questions/query              
                query = st.text_input("**Ask questions about your PDF file:**")
                if query:
                    docs = VectorStore.similarity_search(query=query, k=3)
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=query)
                        print(cb)
                    st.write(response)
        else:
            st.write('Please upload a PDF file')
    except:
        pass

#if __name__ == '__main__':
#    main()
