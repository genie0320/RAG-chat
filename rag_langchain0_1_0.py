import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain

# Set variables
MODEL = "gpt-3.5-turbo-1106"
direction = "You are my husband. Answer my questions in 3 korean sentences at the most."
user_input = "I want eat pizza today. Could you cook for me?"

# Init objects
llm = ChatOpenAI(model=MODEL, max_tokens=100, temperature=0)


# Simplest sample
llm.invoke("hi")

# add Prompt
prompt = ChatPromptTemplate.from_messages([("system", direction), ("user", "{input}")])

chain = prompt | llm
chain.invoke({"input": user_input})


# add Output parser.
output_parser = StrOutputParser()

chain = prompt | llm | output_parser
chain.invoke({"input": user_input})


# Embedding.
import bs4

loader = WebBaseLoader(
    "https://www.allprodad.com/10-steps-make-life-family-man-happier/"
)
# TODO: bs4.SoupStrainer 연구

docs = loader.load()

splitter = RecursiveCharacterTextSplitter()
documents = splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(documents, embeddings)


# Pick documents related to user question.
template = """Answer me on the provided context:

<context>
{context}
</context>

Question : {input}

"""
prompt = ChatPromptTemplate.from_template(template)
document_chain = create_stuff_documents_chain(llm, prompt)

res = document_chain.invoke(
    {
        "input": user_input,
        # 'context' : [Document(page_content="womans doesn't like picky man.")]
        "context": [],
    }
)


print(f"Before text given : {res}")


# Create retrieval chain
retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

res = retrieval_chain.invoke({"input": user_input})

print(res["answer"])


# conversational retrieval chain.
prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query"),
    ]
)

retriever_chain = create_retriever_chain = create_history_aware_retriever(
    llm, retriever, prompt
)

# Inject chat history
chat_history = [
    HumanMessage(content="what do you want to eat tonight?"),
    AIMessage(
        content="I don't have a specific preference, but I don't want a picky eater."
    ),
    HumanMessage(content="I am so tired today."),
    AIMessage(
        content="Really? Then I can cook for you. I will be home, earlier than other day."
    ),
]

res = retriever_chain.invoke(
    {"chat_history": chat_history, "input": "Thank you. it makes me so happy."}
)


print(res)

# Create retrieval chain
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's question based on the below context:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)

document_chain = create_stuff_documents_chain(llm, prompt)
conversational_retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

res = conversational_retrieval_chain.invoke(
    {"chat_history": [], "input": "What is the weather like today?"}
)

print(res)

chat_history = [
    HumanMessage(content="what do you want to eat tonight?"),
    AIMessage(
        content="I don't have a specific preference, but I don't want a picky eater."
    ),
    HumanMessage(content="I am so tired today."),
    AIMessage(
        content="Really? Then I can cook for you. I will be home, earlier than other day."
    ),
]

res = conversational_retrieval_chain.invoke(
    {"chat_history": chat_history, "input": "What is the weather like today?"}
)

print(res)

# pip install --quiet langchain langchain-openai
# pip install --quiet beautifulsoup4 faiss-cpu
# pip --quiet install beautifulsoup4
