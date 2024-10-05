import marimo

__generated_with = "0.9.1"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    from secret_key import openapi_key
    import os
    os.environ["OPENAI_API_KEY"] = openapi_key
    os.environ["USER_AGENT"] = "Mozilla/5.0 (iPhone; CPU iPhone OS 15_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'"
    return openapi_key, os


@app.cell
def __(mo):
    mo.md(r"""#Import Data From Site""")
    return


@app.cell
def __():
    import bs4
    from langchain_community.document_loaders import WebBaseLoader

    # filtering out all tags except for p, h1, h2, h3, h4
    bs4_strainer = bs4.SoupStrainer(['p', 'h1', 'h2', 'h3', 'h4'])
    loader = WebBaseLoader(
        web_paths=("https://www.anthropic.com/news/contextual-retrieval",),
        bs_kwargs={"parse_only": bs4_strainer},
        bs_get_text_kwargs={"separator": "| "}
    )
    docs = loader.load()

    document = docs[0].page_content
    return WebBaseLoader, bs4, bs4_strainer, docs, document, loader


@app.cell
def __(document, mo):
    mo.md(f"""{document}""")
    return


@app.cell
def __(mo):
    mo.md("""# Split Document into Chunks""")
    return


@app.cell
def __(document):
    from langchain_text_splitters import CharacterTextSplitter

    # separete on \n because of how the page was read
    text_splitter = CharacterTextSplitter(
        separator="|",
        chunk_size=600,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.create_documents([document])
    return CharacterTextSplitter, chunks, text_splitter


@app.cell
def __(chunks, mo):
    number = mo.ui.number(
        start=0,
        stop=len(chunks),
        step=1,
        label="## Chunk to Show: ",
        value=0,
    )
    return (number,)


@app.cell
def __(number):
    number
    return


@app.cell
def __(chunks, mo, number):
    mo.md(chunks[number.value].page_content)
    return


@app.cell
def __(mo):
    mo.md("""# Add Context to Chunks""")
    return


@app.cell
def __():
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    return ChatOpenAI, ChatPromptTemplate


@app.cell
def __(ChatOpenAI):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    return (llm,)


@app.cell
def __(mo):
    default_context_prompt = """<purpose>
    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
    </purpose>
    <document>
    {document}
    </document>
    <chunk>
    Here is the chunk we want to situate within the whole document:
    {chunk}
    </chunk>"""

    context_prompt = mo.ui.code_editor(
        language="xml",
        value=default_context_prompt,
        label="Context Prompt",
        #full_width=True,
        #rows=10,
    )
    return context_prompt, default_context_prompt


@app.cell
def __(context_prompt):
    context_prompt
    return


@app.cell
def __(ChatPromptTemplate, chunks, document, llm):
    from langchain.schema import Document

    contextualized_chunks = []

    for chunk in chunks:
        prompt = ChatPromptTemplate.from_template(
            """<document> 
            {document}
            </document> 
            Here is the chunk we want to situate within the whole document
            <chunk> 
            {chunk}
            </chunk> 
            Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
        )
        messages = prompt.format_messages(document=document, chunk=chunk)
        response = llm.invoke(messages)
        contextualized_chunk = f"{response.content}\n\n{chunk.page_content}"
        contextualized_chunks.append(Document(page_content=contextualized_chunk, metadata=chunk.metadata))
    return (
        Document,
        chunk,
        contextualized_chunk,
        contextualized_chunks,
        messages,
        prompt,
        response,
    )


@app.cell
def __(contextualized_chunks):
    print(contextualized_chunks[0].page_content)
    return


@app.cell
def __(mo):
    mo.md("""### Add Contextualized Chunks to Another DB""")
    return


@app.cell
def __(AstraDBVectorStore, astra_api_endpoint, astra_token, my_embedding):
    new_store = AstraDBVectorStore(
      embedding=my_embedding,
      collection_name="rag_context",
      api_endpoint=astra_api_endpoint,
      token=astra_token,
    )
    return (new_store,)


@app.cell
def __(contextualized_chunks, new_store, uuid4):
    uuids_1 = [str(uuid4()) for _ in range(len(contextualized_chunks))]
    new_store.add_documents(documents=contextualized_chunks, ids=uuids_1)
    return (uuids_1,)


@app.cell
def __(mo):
    mo.md("""### Try using Rag (Without Context)""")
    return


@app.cell
def __(ChatPromptTemplate, llm):
    from typing import List

    def generate_answer(query: str, relevant_chunks: List[str]) -> str:
        prompt = ChatPromptTemplate.from_template(
            """<purpose>
            1. Based on the given context answer the question.
            2. If the given context is not sufficient to answer the question, say it.
            </purpose>
            <question>
            {query}
            </question>
            <context>
            {chunks}
            </context>"""                       
        )
        messages = prompt.format_messages(query=query, chunks="\n\n".join(relevant_chunks))
        response = llm.invoke(messages)
        return response
    return List, generate_answer


@app.cell
def __(Document, List, chunks):
    from rank_bm25 import BM25Okapi

    #bm 25 for rag
    def create_bm25_index(chunks: List[Document]) -> BM25Okapi:
        tokenized_chunks = [chunk.page_content.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)

    bm25_index = create_bm25_index(chunks)
    return BM25Okapi, bm25_index, create_bm25_index


@app.cell
def __(mo):
    mo.md("""#### Testing""")
    return


@app.cell
def __():
    query = "In traditional RAG it is difficult to retrieve the right information, how does Contextual Retrieval solves that?"
    return (query,)


@app.cell
def __(bm25_index, bm25_results, chunks, query, tokenized_query):
    _tokenized_query = query.split()
    _bm25_results = bm25_index.get_top_n(tokenized_query, chunks, n=1)
    bm25_results_content = [doc.page_content for doc in bm25_results]
    return (bm25_results_content,)


@app.cell
def __(bm25_results_content):
    print(bm25_results_content[0])
    return


@app.cell
def __(bm25_results_content, generate_answer, query):
    response_1 = generate_answer(query=query, relevant_chunks=bm25_results_content)
    return (response_1,)


@app.cell
def __(response_1):
    response_1.content
    return


@app.cell
def __(mo):
    mo.md("""### Using Rag With Context""")
    return


@app.cell
def __(
    bm25_index,
    bm25_results,
    contextualized_chunks,
    query,
    tokenized_query,
):
    _tokenized_query = query.split()
    _bm25_results = bm25_index.get_top_n(tokenized_query, contextualized_chunks, n=1)
    bm25_results_content_1 = [doc.page_content for doc in bm25_results]
    return (bm25_results_content_1,)


@app.cell
def __(bm25_results_content_1):
    bm25_results_content_1[0]
    return


@app.cell
def __(bm25_results_content_1, generate_answer, query):
    response_2 = generate_answer(query=query, relevant_chunks=bm25_results_content_1)
    response_2.content
    return (response_2,)


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md("""# Initializing Vector DB""")
    return


@app.cell
def __():
    from langchain_openai import OpenAIEmbeddings

    my_embedding = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    return OpenAIEmbeddings, my_embedding


@app.cell
def __():
    import getpass

    astra_token = getpass.getpass(prompt="Astra Token: ")
    astra_api_endpoint = getpass.getpass(prompt="Astra Token: ")
    return astra_api_endpoint, astra_token, getpass


@app.cell
def __(astra_api_endpoint, astra_token, my_embedding):
    from langchain_astradb import AstraDBVectorStore

    my_store = AstraDBVectorStore(
      embedding=my_embedding,
      collection_name="rag",
      api_endpoint=astra_api_endpoint,
      token=astra_token,
    )
    return AstraDBVectorStore, my_store


@app.cell
def __(mo):
    mo.md("""### Add Chunks to Vector DB""")
    return


@app.cell
def __(chunks):
    from uuid import uuid4

    # create id for each doc
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    return uuid4, uuids


@app.cell
def __(chunks, my_store, uuids):
    my_store.add_documents(documents=chunks, ids=uuids)
    return


if __name__ == "__main__":
    app.run()
