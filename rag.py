import marimo

__generated_with = "0.9.1"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(r"""#Import Data From Site""")
    return


@app.cell
def __(mo):
    _site = mo.ui.text(
        kind='url',
        value="https://www.anthropic.com/news/contextual-retrieval",
        full_width=True,
    )

    site_form = (
        mo.md("""### Site to extract data from: 

        {site}""")
        .batch(
            site=_site
        )
        .form()
    )

    site_form
    return (site_form,)


@app.cell
def __(mo, site_form):
    mo.stop(
        site_form.value == None,
        mo.md("Choose the Site to Continue.")
    )

    site = site_form.value["site"]
    return (site,)


@app.cell
def __(site):
    import bs4
    from langchain_community.document_loaders import WebBaseLoader

    # filtering out all tags except for p, h1, h2, h3, h4
    bs4_strainer = bs4.SoupStrainer(['p', 'h1', 'h2', 'h3', 'h4'])
    loader = WebBaseLoader(
        web_paths=(site,),
        bs_kwargs={"parse_only": bs4_strainer},
        bs_get_text_kwargs={"separator": "| "}
    )
    docs = loader.load()

    document = docs[0].page_content
    return WebBaseLoader, bs4, bs4_strainer, docs, document, loader


@app.cell
def __(document, mo):
    mo.md(f""" ### Text from the Site:
    \"{document[:1004]} [...]\" """)
    return


@app.cell
def __(mo):
    mo.md("""# Split Document""")
    return


@app.cell
def __(mo):
    _chunk_size = mo.ui.slider(
        start=200,
        stop=1000,
        step=50, 
        value=600,
        label="Chunck Size:",
    )

    _chunk_overlap = mo.ui.slider(
        start=0,
        stop=200,
        step=20, 
        value=0,
        label="Chunck Overlap:",
    )

    chunk_form = (
        mo.md("""### Choose How to Split the Document
        {chunk_size}\n
        {chunk_overlap}
        """)
        .batch(
            chunk_size=_chunk_size,
            chunk_overlap=_chunk_overlap
        )
        .form(show_clear_button=True)
    )

    chunk_form
    return (chunk_form,)


@app.cell
def __(chunk_form, mo):
    mo.stop(
        chunk_form.value == None,
        mo.md("Choose How to Split the Document to Continue.")
    )

    chunk_size = chunk_form.value["chunk_size"]
    chunk_overlap = chunk_form.value["chunk_overlap"]
    return chunk_overlap, chunk_size


@app.cell
def __(chunk_overlap, chunk_size, document):
    from langchain_text_splitters import CharacterTextSplitter

    # separete on \n because of how the page was read
    text_splitter = CharacterTextSplitter(
        separator="|",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.create_documents([document])
    return CharacterTextSplitter, chunks, text_splitter


@app.cell
def __(chunks, mo):
    chunk_number = mo.ui.number(
        start=0,
        stop=len(chunks)-1,
        step=1,
        # label="### Chunk to Show:",
        value=0,
        full_width=False,
    )

    mo.vstack([
       mo.md("### Show Chunk Number:"),
       chunk_number
    ])
    return (chunk_number,)


@app.cell
def __(chunk_number, chunks, mo):
    mo.md(chunks[chunk_number.value].page_content)
    return


@app.cell
def __(mo):
    mo.md("""# Add Context""")
    return


@app.cell
def __():
    from secret_key import openapi_key
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=openapi_key,
    )
    return ChatOpenAI, llm, openapi_key


@app.cell
def __(mo):
    _context_prompt = """<purpose>
    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
    </purpose>
    <document>
    {document}
    </document>
    <chunk>
    Here is the chunk we want to situate within the whole document:
    {chunk}
    </chunk>"""

    context_form = mo.ui.code_editor(
        language="xml",
        value=_context_prompt,
        label="Prompt to Generate Context: ",
    ).form()

    context_form
    return (context_form,)


@app.cell
def __(context_form, mo):
    mo.stop(
        context_form.value == None,
        mo.md("Choose the Prompt to Generate Context to Continue.")
    )

    context_prompt = context_form.value
    return (context_prompt,)


@app.cell
def __(List, chunks, context_prompt, document, llm):
    from langchain.schema import Document
    from langchain.prompts import ChatPromptTemplate

    def contextualize_chunks(chunks: List[Document], context: str) -> List[Document]:
        contextualized_chunks = []

        for chunk in chunks:
            prompt = ChatPromptTemplate.from_template(context)
            messages = prompt.format_messages(document=document, chunk=chunk)
            response = llm.invoke(messages)

            contextualized_chunk = f"{response.content}\n\n{chunk.page_content}"
            contextualized_chunks.append(
                Document(page_content=contextualized_chunk, metadata=chunk.metadata)
            )

        return contextualized_chunks

    contextualized_chunks = contextualize_chunks(chunks, context_prompt)
    return (
        ChatPromptTemplate,
        Document,
        contextualize_chunks,
        contextualized_chunks,
    )


@app.cell
def __(contextualized_chunks, mo):
    ctx_number = mo.ui.number(
        start=0,
        stop=len(contextualized_chunks)-1,
        step=1,
        value=0,
        full_width=False,
    )

    mo.vstack([
       mo.md("### Show Contextualized Chunk Number:"),
       ctx_number
    ])
    return (ctx_number,)


@app.cell
def __(contextualized_chunks, ctx_number, mo):
    mo.md(contextualized_chunks[ctx_number.value].page_content)
    return


@app.cell
def __(mo):
    mo.md("""# Add Embeddings to AstraDB""")
    return


@app.cell
def __(mo):
    _astra_token = mo.ui.text(
        kind='password',
        label = 'Astra Token: '
    )

    _astra_api_endpoint = mo.ui.text(
        kind='password',
        label = 'Astra API Endpoint: '
    )

    astra_form = (
        mo.md("""### AstraDB Info
            {astra_token}\n
            {astra_api_endpoint}"""
        )
        .batch(
            astra_token=_astra_token,
            astra_api_endpoint=_astra_api_endpoint
        )
        .form()
    )

    astra_form
    return (astra_form,)


@app.cell
def __(astra_form, mo):
    mo.stop(
        astra_form.value == None,
        mo.md("Submit AstraDB Info to continue.")
    )

    astra_token = astra_form.value["astra_token"]
    astra_api_endpoint = astra_form.value["astra_api_endpoint"]
    return astra_api_endpoint, astra_token


@app.cell
def __(astra_api_endpoint, astra_token, contextualized_chunks):
    from langchain_astradb import AstraDBVectorStore
    from langchain_openai import OpenAIEmbeddings
    from uuid import uuid4

    def add_embeddings_to_astradb(contextualized_chunks, astra_api_endpoint, astra_token):
        embedding = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        vector_store = AstraDBVectorStore(
            embedding=embedding,
            collection_name="rag",
            api_endpoint=astra_api_endpoint,
            token=astra_token,
        )

        uuids = [str(uuid4()) for _ in range(len(contextualized_chunks))]
        vector_store.add_documents(documents=contextualized_chunks, ids=uuids)

    add_embeddings_to_astradb(
        contextualized_chunks,
        astra_token,
        astra_api_endpoint,
    )
    return (
        AstraDBVectorStore,
        OpenAIEmbeddings,
        add_embeddings_to_astradb,
        uuid4,
    )


if __name__ == "__main__":
    app.run()
