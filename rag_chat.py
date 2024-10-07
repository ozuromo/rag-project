import marimo

__generated_with = "0.9.1"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(r"""# Import Data from AstraDB""")
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
def __(astra_api_endpoint, astra_token, mo):
    from langchain_astradb import AstraDBVectorStore
    from langchain_openai import OpenAIEmbeddings
    from secret_key import openapi_key

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openapi_key,
    )

    vector_store = AstraDBVectorStore(
        collection_name="rag",
        embedding=embedding,
        api_endpoint=astra_api_endpoint,
        token=astra_token,
        autodetect_collection=True,
    )

    mo.md("### AstraDB Loaded!")
    return (
        AstraDBVectorStore,
        OpenAIEmbeddings,
        embedding,
        openapi_key,
        vector_store,
    )


@app.cell
def __(mo):
    mo.md(r"""# Set Default Prompt""")
    return


@app.cell
def __(openapi_key):
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=openapi_key,
    )
    return ChatOpenAI, llm


@app.cell
def __(mo):
    _prompt_template = """<purpose>
    1. Based on the given context answer the question.
    2. If the given context is not sufficient to answer the question, say it.
    </purpose>
    <question>
    {question}
    </question>
    <context>
    {context}
    </context>"""

    prompt_form = mo.ui.code_editor(
        language="xml",
        value=_prompt_template,
        label="Prompt to Generate Response: ",
    ).form()

    prompt_form
    return (prompt_form,)


@app.cell
def __(mo, prompt_form):
    mo.stop(
        prompt_form.value == None,
        mo.md("Choose the Prompt to Continue.")
    )

    prompt_template = prompt_form.value
    return (prompt_template,)


@app.cell
def __(mo):
    mo.md("""# Chat with Site!""")
    return


@app.cell
def __(List, llm, mo, prompt_template, vector_store):
    from langchain.schema import Document
    from langchain.prompts import ChatPromptTemplate

    def generate_answer(question: str, context: List[str]) -> str:
        prompt = ChatPromptTemplate.from_template(prompt_template)
        messages = prompt.format_messages(question=question, context="\n\n".join(context))
        response = llm.invoke(messages)
        return response

    def rag_model(messages, config):
        question = messages[-1].content
        context = vector_store.similarity_search(question, k=3)
        response = generate_answer(question, [doc.page_content for doc in context]).content
        return response

    mo.ui.chat(rag_model)

    # In traditional RAG it is difficult to retrieve the right information, how does Contextual Retrieval solves that?
    return ChatPromptTemplate, Document, generate_answer, rag_model


if __name__ == "__main__":
    app.run()
