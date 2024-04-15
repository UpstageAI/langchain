from typing import List

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableSerializable

from langchain_upstage import ChatUpstage, UpstageEmbeddings


def test_upstage_rag() -> None:
    """Test simple RAG."""

    model = ChatUpstage()

    # TODO: Do Layout Analysis

    # TODO: Embed each html tag.
    vectorstore = DocArrayInMemorySearch.from_texts(
        ["harrison worked at kensho", "bears like to eat honey"],
        embedding=UpstageEmbeddings(),
    )
    retriever = vectorstore.as_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    retrieved_docs = retriever.get_relevant_documents("What did Harrison do?")

    groundedness_check = GroundednessCheck()
    groundedness = ""
    while groundedness != "grounded":
        chain: RunnableSerializable = (
            RunnablePassthrough() | prompt | model | output_parser
        )

        result = chain.invoke(
            {"context": retrieved_docs, "question": "What did Harrison do?"}
        )

        # convert all Documents to string
        def formatDocumentsAsString(docs: List[Document]) -> str:
            return "\n".join([doc.page_content for doc in docs])

        groundedness = groundedness_check.run(
            {
                "context": formatDocumentsAsString(retrieved_docs),
                "assistant_message": result,
            }
        ).content

    assert groundedness == "grounded"
    assert isinstance(result, str)
    assert len(result) > 0
