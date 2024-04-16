from pathlib import Path
from typing import List

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableSerializable

from langchain_upstage import ChatUpstage, LayoutAnalysis
from langchain_upstage.embeddings import UpstageEmbeddings
from langchain_upstage.tools.groundedness_check import GroundednessCheck

EXAMPLE_PDF_PATH = Path(__file__).parent.parent / "examples/solar.pdf"


def test_upstage_rag() -> None:
    """Test simple RAG."""

    model = ChatUpstage()

    loader = LayoutAnalysis(file_path=EXAMPLE_PDF_PATH, split="element")
    docs = loader.load()

    vectorstore = DocArrayInMemorySearch.from_documents(
        docs, embedding=UpstageEmbeddings()
    )
    retriever = vectorstore.as_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    retrieved_docs = retriever.get_relevant_documents(
        "How many parameters in SOLAR model?"
    )

    groundedness_check = GroundednessCheck()
    groundedness = ""
    while groundedness != "grounded":
        chain: RunnableSerializable = (
            RunnablePassthrough() | prompt | model | output_parser
        )

        result = chain.invoke(
            {
                "context": retrieved_docs,
                "question": "How many parameters in SOLAR model?",
            }
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
