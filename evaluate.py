from main import (
    build_rag_resources,
)

from dotenv import load_dotenv
import os


TEST_CASES = [
    {
        "query": "What is the monthly rent?",
        "expected_source": "sample_rental_agreement.txt",
    },
    {
        "query": "Who is the tenant?",
        "expected_source": "sample_rental_agreement.txt",
    },
    {
        "query": "What is the security deposit?",
        "expected_source": "sample_rental_agreement.txt",
    },
    {
        "query": "How many days notice is required to terminate the rental agreement?",
        "expected_source": "sample_rental_agreement.txt",
    },
    {
        "query": "What happens if the tenant leaves without notice?",
        "expected_source": "sample_rental_agreement.txt",
    },
    {
        "query": "What is the purpose of the NDA?",
        "expected_source": "sample_nda.txt",
    },
]


def main():
    load_dotenv()

    docs_path = os.getenv("DOCS_PATH", "docs")
    google_api_key = os.getenv("GOOGLE_API_KEY", "")

    print("Building RAG resources...\n")

    retriever, reranker, client = build_rag_resources(
        docs_path,
        google_api_key
    )

    total_tests = len(TEST_CASES)
    passed_tests = 0

    for index, test in enumerate(TEST_CASES, start=1):
        query = test["query"]
        expected_source = test["expected_source"]

        retrieved_docs = retriever.get_relevant_documents(query)

        reranked_docs = reranker.get_relevant_documents(
            query,
            retrieved_docs
        )

        retrieved_sources = [
            os.path.basename(
                doc.metadata.get("source", "")
            )
            for doc in reranked_docs
        ]

        passed = expected_source in retrieved_sources

        if passed:
            passed_tests += 1

        print(f"Test {index}")
        print(f"Query: {query}")
        print(f"Expected source: {expected_source}")
        print(f"Retrieved sources: {retrieved_sources}")
        print(f"Result: {'PASS' if passed else 'FAIL'}")
        print("-" * 60)

    accuracy = (passed_tests / total_tests) * 100

    print("\nEvaluation Summary")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Retrieval Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()