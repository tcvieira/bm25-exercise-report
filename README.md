# Building a Simple Information Retrieval System using BM25 and GPT-3 and evaluated in the CISI collection

Enrollment exercise fot the course "Deep Learning aplicado a sistemas de buscas", FEEC-Unicamp, first semester of 2023.

## report

A brief [report](report.pdf) describing the implementation details, results, how to test the IR system and how chatGPT helped with the project. The markdown version can be found [here](report.md).

## notebook

Jupyter notebook with all the associated functions and libraries, with the code for the IR system.

[![google colab link](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tcvieira/bm25-exercise-report/blob/main/notebook.ipynb)

## app

We know that one of the measures of a good search engine is its UI and how the user interact with it alongside with the query language expressiveness and the latency of the response.

Thus, we build a poc of a search engine interface for our models using [streamlit](https://streamlit.io/). This poc can be accessed on [streamlit cloud](https://tcvieira-bm25-exercise-report-app-41maio.streamlit.app/) or [hugginface spaces](https://huggingface.co/spaces/tcvieira/bm25-information-retrieval).
