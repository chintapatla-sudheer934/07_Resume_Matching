import streamlit as st
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Resume and Job Description Matcher")
job=st.text_area("Paste job description")
files=st.file_uploader("Upload resume PDFs",type=["pdf"],accept_multiple_files=True)
if st.button("Rank resumes"):
    if not job.strip() or not files:
        st.warning("Enter a job description and upload at least one PDF.")
    else:
        docs=[]
        for f in files:
            pdf=fitz.open(stream=f.read(),filetype="pdf")
            docs.append((f.name," ".join(p.get_text() for p in pdf)))
        corpus=[job]+[x[1] for x in docs]
        mat=TfidfVectorizer(stop_words="english").fit_transform(corpus)
        scores=cosine_similarity(mat[0:1],mat[1:]).flatten()
        ranking=sorted(zip([x[0] for x in docs],scores),key=lambda z:z[1],reverse=True)
        st.dataframe([{"Resume":n,"Similarity":f"{s:.1%}"} for n,s in ranking])
        st.caption("Similarity is a text-matching aid, not a hiring decision.")
