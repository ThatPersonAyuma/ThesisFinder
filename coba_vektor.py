from src.vsm.cleaning.tokenizer import Tokenizer
from src.vsm.searcher import VSMSearcher
from src.vsm.tfidf import INDEX_DIR

tok = Tokenizer()

teks = "Graph neural networks learn from graphs; training graph models"
v = tok.vectorize(teks)

print("panjang vektor BoW :", v.shape)
print("jumlah kata unik di vocab:", len(tok))
print("isi yang tidak nol (BoW):")
for i in v.nonzero()[0]:
    print(f"  posisi {i:>6} | {tok.terms[i]:<10} | muncul {v[i]}x")

sv = tok.vectorize(teks, sparse=True)
print("bentuk sparse BoW  :", sv.shape, "| non-nol:", sv.nnz)

if (INDEX_DIR / "config.json").exists():
    searcher = VSMSearcher()
    tv = searcher.vectorize_query(teks, field="abstract")
    print("\npanjang vektor TF-IDF:", tv.shape, "| L2 norm:", float((tv @ tv) ** 0.5))
    for i in tv.nonzero()[0]:
        print(f"  posisi {i:>6} | {searcher.terms[i]:<10} | bobot {tv[i]:.4f}")
