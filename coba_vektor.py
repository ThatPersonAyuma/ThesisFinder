from src.vsm.cleaning.tokenizer import Tokenizer

tok = Tokenizer()

teks = "Graph neural networks learn from graphs; training graph models"
v = tok.vectorize(teks)

print("panjang vektor :", v.shape)
print("jumlah kata unik di vocab:", len(tok))
print("isi yang tidak nol:")
for i in v.nonzero()[0]:
    print(f"  posisi {i:>6} | {tok.terms[i]:<10} | muncul {v[i]}x")

sv = tok.vectorize(teks, sparse=True)
print("bentuk sparse  :", sv.shape, "| non-nol:", sv.nnz)