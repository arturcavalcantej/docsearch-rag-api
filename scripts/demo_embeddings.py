"""
Demonstração prática de Vector Embeddings e Busca Semântica.

Execute: python scripts/demo_embeddings.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Carrega o mesmo modelo usado na API
print("Carregando modelo de embeddings...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print("Modelo carregado!\n")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calcula similaridade cosseno entre dois vetores."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed(text: str) -> list[float]:
    """Gera embedding para um texto."""
    return model.encode(text, normalize_embeddings=True).tolist()



# ============================================================
# DEMONSTRAÇÃO 1: Similaridade Semântica
# ============================================================
print("=" * 60)
print("DEMO 1: SIMILARIDADE SEMÂNTICA")
print("=" * 60)
print()

# Textos sobre o mesmo assunto (recuperar senha)
textos_senha = [
    "Como recuperar minha senha?",
    "Esqueci minha password, como faço?",
    "Preciso resetar meu acesso",
    "Não lembro a senha do sistema",
    "How to reset my password?",  # Em inglês!
]

# Textos sobre outro assunto (pagamento)
textos_pagamento = [
    "Quais formas de pagamento vocês aceitam?",
    "Posso pagar com PIX?",
    "Aceita cartão de crédito?",
]

# Gera embeddings
print("Gerando embeddings para frases sobre SENHA:")
embeddings_senha = [embed(t) for t in textos_senha]
for t in textos_senha:
    print(f"  • {t}")

print("\nGerando embeddings para frases sobre PAGAMENTO:")
embeddings_pagamento = [embed(t) for t in textos_pagamento]
for t in textos_pagamento:
    print(f"  • {t}")

# Compara similaridades
print("\n" + "-" * 60)
print("COMPARANDO SIMILARIDADES:")
print("-" * 60)

base = textos_senha[0]
base_emb = embeddings_senha[0]

print(f"\nFrase base: \"{base}\"\n")
print("Similaridade com outras frases de SENHA:")
for t, e in zip(textos_senha[1:], embeddings_senha[1:]):
    sim = cosine_similarity(base_emb, e)
    bar = "█" * int(sim * 30)
    print(f"  {sim:.3f} {bar} \"{t}\"")

print("\nSimilaridade com frases de PAGAMENTO:")
for t, e in zip(textos_pagamento, embeddings_pagamento):
    sim = cosine_similarity(base_emb, e)
    bar = "█" * int(sim * 30)
    print(f"  {sim:.3f} {bar} \"{t}\"")


# ============================================================
# DEMONSTRAÇÃO 2: Busca Semântica vs Keyword
# ============================================================
print("\n")
print("=" * 60)
print("DEMO 2: BUSCA SEMÂNTICA vs KEYWORD")
print("=" * 60)
print()

# Simula chunks de um documento
chunks = [
    "Para recuperar sua senha, acesse a página de login e clique em 'Esqueci minha senha'. Você receberá um email com um link.",
    "Aceitamos cartão de crédito (Visa, Mastercard), boleto bancário e PIX para pagamentos.",
    "O aplicativo está travando? Tente fechar completamente e abrir novamente. Se persistir, limpe o cache.",
    "Utilizamos criptografia AES-256 para proteger seus dados. Nossos servidores seguem a LGPD.",
    "Para cancelar sua assinatura, acesse Configurações > Assinatura > Cancelar. O cancelamento é efetivado ao final do período.",
]

print("CHUNKS DO DOCUMENTO:")
for i, c in enumerate(chunks):
    print(f"  [{i}] {c[:70]}...")

# Gera embeddings dos chunks
chunk_embeddings = [embed(c) for c in chunks]

# Query que NÃO tem palavras exatas
query = "meu app não funciona direito"

print(f"\n{'─' * 60}")
print(f"QUERY: \"{query}\"")
print(f"{'─' * 60}")

# Busca por keyword (simulada)
print("\n🔤 BUSCA POR KEYWORD (palavras exatas):")
keywords = query.lower().split()
for i, c in enumerate(chunks):
    matches = [k for k in keywords if k in c.lower()]
    if matches:
        print(f"  [{i}] Match: {matches}")
    else:
        print(f"  [{i}] Nenhuma palavra encontrada")

# Busca semântica
print("\n🧠 BUSCA SEMÂNTICA (significado):")
query_emb = embed(query)
similarities = [(i, cosine_similarity(query_emb, ce)) for i, ce in enumerate(chunk_embeddings)]
similarities.sort(key=lambda x: x[1], reverse=True)

for i, sim in similarities:
    bar = "█" * int(sim * 40)
    print(f"  [{i}] {sim:.3f} {bar}")
    print(f"       {chunks[i][:60]}...")

print(f"\n✅ RESULTADO: Chunk [{similarities[0][0]}] é o mais relevante!")
print(f"   Mesmo sem palavras em comum, a busca semântica encontrou")
print(f"   o chunk sobre 'app travando' porque entendeu o SIGNIFICADO.")


# ============================================================
# DEMONSTRAÇÃO 3: Multilíngue
# ============================================================
print("\n")
print("=" * 60)
print("DEMO 3: BUSCA MULTILÍNGUE")
print("=" * 60)
print()

chunk_pt = "Como cancelar minha assinatura e receber reembolso do valor pago"
chunk_emb = embed(chunk_pt)

queries_multi = [
    ("Português", "quero cancelar meu plano"),
    ("Inglês", "I want to cancel my subscription"),
    ("Espanhol", "quiero cancelar mi suscripción"),
    ("Francês", "je veux annuler mon abonnement"),
]

print(f"CHUNK: \"{chunk_pt}\"\n")
print("QUERIES em diferentes idiomas:")

for idioma, q in queries_multi:
    q_emb = embed(q)
    sim = cosine_similarity(chunk_emb, q_emb)
    bar = "█" * int(sim * 40)
    print(f"  {idioma:10} {sim:.3f} {bar} \"{q}\"")

print("\n✅ O modelo entende o SIGNIFICADO independente do idioma!")


# ============================================================
# DEMONSTRAÇÃO 4: Visualização do Vetor
# ============================================================
print("\n")
print("=" * 60)
print("DEMO 4: ANATOMIA DE UM EMBEDDING")
print("=" * 60)
print()

texto = "Resetar senha do sistema"
vetor = embed(texto)

print(f"Texto: \"{texto}\"")
print(f"\nDimensões: {len(vetor)}")
print(f"\nPrimeiros 10 valores do vetor:")
print(f"  {vetor[:10]}")
print(f"\nÚltimos 10 valores:")
print(f"  {vetor[-10:]}")
print(f"\nEstatísticas:")
print(f"  Min: {min(vetor):.4f}")
print(f"  Max: {max(vetor):.4f}")
print(f"  Média: {np.mean(vetor):.4f}")
print(f"\n💡 Cada número captura uma 'dimensão' do significado!")
print(f"   Textos similares terão vetores parecidos.")


print("\n")
print("=" * 60)
print("FIM DA DEMONSTRAÇÃO")
print("=" * 60)
print("""
RESUMO:
1. Embeddings transformam texto em vetores numéricos
2. Textos com significado similar ficam próximos no espaço vetorial
3. Busca semântica encontra resultados mesmo sem palavras exatas
4. O modelo é multilíngue - entende significado em vários idiomas
5. Cada vetor tem 384 dimensões que capturam aspectos do significado
""")
