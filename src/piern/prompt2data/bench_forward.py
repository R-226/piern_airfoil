# Benchmark forward-only vs pipeline (base_model + pool + classifier)
import time, statistics, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from transformers import AutoTokenizer, AutoModel
# import both implementations
from src.piern.prompt2data import mlp as mlp_mod
from src.piern.prompt2data import mlp_hidden as mlp_hidden_mod

MODEL_PATH = "./model/Qwen3.5-0.8B"
TRAIN_DATA = "./data/2com/train_data.jsonl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('DEVICE ->', DEVICE)

def prepare_batch(tokenizer, batch_size=8):
    ds = mlp_mod.SeqRouterDataset(TRAIN_DATA, tokenizer)
    # get first sample and repeat
    input_ids, attention_mask, _ = ds[0]
    input_ids = input_ids.unsqueeze(0).repeat(batch_size, 1)
    attention_mask = attention_mask.unsqueeze(0).repeat(batch_size, 1)
    return input_ids, attention_mask


def bench_classifier_forward(router, input_ids, attention_mask, iters=10, warmup=2):
    router.eval()
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    with torch.no_grad():
        for _ in range(warmup):
            _ = router(input_ids, attention_mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.time()
            _ = router(input_ids, attention_mask)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t1 = time.time()
            times.append(t1 - t0)
    return statistics.median(times), statistics.mean(times)


def bench_pipeline(base_model, router_hidden, input_ids, attention_mask, iters=10, warmup=2):
    base_model.eval(); router_hidden.eval()
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    with torch.no_grad():
        for _ in range(warmup):
            pooled = mlp_hidden_mod.encode_batch(base_model, input_ids, attention_mask)
            _ = router_hidden(pooled)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.time()
            pooled = mlp_hidden_mod.encode_batch(base_model, input_ids, attention_mask)
            _ = router_hidden(pooled)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t1 = time.time()
            times.append(t1 - t0)
    return statistics.median(times), statistics.mean(times)


def main():
    print('Loading tokenizer and base model (may take time)')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
    embedding_dim = base_model.get_input_embeddings().embedding_dim

    # instantiate routers
    mlp_router = mlp_mod.SeqRouter(base_model, embed_dim=embedding_dim, hidden_dim1=512, hidden_dim2=256, hidden_dim3=128, hidden_dim4=64, hidden_dim5=32).to(DEVICE)
    hidden_router = mlp_hidden_mod.SeqRouter(hidden_dim=embedding_dim, hidden_dim1=512, hidden_dim2=256, hidden_dim3=128, hidden_dim4=64, hidden_dim5=32).to(DEVICE)

    print('Preparing a small batch...')
    input_ids, attention_mask = prepare_batch(tokenizer, batch_size=4)

    print('Benchmark: classifier-only (embedding -> MLP)')
    med_b, mean_b = bench_classifier_forward(mlp_router, input_ids, attention_mask, iters=10, warmup=2)
    print(f'classifier-only per-batch median: {med_b:.6f}s, mean: {mean_b:.6f}s, per-sample median: {med_b/4:.6f}s')

    print('Benchmark: pipeline (base_model forward + pool + MLP)')
    med_p, mean_p = bench_pipeline(base_model, hidden_router, input_ids, attention_mask, iters=5, warmup=1)
    print(f'pipeline per-batch median: {med_p:.6f}s, mean: {mean_p:.6f}s, per-sample median: {med_p/4:.6f}s')

if __name__ == '__main__':
    main()
