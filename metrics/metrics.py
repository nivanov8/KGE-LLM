def hits_at_k(results, ground_truth, k):
    assert len(results) == len(ground_truth)
    
    hits = 0
    for retrieved, correct in zip(results, ground_truth):
        if correct in retrieved[:k]:
            hits += 1
    return hits / len(results)

def mean_reciprocal_rank(results, ground_truth):
    assert len(results) == len(ground_truth)

    reciprocal_ranks = []
    for retrieved, correct in zip(results, ground_truth):
        try:
            rank = retrieved.index(correct) + 1  # 1-based index
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)  # correct not found in retrieved list

    return sum(reciprocal_ranks) / len(reciprocal_ranks)

def mean_rank(results, ground_truth):
    assert len(results) == len(ground_truth)

    ranks = []
    for retrieved, correct in zip(results, ground_truth):
        try:
            rank = retrieved.index(correct) + 1  # 1-based index
            ranks.append(rank)
        except ValueError:
            ranks.append(0.0)  # correct not found in retrieved list

    return sum(ranks) / len(ranks)

def get_metrics(results, ground_truth):
    return {
        "Hits@1": hits_at_k(results, ground_truth, k=1),
        "Hits@3": hits_at_k(results, ground_truth, k=3),
        "Hits@10": hits_at_k(results, ground_truth, k=10),
        "MRR": mean_reciprocal_rank(results, ground_truth),
        "MR": mean_rank(results, ground_truth)
    }