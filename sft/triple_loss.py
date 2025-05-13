from typing import Tuple
import torch
import scipy.optimize
import torch.nn.functional as F
from torch import nn
from triple_utils import extract_student_triples
from transformers import Trainer


# Triple 단위 embedding
# def embed_triple(triple: Tuple[str, str, str], tokenizer, model) -> torch.Tensor:
#     triple_text = f"{triple[0]} [PRED] {triple[1]} [OBJ] {triple[2]}"
#     inputs = tokenizer(triple_text, return_tensors='pt').to(model.device)
#     outputs = model(**inputs)
#     # 예시: CLS token or mean pooling
#     embedding = outputs.last_hidden_state.mean(dim=1)  # shape: [1, hidden_dim]
#     return embedding


# 각 요소 단위 embedding
def embed_component(text: str, tokenizer, model) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # 예: 마지막 레이어 mean pooling
        hidden = outputs.hidden_states[-1]  # shape: [1, seq_len, dim]
        emb = hidden.mean(dim=1)  # shape: [1, dim]
        
    return emb.squeeze(0)


def embed_structured_triple(triple: Tuple[str, str, str], tokenizer, model, method='concat'):
    subj_emb = embed_component(triple[0], tokenizer, model)
    pred_emb = embed_component(triple[1], tokenizer, model)
    obj_emb = embed_component(triple[2], tokenizer, model)
    
    if method == 'concat':
        triple_emb = torch.cat([subj_emb, pred_emb, obj_emb], dim=-1)  # shape: [3 * dim]
    elif method == 'mean':
        triple_emb = (subj_emb + pred_emb + obj_emb) / 3
    else:
        raise ValueError(f"Unknown method: {method}")
        
    return triple_emb

def compute_cost_matrix(teacher_embeds, student_embeds):
    # cosine distance (1 - similarity)
    normalized_t = F.normalize(teacher_embeds, dim=-1)
    normalized_s = F.normalize(student_embeds, dim=-1)
    sim = torch.matmul(normalized_t, normalized_s.T)  # shape: [n_teacher, n_student]
    cost = 1 - sim  # distance
    return cost


def hungarian_matching(cost_matrix):
    cost = cost_matrix.detach().cpu().numpy()
    row_idx, col_idx = scipy.optimize.linear_sum_assignment(cost)
    return list(zip(row_idx, col_idx))  # matched index pairs


def semantic_set_loss(teacher_triples, student_triples, tokenizer, model):
    device = next(model.parameters()).device
    
    teacher_embeds = torch.stack([embed_structured_triple(t, tokenizer, model) for t in teacher_triples]).to(device)
    student_embeds = torch.stack([embed_structured_triple(s, tokenizer, model) for s in student_triples]).to(device)
    
    cost_matrix = compute_cost_matrix(teacher_embeds, student_embeds)
    matches = hungarian_matching(cost_matrix)
    print(f"Matches: {matches}")
    # Matching Loss (semantic distance)
    loss = sum(cost_matrix[i, j] for i, j in matches) / max(len(matches), 1)
    
    # Unmatched penalty
    unmatched = (len(teacher_triples) + len(student_triples) - 2 * len(matches))
    penalty = unmatched * 0.1  # λ = 0.1
    print(f"Unmatched penalty: {penalty}")
    
    return loss + penalty


class CustomSFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_hidden_states=True,
        )

        decoded = self.tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)[0]
        student_triples = extract_student_triples(decoded)
        teacher_triples = inputs["labels_triples"]

        loss = semantic_set_loss(teacher_triples, student_triples, self.tokenizer, model)

        return (loss, outputs) if return_outputs else loss