import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

class SmallLanguageModel(nn.Module):
    def __init__(self, vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12):
        super(SmallLanguageModel, self).__init__()
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        )
        self.model = GPT2Model(self.config)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return logits

# Example usage
if __name__ == "__main__":
    model = SmallLanguageModel()
    print("SLM Model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")