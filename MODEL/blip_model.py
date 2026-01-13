import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class SimpleViT(nn.Module):
    """Lightweight Vision Transformer for image encoding"""
    def __init__(self, image_size=224, patch_size=16, hidden_size=256, num_layers=4, num_heads=8):
        super(SimpleViT, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Linear((patch_size ** 2) * 3, hidden_size)
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=512,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hidden_size = hidden_size
    
    def forward(self, pixel_values):
        # Simulate patch embedding and encoding
        batch_size = pixel_values.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.randn(batch_size, 196, self.hidden_size)  # 14x14 patches
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embedding
        x = self.encoder(x)
        return x

class BLIPModel(nn.Module):
    """BLIP: Bootstrapping Language-Image Pre-training"""
    def __init__(self, vocab_size=30522, hidden_size=256, num_hidden_layers=4, num_attention_heads=8):
        super(BLIPModel, self).__init__()
        
        # Vision encoder
        self.vision_encoder = SimpleViT(hidden_size=hidden_size, num_layers=num_hidden_layers)
        
        # Text encoder using simplified BERT
        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=512,
        )
        self.text_encoder = BertModel(bert_config)
        
        # Vision-Text projection layers
        self.vision_proj = nn.Linear(hidden_size, hidden_size)
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        
        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(
            hidden_size, num_attention_heads, batch_first=True
        )
        
        # Decoder for caption generation
        self.decoder = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=512,
            batch_first=True
        )
        
        # Output projection for token prediction
        self.output_projection = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids=None, pixel_values=None, attention_mask=None):
        # Process image through vision encoder
        if pixel_values is not None:
            vision_hidden_states = self.vision_encoder(pixel_values)
            vision_embedding = self.vision_proj(vision_hidden_states)
        else:
            vision_embedding = None
        
        # Process text through text encoder
        if input_ids is not None:
            text_outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
            text_hidden_states = text_outputs.last_hidden_state
            text_embedding = self.text_proj(text_hidden_states)
        else:
            text_embedding = None
        
        # Cross-modal alignment (if both modalities are present)
        if vision_embedding is not None and text_embedding is not None:
            attn_output, _ = self.cross_modal_attention(
                text_embedding, vision_embedding, vision_embedding
            )
            combined_embedding = attn_output + text_embedding
        elif text_embedding is not None:
            combined_embedding = text_embedding
        else:
            combined_embedding = vision_embedding
        
        # Generate logits
        logits = self.output_projection(combined_embedding)
        return logits

# Example usage
if __name__ == "__main__":
    model = BLIPModel()
    print("BLIP Model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
