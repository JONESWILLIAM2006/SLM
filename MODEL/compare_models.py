import torch
import torch.nn as nn
import time
from slm_model import SmallLanguageModel
from blip_model import BLIPModel

def compare_models():
    print("=" * 80)
    print("COMPARING GPT-2 (SLM) VS BLIP MODELS")
    print("=" * 80)
    
    # Initialize models
    print("\n[1] Initializing Models...")
    gpt2_model = SmallLanguageModel()
    blip_model = BLIPModel()
    
    print("✓ Models initialized successfully")
    
    # Compare parameters
    print("\n[2] Parameter Comparison:")
    print("-" * 80)
    gpt2_params = sum(p.numel() for p in gpt2_model.parameters())
    blip_params = sum(p.numel() for p in blip_model.parameters())
    
    print(f"GPT-2 (SLM) Parameters:     {gpt2_params:,}")
    print(f"BLIP Parameters:            {blip_params:,}")
    print(f"Difference:                 {abs(blip_params - gpt2_params):,}")
    
    # Compare model architecture
    print("\n[3] Architecture Comparison:")
    print("-" * 80)
    print("\nGPT-2 (SLM):")
    print("  - Type: Text-only Language Model")
    print("  - Backbone: GPT-2 Transformer")
    print("  - Input: Text tokens")
    print("  - Output: Token logits")
    print("  - Key Components:")
    print("    • Token embeddings")
    print("    • Multi-head self-attention layers (12)")
    print("    • Feed-forward networks")
    print("    • Language modeling head")
    
    print("\nBLIP:")
    print("  - Type: Multimodal Vision-Language Model")
    print("  - Backbone: Vision Transformer (ViT) + BERT")
    print("  - Input: Images + Text tokens")
    print("  - Output: Token logits")
    print("  - Key Components:")
    print("    • Vision Encoder (ViT)")
    print("    • Text Encoder (BERT)")
    print("    • Vision-Text Projection layers")
    print("    • Cross-modal attention")
    print("    • Transformer decoder")
    
    # Compare inference speed
    print("\n[4] Inference Speed Comparison:")
    print("-" * 80)
    
    # Create dummy inputs for GPT-2
    batch_size = 4
    seq_length = 128
    gpt2_input_ids = torch.randint(0, 50257, (batch_size, seq_length))
    
    # Create dummy inputs for BLIP (must be within vocab size)
    blip_input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    
    # Benchmark GPT-2
    gpt2_model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            _ = gpt2_model(gpt2_input_ids)
        gpt2_time = (time.time() - start) / 10
    
    # Benchmark BLIP (text only)
    blip_model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            _ = blip_model(input_ids=blip_input_ids)
        blip_time = (time.time() - start) / 10
    
    print(f"GPT-2 (SLM) - Avg inference time: {gpt2_time*1000:.2f}ms")
    print(f"BLIP - Avg inference time (text only): {blip_time*1000:.2f}ms")
    print(f"Speed ratio (BLIP/GPT-2): {blip_time/gpt2_time:.2f}x")
    
    # Compare capabilities
    print("\n[5] Capability Comparison:")
    print("-" * 80)
    
    comparison_table = {
        "Feature": ["Text Generation", "Vision Understanding", "Image Captioning", "VQA", "Multimodal", "Parameter Efficiency"],
        "GPT-2 (SLM)": ["✓ Excellent", "✗ None", "✗ No", "✗ No", "✗ No", "✓ Small"],
        "BLIP": ["✓ Good", "✓ Excellent", "✓ Yes", "✓ Yes", "✓ Yes", "✗ Large"]
    }
    
    print(f"\n{'Feature':<25} {'GPT-2 (SLM)':<20} {'BLIP':<20}")
    print("-" * 65)
    for i in range(len(comparison_table["Feature"])):
        print(f"{comparison_table['Feature'][i]:<25} {comparison_table['GPT-2 (SLM)'][i]:<20} {comparison_table['BLIP'][i]:<20}")
    
    # Summary
    print("\n[6] Summary & Recommendations:")
    print("-" * 80)
    print("\nGPT-2 (SLM) is better for:")
    print("  • Pure text generation tasks")
    print("  • Lightweight deployments with limited compute")
    print("  • Fine-tuning on text-only datasets")
    print("  • Inference speed and efficiency")
    
    print("\nBLIP is better for:")
    print("  • Multimodal tasks (vision + language)")
    print("  • Image captioning and description")
    print("  • Visual question answering (VQA)")
    print("  • Understanding relationships between images and text")
    print("  • Tasks requiring visual understanding")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    compare_models()
