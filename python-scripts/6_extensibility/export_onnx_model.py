import torch
from transformers import AutoModel, CLIPModel

# Configuration
# models: openai/clip-vit-base-patch32, openai/clip-vit-large-patch14, google/siglip-large-patch16-384
model_id = "openai/clip-vit-base-patch32"
is_siglip = "siglip" in model_id
output_path = f"/var/lib/onnx_models/{model_id}/model.onnx"

# 1. Load the appropriate model class
if is_siglip:
    model = AutoModel.from_pretrained(model_id)
else:
    model = CLIPModel.from_pretrained(model_id)

model.eval()


# 2. Define the wrapper
class VisionEncoder(torch.nn.Module):
    def __init__(self, m, use_projection):
        super().__init__()
        self.vision_model = m.vision_model
        self.use_projection = use_projection
        # CLIP needs the separate projection layer; SigLIP integrates it
        if use_projection:
            self.visual_projection = m.visual_projection

    def forward(self, pixel_values):
        out = self.vision_model(pixel_values=pixel_values)
        if self.use_projection:
            return self.visual_projection(out.pooler_output)
        return out.pooler_output


# Instantiate wrapper: SigLIP uses the pooler_output directly
wrapper = VisionEncoder(model, use_projection=(not is_siglip))

# 3. Setup Dummy Input
image_size = model.config.vision_config.image_size
dummy = torch.zeros(1, 3, image_size, image_size)

# 4. Export
print(f"Exporting {model_id} to {output_path}...")

torch.onnx.export(
    wrapper,
    dummy,
    output_path,
    input_names=["pixel_values"],
    output_names=["image_embeds"],
    dynamic_axes={"pixel_values": {0: "batch_size"}},
    opset_version=18,
)

print(f"Done! image_size={image_size}")
