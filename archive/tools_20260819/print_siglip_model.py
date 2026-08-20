from transformers import AutoProcessor, SiglipVisionModel, SiglipVisionConfig

if __name__ == "__main__":
    path = 'pretrained_models/google/siglip2-base-patch16-512'
    config = SiglipVisionConfig.from_pretrained(path)
    print(config)
    model = SiglipVisionModel.from_pretrained(path)
    print(model)