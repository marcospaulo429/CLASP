import sys

import torch
import torch.nn as nn

from clasp.models.fusion import HubertLabseConcat


def register_pickled_fusion_classes_for_torch_load():
    """Checkpoints from notebooks pickle models as __main__.HubertLabseConcat; re-bind for torch.load."""
    main = sys.modules["__main__"]
    from clasp.models.fusion import HubertLabseGating, Wav2vecConcat

    for cls in (HubertLabseConcat, HubertLabseGating, Wav2vecConcat):
        setattr(main, cls.__name__, cls)


def load_model(model_path, device):
    register_pickled_fusion_classes_for_torch_load()
    try:
        model_to_test = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        model_to_test = torch.load(model_path, map_location=device)
    # Older checkpoints (pre-`mode` field) omit `mode`; forward() still reads self.mode.
    if isinstance(model_to_test, HubertLabseConcat) and not hasattr(model_to_test, "mode"):
        model_to_test.mode = "joint"
    model_to_test = model_to_test.to(device)
    model_to_test.eval()
    return model_to_test


def build_final_embeddings(model, audio_embeddings, image_embeddings):
    with torch.no_grad():
        return model(audio_embeddings, image_embeddings)


def retrieve_top1(text_embeddings, final_embeddings, sample_index, device):
    sample_embedding = text_embeddings[sample_index].unsqueeze(0).to(device)
    cos_sim = nn.CosineSimilarity(dim=1)
    similarities = cos_sim(sample_embedding, final_embeddings)
    most_similar_idx = torch.argmax(similarities).item()
    return most_similar_idx, similarities[most_similar_idx].item()

