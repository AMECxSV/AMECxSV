# SpeechBrain ASV Embeddings

Simple wrapper for SpeechBrain speaker embedding models.

```python
from embeddings.models.speechbrain.asv_embedding import SpeechBrainASVEmbeddings

extractor = SpeechBrainASVEmbeddings()
embedding = extractor.getEmbeddings("ecapa", "datasets/example.wav")
```

Supported aliases:

- `ecapa` -> `speechbrain/spkrec-ecapa-voxceleb`
- `xvector` -> `speechbrain/spkrec-xvect-voxceleb`
- `resnet` -> `speechbrain/spkrec-resnet-voxceleb`

You can also pass any SpeechBrain ASV model id that starts with
`speechbrain/spkrec-`.
