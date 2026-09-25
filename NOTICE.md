# Notices

Verboa Image is a fine-tune of, and is distributed with, components released by others under the Apache License,
Version 2.0. Those components keep their own license; the Verboa weights are licensed by Verboa AI (verboa.io) under
the CreativeML Open RAIL++-M License with one additional use restriction, in LICENSE.md.

| component | author | license | role |
|---|---|---|---|
| ERNIE-Image (transformer, pipeline), revision 5346b31d | Baidu | Apache-2.0 | the base model Verboa's transformer was fine-tuned from |
| ERNIE-Image-Turbo | Baidu | Apache-2.0 | source of the distillation delta applied in Verboa Image Flash |
| Ministral 3 3B | Mistral AI | Apache-2.0 | text encoder, unmodified |
| FLUX.2 VAE | Black Forest Labs | Apache-2.0 | image autoencoder, unmodified |

The Apache License 2.0 text: https://www.apache.org/licenses/LICENSE-2.0

Modifications: the ERNIE-Image transformer weights were fully fine-tuned on a curated dataset of photographs and then
edited by a weight-level safety pass (concept erasure of children; see the model card). Verboa Image Flash adds the
difference between ERNIE-Image-Turbo and ERNIE-Image at a scale of 0.85. The text encoder and the VAE are unmodified.
