# Glossary

Part of [image-retrieval](../README.md). The terms the other pages use without stopping to
explain them.

## The task and how it is scored

**Content-based image retrieval (CBIR).** Given a photo (the *query*), rank every photo in
a collection (the *database*) by how likely it shows the same thing — here, the same
building. No captions or tags: only pixels.

**Instance retrieval.** Finding *this* building, not *a* building. A system that returns
other cathedrals for a query of Notre-Dame is good at semantic similarity and wrong here.

**roxford5k, rparis6k.** The benchmark: revisited Oxford (4993 photos, 70 queries) and Paris
(6322 photos, 70 queries) landmark datasets (Radenović et al., 2018). Each query comes with
a bounding box, and the labels describe what is inside it.

**bbx crop.** Queries are cropped to their bounding box before anything is computed. The
labels were made against the crop, so an uncropped query is scored against labels that do
not describe it.

**Easy / Medium / Hard.** The benchmark's three protocols, differing in which database
photos count as correct. *Hard* positives are seen from extreme viewpoints, far away or
heavily occluded; *Medium* counts both easy and hard positives and is the headline number in
the literature.

**AP, mAP.** Average precision scores one ranked list: 1.0 when every correct photo is
ranked above every wrong one, lower the more wrong photos come first. mAP is its mean over
the queries. All numbers here are mAP on 0–1; papers often print ×100.

**mP@10.** Mean precision among the top 10 results. Recorded, not headlined.

**Held-out set.** Anything fitted — a vocabulary, a PCA, a whitening — is fitted on the
*other* dataset (rparis6k when evaluating roxford5k, and vice versa), never on the photos
being searched.

## Descriptors

**Descriptor.** The vector a method turns an image into. Retrieval ranks the database by
similarity between descriptors. *Global*: one vector per image, which is what every method
here produces.

**SIFT, RootSIFT.** Hand-crafted local features: hundreds of small patch descriptors per
image. RootSIFT is SIFT with a square-root normalization that makes comparisons fairer.

**Visual vocabulary (codebook).** k-means clusters of local descriptors; each cluster is a
"visual word".

**BoW, VLAD, Fisher vectors.** Three ways of turning an image's local descriptors into one
vector via the vocabulary: count the words (BoW), sum each word's residuals (VLAD), or
record first- and second-order statistics (Fisher).

**Neural Codes.** The activations of a CNN's first fully connected layer, used as the
descriptor. The first deep-learning approach to this task (2014).

**Pooling: SPoC, MAC, GeM, R-MAC.** Ways to collapse a CNN's last feature map into one
vector: average (SPoC), max (MAC), the generalized mean between the two with exponent `p`
(GeM), or max over a grid of regions (R-MAC).

**Whitening.** A linear projection learned on held-out descriptors that decorrelates and
rescales their dimensions. *PCA-whitening* learns it without labels; *supervised*
whitening learns it from pairs of matching photos.

**Multi-scale.** Describing an image at several resolutions (here 1, 1/√2 and 1/2) and
pooling the results.

**CLS token.** A vision transformer's summary of the whole image, the vector it was trained
to put its answer in. The usual descriptor for a ViT.

## Models and training

**Frozen / off-the-shelf.** A network used as published, trained for something else
(ImageNet classification, self-supervision) and never trained here for retrieval.

**Fine-tuning.** Continuing a network's training on the retrieval task itself.

**retrieval-SfM-120k.** The training corpus: landmark photos grouped into 3D models by
structure-from-motion. Two photos reconstructed into the same model are a matching pair —
supervision with no human labelling.

**Contrastive loss, hard negatives.** Training pulls matching pairs together and pushes
non-matching ones apart; *hard* negatives are the non-matching photos the current network
finds most similar, re-mined every epoch.

**Caffe weights.** ImageNet weights converted from the original Caffe models, which the
reference implementation uses instead of torchvision's. They score differently enough to
matter.

**ViT.** Vision transformer: an image cut into patches, processed as a sequence.

**DINOv2, DINOv3.** ViTs trained with *self-supervision*: no labels, just learning to give
different views of the same image the same representation.

**SigLIP 2.** A ViT trained to match images with their text captions.

**DeiT (†).** A ViT trained as an ImageNet classifier; † marks the distilled variant.

**IRT.** *Image Retrieval Transformer* (El-Nouby et al., 2021): the GeM fine-tuning recipe
applied to DeiT.

**LoRA.** Low-rank adaptation: training a small add-on to each weight matrix while the
pretrained weights stay frozen.

**`pad`, `stretch`.** How an image becomes the square a ViT takes: centred on black
bars (`pad`, the DINOv3 protocol) or squashed (`stretch`, IRT's).
