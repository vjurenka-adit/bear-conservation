# Masterclass LightGlue

AI for turtle challenge for identification - they present their results.

Result for detecting out of dataset turtles.

__Speakers__: Marcus Leiwe (Consultant DataScientist at Metacell) and Laurenz Schindler

## Comparisons

LightGlue vs Metric Learning vs LoFRT
LightGlue was by far the best

## Keypoints

What are keypoints? How to match them?

## Feature extraction

Definition: Converting key parts of the image into vectors.

Questions that arise:

1. How do you find what is a key part of the image?
2. What information do you store in the vector? (descriptors)

### Classical

Relies on hard rules to find keypoints.

__SIFT__: Gold standard in classical feature extraction

Face detection and segmentation can be used to remove any noise from the image.
Table reference with pros/cons of classical approaches.

#### SIFT

Process:

1. Grayscale
2. Scale-space extrema detection (make spatial invariants)
3. keypoint refinement
4. Orientation assignment and refinement
5. Descriptor computation

_SIFT_ works by identifying areas of high contrast that are insensitive to scaling.

### Deep Learning

Rely heavily on assumptions and heuristics. It does not give us a
guarantee about the robustness or correctness.

A more data driven approach can be picked for this approach.

=> Improve keypoint location and descriptor embeddings.

3 supported feature extractors:

- SuperPoint
- DISK
- ALIKED
- (Dog HardNet) - recently added

There is no best approach...
SIFT worked best for them - most simple standard.

Trained/Evaluated on landmark datasets:

- Performance comparisons limited to this domain
- Not necessarily translatable to your problem

#### SuperPoint

CNN for computing keypoint locations and descriptors in a single pass.
Works in a self supervised manner.

### LightGlue

Reworked version of SuperGlue.

__Sparse feature matcher__: computes matchability given two keypoint/descriptor sets

Assignment vector from LightGlue output.

### Identifying the best matches

#### Base method

The base method for comparison is finding the
image with the most matching point: 77%
accuracy score on a trial dataset.

Possible reasons for suboptimal performance:
- number of keypoints not consistent across images
- number of thresholding scores also throws away a lot of information

We are throwing a lot of information away!

#### Distributions of scores

- Cumulative frequency plots
- Keypoint scores from matching images are much higher (curve shifts right/down)
- So what we are really doing is comparing two distributions
- Using the Wasserstein distances we can
calculate how much energy is required to
match the null distribution:
  - __Larger number__ = Matching image
  - __Small Number__ = Non-matching
- Could take the area under the curve

#### Additional factors to consider

- Segmentation
- Image quality (downsizing reduces the quality of the matching)
  - The faces of the turtles have very high quality patches
- Rotation angle (large angle difference (> 90 degrees) tricky
- Processing time: 30s on a GPU (20min on CPU)
- Novelty detection is a separate question

#### Misc

__Metric learning__: one needs to have many pairs of matching images in the set for a proper training (positive examples).

## Questions

- How did you setup your train/val/test splits? Any potential data leaks?
- What's the matching accuracy without segmenting the head? Just using the head patch detected by the face detector?
- For the 30s inference time for LightGlue, where is the time spent? How d
- Siamese Networks

### Asked

- How heavy is SIFT in terms of computation? Can it be used on microcontrollers?
- Can it be embedded on small hardware? What is the hardware constraints?
