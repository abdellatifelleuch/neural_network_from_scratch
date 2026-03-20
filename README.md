# neural_network_from_scratch

## Challenges & Lessons Learned

Building this neural network from scratch was a great learning experience, but came with several real-world difficulties that are rarely mentioned in tutorials. The first major issue was **memory** — loading the full PetImages dataset (25,000 images at 64×64 RGB) consumed all 24GB of RAM, freezing the kernel before training even started. The fix was reducing image size to 32×32 and limiting the dataset to 2,000 images, which brought memory usage down to ~200MB. The second issue was **numerical instability** — the cost function crashed with `log(0) = -inf` because the sigmoid output was saturating to exact 0.0 or 1.0 in float32, caused by an aggressive learning rate of 1.2 exploding the weights. This was fixed by clipping A2 between `1e-8` and `1-1e-8` before taking the log, and lowering the learning rate to `0.01`. The third challenge was **training speed** — on an Intel i5 12th gen CPU, one iteration over the full dataset takes ~1.5 seconds, meaning 10,000 iterations would take roughly 4 hours. Using a smaller dataset reduced this to ~13 minutes. These challenges taught me that in practice, data pipeline efficiency, numerical stability, and hardware constraints matter just as much as getting the math right. 


