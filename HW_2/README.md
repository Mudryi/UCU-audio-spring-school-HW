# HomeWork 2 - Alignment Decoding

I compute text aligment for a two samples (0-th and 100-th) in dev-clean dataset and got following results in Praat

![results_0.png](results_0.png)
![results_100.png](results_100.png)

Both results are not suitable for the following reasons:
- Model predicts CTC blank "ε" in the beginning more than actual silence is, so the speaker starts talking, but we still have an "ε" symbol.
- As I know, each phoneme has a different duration, but we have a raw estimation that each phoneme is the same, so there are a lot of phonemes at the end of the sentence with alignment duration much more than speaking duration.
- It may be a problem that we use convolution with size four so some phonemes can intersect. For example, we have five frames, 3 have a phoneme "AA1" and 2 "Z" model will predict the phoneme "AA1" and with my simple alignment method, we will give all five frames the label "AA1".
