# Assignment 2 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you. Write only what you
can actually explain.*

- **Name**:
- **Student ID**:
- **Email**:
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

## Business problem, and the cost of being wrong

**Decision.** An agricultural cooperative in Andalusia employs a field technician
who visits members' olive groves (olive oil production) in autumn and spring, when
olive peacock spot (*repilo*) spreads with humidity. At each parcel the technician
picks a sample of leaves, photographs one on a white card with a phone, and must
decide: **apply a copper treatment to the parcel, or not**. The model classifies
the leaf as healthy, Aculus olearius or peacock spot; the decision depends on the
probability of peacock spot. Unclear cases are sent to the cooperative's
agronomist.

**Positive class: peacock spot.** Aculus olearius is detected but does not trigger
treatment: in oil olives its damage is generally not significant and rarely needs a
specific treatment (it matters mainly for table olives and nursery plants). The
technician only logs it for monitoring.

**What a false negative costs (~€400/ha).** Missing peacock spot lets it defoliate
the trees, which reduces this harvest and weakens the next one. Historical average
losses in Spain are around 6% of production, and the "repilos" together cause more
than 10%, with catastrophic years locally. A traditional irrigated grove produces
about 6,000 kg of olives/ha at a 21% oil yield (≈1,260 kg oil/ha), worth ≈€4,000/ha
at the current origin price of ≈€3.20/kg (Poolred, October 2026). A 10% loss is
≈€400/ha.

**What a false positive costs (~€110/ha).** An unnecessary copper spray: product,
diesel and labour. Phytosanitary treatments cost ≈€155–200/ha per year across 2–3
applications in AEMO's cost study, and agricultural inputs have risen by more than
70% since. It also uses up the EU copper allowance (max. 28 kg Cu/ha over 7 years),
which a grower may need in a wet year.

**Asymmetry.** Treating costs €110 whether or not the parcel is infected; not
treating costs €400 only if it is. So treating pays off when
P(peacock spot) × 400 > 110, i.e. P > 0.275, well below 0.5. This break-even moves
with the olive oil price: at double the price (a missed case costs ~€800) it drops
to 0.14; at half the price (~€200) it rises to 0.55.

**How often the positive case occurs.** I assume about 25% of the parcels the
technician visits actually need a copper treatment. Visits are not random: the
technician is mostly called to parcels where the grower already suspects a
problem. This base rate drives the cost analysis below, where I also test how the
answer changes between 5% and 60%.

## Dataset

**Source.** Roboflow Universe, workspace `project-9k7w7`, project `olive1-ngwty`,
version 1 (classification, CC BY 4.0):
https://universe.roboflow.com/project-9k7w7/olive1-ngwty

**What one image is.** A single olive leaf photographed on a plain light
background, labelled with one of three classes: `Healthy`, `aculus_olearius` or
`olive_peacock_spot`. The task is image classification, so I use `yolo11n-cls`.
A leaf dataset with the same three classes and a similar size (~3,400 leaves) has
been used in published research on olive disease classification, which gives me
some confidence in the labels.

**Why it serves the decision, not just resembles it.** The technician's decision
is made from a leaf, and peacock spot is diagnosed from its visible lesions on the
leaf surface, so a leaf-level classifier matches how the decision is actually
taken. The main gap is the background: these are lab-style photos on white, while
a phone in the field would see branches, sky and shade. I close that gap with the
protocol itself: the technician photographs the sampled leaf on a white card.
This costs a few seconds per parcel and makes production images resemble the
dataset. What the card does not remove (a different phone, lighting, olive
varieties not in the dataset) is discussed under limitations.

## What the dataset's composition implies

## What the dataset's composition implies

| Class | Train (originals) | Train (files on disk) | Val | Test |
|---|---|---|---|---|
| Healthy | 756 | 2,268 | 49 | 86 |
| aculus_olearius | 746 | 2,238 | 42 | 88 |
| olive_peacock_spot | 1,242 | 3,726 | 67 | 149 |
| **Total** | **2,744** | **8,232** | **158** | **323** |

**The headline split is misleading.** Roboflow reports 94/2/4, but the train split
contains 3 augmented copies of every photo. Counting distinct originals (the part of
the filename before `.rf.`), the real split is 85/5/10 over 3,225 photos. I checked
that no original appears in more than one split: train/val, train/test and val/test
overlaps are all 0, so there is no leakage. All images are 224×224 (stretched by
Roboflow); one leaf fills a large part of the frame, but early peacock spot
lesions are only a few pixels at this resolution.

**Consequences.**
- *Peacock spot is the majority class* (46% of test). The trivial majority baseline
  therefore always predicts peacock spot, which in business terms means "treat every
  parcel". It is a meaningful baseline and I price it in euros later.
- *The test base rate (46%) is not the production base rate (25%).* Costs computed
  directly on the test set would overweight positives, so I reweight the test
  results to the assumed base rate.
- *Per-class numbers have wide error bars.* With 86–149 test images per class, a
  95% recall carries roughly ±4–5 points of uncertainty, so differences of 2–3
  points between models are not meaningful. Validation is very small (42–67 per
  class), so the choice of best epoch is noisy.
- *Backgrounds differ by class.* Healthy and Aculus leaves are all on the same
  uniform light paper; peacock spot images include dark vignetting, a wooden table
  and bluish backgrounds, and follow a different file-naming pattern, suggesting
  separate photo sessions. The model could partly learn "unusual background =
  peacock spot". The white-card protocol limits this in production, and I check the
  misclassified images for it during evaluation.

## Finetuning and baselines

I treat this as image classification and finetune `yolo11n-cls` from its ImageNet
checkpoint. Each image is one leaf and the decision is about that leaf, so a
classifier fits. The nano model takes about 3 ms per image, which is light enough
to run on a phone. I trained on Google Colab (Tesla T4) at 224×224 with batch 64,
up to 30 epochs with early stopping (patience 10) and seed 42. The best validation
epoch was 20, and training took about 19 minutes. Train loss kept falling while
validation loss flattened after epoch 20, a mild sign of overfitting, so I keep
`best.pt` rather than `last.pt`.

| Model (same 323 test images) | Accuracy | Peacock spot recall | Peacock spot precision |
|---|---|---|---|
| Finetuned yolo11n-cls | **0.975** | 0.973 | 0.993 |
| Pretrained zero-shot (val mapping) | 0.409 | 0.450 | 0.493 |
| Majority class (always peacock spot) | 0.461 | 1.000 | 0.461 |

The pretrained ImageNet model labels olive leaves as "cucumber", "lacewing" or
"quill", in similar proportions for all three classes. It recognises something
green and elongated but has no notion of lesions. Since its labels never match
ours, I map each ImageNet label to the olive class it most often corresponds to on
the validation split. Even with that help it scores 40.9%, below the trivial
baseline, so without finetuning the model is of no use for this decision.

Always predicting peacock spot gives perfect recall because it means treating
every parcel, and it pays for an unnecessary treatment on more than half of them.
Comparing it with the model on accuracy or recall says little; the cost section
compares them in euros.

The best epoch scored 95.6% on validation and the model scores 97.5% on test. With
158 validation images each error moves accuracy by 0.6 points, so I read the gap
as noise.

## Per-class evaluation and failure modes

| Class | Precision | Recall | Test images |
|---|---|---|---|
| Healthy | 0.955 | 0.988 | 86 |
| aculus_olearius | 0.966 | 0.966 | 88 |
| olive_peacock_spot | 0.993 | 0.973 | 149 |

In the confusion matrix (rows are true classes, columns predicted), Healthy is
85/1/0, Aculus 2/85/1 and peacock spot 2/2/145. Peacock spot is the positive class
and carries the headline number, as the largest and most accurate class. With 149
images its 97.3% recall has roughly ±2.6 points of uncertainty.

Only 5 of the 8 errors cost money. Mixing up Healthy and Aculus does not change the
decision, because neither gets copper. The errors that matter are 4 missed peacock
spots (false negatives) and 1 Aculus leaf flagged as peacock spot (a false
positive).

Two of the missed peacock spots were near misses, with P(peacock spot) of 0.43 and
0.46. One is a thin leaf with a single small, faint lesion near the tip; the other
is a deformed leaf with a hole, on a vignetted background. A peacock spot threshold
below about 0.43 would catch both, which is why I move the threshold away from the
default of picking the most probable class. The other two were confident misses
(0.03 and 0.01): an underexposed, almost black leaf where the lesions are barely
visible, and a leaf with an unusual bluish tint on a wooden table. No reasonable
threshold recovers these; photographing on the white card in decent light is the
mitigation. The false positive is an Aculus leaf with a whitish patch that looks
like a lesion. The model gave it P = 0.85, so a small change in threshold won't
remove it.

I also checked whether the model relies on the background. The peacock spot leaf on
the wooden table was predicted as Aculus. If the model had learned "unusual
background means peacock spot", this is the image it would most easily flag. That,
and the zero-shot model giving the same labels to all three classes, suggest the
model is using the leaf. I can't rule out some use of the background with this
test set.

## Cost matrix and operating point

*Your euro cost for each kind of error and where each figure came from. The
base rate you assumed. The cost-against-threshold sweep, and the operating
point you chose. State the assumption that would most change your answer if it
were wrong.*

## What a person still has to check

*Your two thresholds, and what share of cases they send to a person. How many
that is per day, how long one takes to check, and how many people that needs.
Is that realistic? Then your design priced against the three alternatives:
doing nothing, having a person check everything, and letting the model decide
on its own.*

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data. Not generic hedging.*

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
