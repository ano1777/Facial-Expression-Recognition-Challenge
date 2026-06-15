# სახის გამომეტყველების ამოცნობა — FER2013

**Kaggle Competition:** [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge)
**WandB Project:** [ყველა ექსპერიმენტის ნახვა](https://wandb.ai/akave23-free-university-of-tbilisi-/Facial-Expression-Recognition-Challenge)
**WandB Report:** [სრული ანალიზი](https://wandb.ai/akave23-free-university-of-tbilisi-/Facial-Expression-Recognition-Challenge/reports)

---

## დავალების არსი

48×48 პიქსელიანი შავ-თეთრი სახის სურათების კლასიფიკაცია 7 ემოციის კატეგორიად PyTorch-ის გამოყენებით.

| ID | ემოცია | Train-ის რაოდენობა |
|----|--------|-------------------|
| 0 | Angry | 3,995 |
| 1 | Disgust | 436 |
| 2 | Fear | 4,097 |
| 3 | Happy | 7,215 |
| 4 | Sad | 4,830 |
| 5 | Surprise | 3,171 |
| 6 | Neutral | 4,965 |

**Disgust** კლასი ძალიან ცოტა მაგალითს შეიცავს (436 vs 7,215 Happy) — დაახლოებით 16-ჯერ ნაკლები. ეს severe class imbalance-ია, რომელიც ართულებს მის კლასიფიკაციას. ეს იმიტომ, რომ ინტერნეტში ზიზღის გამომხატველი სახის ფოტო ბევრად იშვიათია, ვიდრე სიხარულის ან ნეიტრალური გამომეტყველება.

---

## რეპოზიტორიის სტრუქტურა

```
├── facial-expressions.ipynb
└── README.md
```
---

## sanity checks (ტრენინგამდე)

სწავლების დაწყებამდე სამი შემოწმება კეთდება, რომ დავრწმუნდეთ კოდის სისწორეში:



- **Forward pass** - მოწმდება აუთფუთის ზომა `(8, 7)`, საწყისი loss ≈ ln(7) ≈ 1.946. ეს დარწმუნდება, რომ weight initialization სწორია — შემთხვევითი წონებით მოდელი ყველა კლასს თანაბრად წინასწარმეტყველებს 
- **Backward pass** - მოწმდება ყველა trainable პარამეტრი იღებს non-zero gradient-ს.  ავლენს dead neurons-სა და broken computation graph-ს |
- **Overfit on 1 batch** - მოწმდება TinyNet 100 gradient step-ში ~100%-ს აღწევს 32 სურათზე (augment=False). ადასტურებს, რომ loss function და training loop სწორად არის დაწერილი 

ყველა 5 არქიტექტურა გადის ყველა 3 შემოწმებას.

---

## Data Augmentation

ტრენინგზე (Exp 3–10) გამოყენებული ტრანსფორმაციები:

| ტრანსფორმაცია | პარამეტრები | დასაბუთება |
|--------------|------------|-----------|
| `RandomHorizontalFlip` | p=0.5 | ემოციები სიმეტრიულია — flipped სახე იდენტური ემოციით |
| `RandomRotation` | ±10° | ადამიანები სხვადასხვა კუთხით ფოტოგრაფირდებიან |
| `RandomCrop` | size=48, padding=4 | მცირე translation invariance; 48×48-ზე padding=4 ოპტიმალურია |
| `ColorJitter` | brightness=0.2, contrast=0.2 | განათების variation-ის სიმულაცია; saturation/hue გამორთულია (grayscale) |

Validation/Test-ზე მხოლოდ `ToTensor()` + `Normalize(mean=[0.5], std=[0.5])`.

---

## Learning Rate Schedulers

| Scheduler | გამოიყენება | დასაბუთება |
|-----------|------------|-----------|
| `CosineAnnealingLR` (T_max=epochs) | Exp 3, 4, 6, 7, 9, 10 | LR გლუვად cos(π·t/T)-ის მიხედვით ეცემა ნულამდე; final convergence-ს აუმჯობესებს |
| `ReduceLROnPlateau` (patience=5, factor=0.5) | Exp 8 | val loss-ი 5 epoch-ს არ გაუმჯობესდეს → LR ×0.5; frozen head-ის სწრაფ training-ს ეხმარება |
| None | Exp 1, 2, 5 | Exp 1 და 2 — სუფთა baseline; Exp 5 — constant LR-ით optimizer-ის შედარება |

---

## არქიტექტურების პროგრესი

მიდგომა სრულიად იტერაციულია — თითოეული ექსპერიმენტი კონკრეტულ პრობლემას ავლენს, შემდეგი ექსპერიმენტი კი ამ პრობლემის გამოსწორებას ახდენს:

```
Exp 1:  TinyNet            → underfitting       → გადაწყვეტილება: მეტი layer-ი სჭირდება
Exp 2:  SmallCNN (d=0)     → overfitting        → გადაწყვეტილება: regularization სჭირდება
Exp 3:  SmallCNN (d=0.4)   → გაუმჯობესებული    → გადაწყვეტილება: სიღრმე გასაზრდელია
Exp 4:  MediumCNN + Adam   → დაბალანსებული     → გადაწყვეტილება: optimizer-ის შედარება
Exp 5:  MediumCNN + SGD    → Adam იმარჯვებს     → გადაწყვეტილება: LR-ის ტესტირება
Exp 6:  MediumCNN LR↑      → არასტაბილური      → გადაწყვეტილება: residual კავშირები
Exp 7:  DeepCNN            → საუკეთესო CNN      → გადაწყვეტილება: transfer learning
Exp 8:  ResNet frozen      → შეზღუდული          → გადაწყვეტილება: სრული fine-tune
Exp 9:  ResNet finetune    → CNN-ზე ნაკლები     → გადაწყვეტილება: არქიტექტურის გამოსწორება
Exp 10: ResNet v2 (fixed)  → გაუმჯობესებული    → საბოლოო მოდელი
```

---

## ექსპერიმენტები

### ექსპერიმენტი 1 — TinyNet საბაზისო მოდელი

**არქიტექტურა:**
```
Input(1×48×48)
  → Conv2d(1→8, 3×3, pad=1) → ReLU → MaxPool2d(2×2)   # 8×24×24
  → Conv2d(8→16, 3×3, pad=1) → ReLU → MaxPool2d(2×2)  # 16×12×12
  → Flatten → FC(2304→7)
```
**Trainable პარამეტრები:** 37,255

**კონფიგურაცია და დასაბუთება:**
- `optimizer=Adam, lr=0.001` — Adam-ის standard default; ნეიტრალური არჩევანი, რომ LR არ იყოს confounding factor
- `batch=64, epochs=30` — ნაკლები epoch საკმარისია; მეტი epoch underfitting-ს მდგომარეობას არ შეცვლის
- `augmentation=False, dropout=0, weight_decay=0` — ყოველგვარი regularization-ის გარეშე, რომ underfitting სუფთად გამოვლინდეს

**ჰიპოთეზა:** 2 conv layer-ი მხოლოდ 8 და 16 ფილტრით ვერ შეძლებს 7 ემოციის განასხვავებას. მოდელს არ აქვს საკმარისი სიმძლავრე.

**მოსალოდნელი შედეგი:** Train და val სიზუსტე ორივე დაბალი (~35–42%), მათ შორის მცირე სხვაობა — underfitting-ის კლასიკური ნიშანი.

**რეალური შედეგი:**
```
საუკეთესო Val : 50.63%
Test           : 49.60%
```

**ანალიზი:** მოდელი underfitting-ს ავლენს. Val სიზუსტე 50.63% — train loss-ი ადრე პლატოზე ავიდა, train და val სიზუსტე ახლოს არის ერთმანეთთან. ეს underfitting-ის კლასიკური ნიშანია: მოდელი ვარჯიშის მონაცემებშიც კი ვერ ასახავს სიგნალს.

**გადაწყვეტილება:** მოდელი ძალიან მარტივია. საჭიროა მეტი conv layer-ი და ფილტრების გაზრდა.

---

### ექსპერიმენტი 2 — SmallCNN (Dropout-ის გარეშე)

**არქიტექტურა:**
```
Input(1×48×48)
  → Conv2d(1→32, 3×3) + BN + ReLU + MaxPool    # 32×24×24
  → Conv2d(32→64, 3×3) + BN + ReLU + MaxPool   # 64×12×12
  → Conv2d(64→128, 3×3) + BN + ReLU + MaxPool  # 128×6×6
  → Flatten → FC(4608→512) → ReLU → FC(512→7)
```
**Trainable პარამეტრები:** 1,234,247

**კონფიგურაცია და დასაბუთება:**
- `optimizer=Adam, lr=0.001, epochs=50` — საკმარისი epoch, რათა მოდელი ტრენინგ-სეტს ზეპირად ისწავლოს
- `augmentation=False, dropout=0, weight_decay=0` — regularization-ი განზრახ გამორთული, overfitting-ის სრულად გამოვლენისთვის

**ჰიპოთეზა:** 3 conv block-ი BatchNorm-ით საკმარის სიმძლავრეს იძლევა, მაგრამ regularization-ის გარეშე მოდელი ტრენინგ-სეტს ზეპირად ისწავლის.

**მოსალოდნელი შედეგი:** Train სიზუსტე მაღალი (~75–85%), val სიზუსტე დაბალი (~55–60%) — დიდი სხვაობა = overfitting.

**რეალური შედეგი:**
```
საუკეთესო Val : 60.24%
Test           : 60.32%
```

**ანალიზი:** WandB-ის გრაფიკები გვიჩვენებს: train სიზუსტე ~80%-ს მიაღწია, val კი 60%-ზე გაჩერდა. ~25 epoch-ის შემდეგ val loss-ი იზრდება, train loss-ი კი კლებულობს — overfitting-ის სახელმძღვანელო ნიმუში. Exp 1-თან შედარებით val +10% — დიდი არქიტექტურა ნამდვილ ნიშნებს სწავლობს, მაგრამ ძალიან "ახსოვს" ტრენინგ-სეტი.

**გადაწყვეტილება:** არქიტექტურას სიმძლავრე აქვს, regularization-ი კი არ გააჩნია. გადაწყვეტა: Dropout + data augmentation.

---

### ექსპერიმენტი 3 — SmallCNN (Dropout=0.4 + Augmentation)

**არქიტექტურა:** იდენტური Exp 2-ის
**Trainable პარამეტრები:** 1,234,247

**კონფიგურაცია და დასაბუთება:**
- `dropout=0.4` FC layer-ების წინ — 0.4 კარგი trade-off-ია regularization-სა და underfitting-ს შორის; ძალიან მაღალი dropout (>0.5) underfitting-ს გამოიწვევს
- `augmentation=True` — RandomFlip, Rotation, Crop, ColorJitter ხელოვნურად ზრდის ეფექტური dataset-ის მოცულობას
- `weight_decay=1e-4` — მსუბუქი L2 regularization, რომელიც dropout-ს ავსებს
- `scheduler=CosineAnnealing` — LR-ს გლუვად ამცირებს, stable convergence-ს უზრუნველყოფს

**ჰიპოთეზა:** Dropout(0.4) ნეირონების ერთობლივ ადაპტაციას ხელს უშლის. Data augmentation ეფექტურ სამპლების რაოდენობას ზრდის. ორივე ერთად Exp 2-ის train-val სხვაობას მნიშვნელოვნად შეამცირებს.

**მოსალოდნელი შედეგი:** train-val სხვაობა მცირდება, val სიზუსტე Exp 2-ზე უკეთესია.

**რეალური შედეგი:**
```
საუკეთესო Val : 64.17%
Test           : 65.37%
```

**ანალიზი:** Val სიზუსტე +4%-ით გაუმჯობესდა Exp 2-სთან შედარებით (60.24% → 64.17%) — იდენტური არქიტექტურიდან, მხოლოდ regularization-ის დამატებით. Train-val სხვაობა მნიშვნელოვნად შემცირდა. ეს bias-variance tradeoff-ის პრაქტიკული დემონსტრირებაა.

**გადაწყვეტილება:** Regularization ეფექტურია, მაგრამ 3 conv block-ი მაინც შემზღუდველია. საჭიროა სიღრმის გაზრდა.

---

### ექსპერიმენტი 4 — MediumCNN + Adam

**არქიტექტურა:**
```
Input(1×48×48)
  Block 1: Conv(1→32)+BN+ReLU + Conv(32→32)+BN+ReLU + MaxPool + Dropout2d(0.1)   # 32×24×24
  Block 2: Conv(32→64)+BN+ReLU + Conv(64→64)+BN+ReLU + MaxPool + Dropout2d(0.1)  # 64×12×12
  Block 3: Conv(64→128)+BN+ReLU + Conv(128→128)+BN+ReLU + MaxPool + Dropout2d(0.2) # 128×6×6
  Block 4: Conv(128→256)+BN+ReLU + MaxPool                                         # 256×3×3
  → Flatten → Dropout(0.4) → FC(2304→512) → ReLU → Dropout(0.4) → FC(512→256) → ReLU → FC(256→7)
```
**Trainable პარამეტრები:** 2,847,879

> პირველი 3 block double-conv-ია (VGG-style), მე-4 block კი single conv. Spatial Dropout2d მთელ feature map-ებს "გამორთავს", რაც redundant ნიშნების სწავლას აიძულებს.

**კონფიგურაცია და დასაბუთება:**
- `optimizer=Adam, lr=0.001` — Adam-ის recommended default; adaptive learning rates კარგია მრავალფეროვანი gradient magnitude-ებისთვის
- `weight_decay=1e-4` — მსუბუქი L2; dropout-ს ავსებს
- `epochs=60` — SmallCNN-ზე მეტი, რადგან მოდელი სიღრმით დიდია
- `scheduler=CosineAnnealing` — 60 epoch-ზე LR-ს გლუვად ნულამდე მიჰყავს
- `batch=64` — სტანდარტული; მეტი სამპლი batch-ში = სტაბილური gradient

**ჰიპოთეზა:** ორმაგი conv block-ები თითოეულ სივრცულ მასშტაბზე უფრო კომპლექსურ ნიშნებს ამოიღებს. 4 block-იანი სიღრმე hierarchical features-ს ისწავლის.

**მოსალოდნელი შედეგი:** აქამდე საუკეთესო სიზუსტე, train ≈ val.

**რეალური შედეგი:**
```
საუკეთესო Val : 66.51%
Test           : 69.41%
```

**ანალიზი:** MediumCNN SmallCNN-ს +2.34%-ით სჯობს. სწავლება სტაბილური იყო, მცირე train-val სხვაობა კარგ regularization-ზე მიუთითებს.

**გადაწყვეტილება:** სიღრმე ეხმარება. ახლა შევამოწმოთ optimizer-ის არჩევანი იზოლირებულად.

---

### ექსპერიმენტი 5 — MediumCNN + SGD (Optimizer-ის შედარება)

**არქიტექტურა:** იდენტური Exp 4-ის

**კონფიგურაცია და დასაბუთება:**
- `optimizer=SGD+Nesterov, momentum=0.9` — Nesterov momentum "lookahead" gradient-ს ითვლის, უფრო სწრაფი კონვერგენცია SGD-სთან შედარებით
- `lr=0.01` — SGD-სთვის Adam-ზე 10× მაღალი LR ნორმალურია; Adam-ის adaptive scaling-ის გარეშე SGD-ს უფრო მაღალი global LR სჭირდება
- ყველა დანარჩენი პარამეტრი Exp 4-ის იდენტური — optimizer-ის ეფექტი იზოლირებულია

**ჰიპოთეზა:** SGD+Nesterov ზოგჯერ Adam-ზე უკეთ განზოგადება. იდენტური არქიტექტურით optimizer-ის ეფექტი სუფთად იზომება.

**მოსალოდნელი შედეგი:** Adam-ზე ნელი კონვერგენცია, მსგავსი ან ოდნავ სხვადასხვა საბოლოო სიზუსტე.

**რეალური შედეგი:**
```
საუკეთესო Val : 63.39%
Test           : 66.09%
```

**ანალიზი:** Adam (66.51%) SGD-ს (63.39%) 3.12%-ით სჯობს. SGD 60 epoch-ში ჯერ კიდევ კონვერგენციის პროცესში შეიძლება იყოს — მეტ epoch-ს ან LR warmup-ს საჭიროებს. მცირე dataset-ებზე Adam-ის ადაპტური learning rate-ები მნიშვნელოვან უპირატესობას იძლევა.

**გადაწყვეტილება:** Adam ამ ამოცანისთვის უკეთესია. ყველა შემდგომ ექსპერიმენტში Adam/AdamW გამოვიყენებ.

---

### ექსპერიმენტი 6 — MediumCNN მაღალი LR-ით (LR Ablation)

**არქიტექტურა:** იდენტური Exp 4-ის

**კონფიგურაცია და დასაბუთება:**
- `optimizer=Adam, lr=0.01` — Exp 4-ის პირდაპირი ablation: მხოლოდ LR 10×-ით გაიზარდა, ყველა სხვა პარამეტრი იდენტური
- LR=0.01 Adam-ისთვის ამ ამოცანაში განზრახ "ცუდი" კომბინაციაა — Adam lr=0.001-ისთვის არის optimized

**ჰიპოთეზა:** LR=0.01 Adam-ისთვის ძალიან მაღალია. Optimizer loss minimum-ს "გადასწვდება", ოსცილაციას გამოიწვევს.

**მოსალოდნელი შედეგი:** არასტაბილური სწავლება, Exp 4-ზე მნიშვნელოვნად ცუდი შედეგი.

**რეალური შედეგი:**
```
საუკეთესო Val : 52.66%
Test           : 53.44%
```

**ანალიზი:** ერთი ჰიპერპარამეტრის ცვლილებამ (LR: 0.001 → 0.01) **14%-იანი სიზუსტის ვარდნა** გამოიწვია (66.51% → 52.66%). ეს ყველა ექსპერიმენტს შორის ყველაზე დრამატული ცალი-ცვლადიანი პოვნაა. WandB-ის loss გრაფიკი მაღალ ოსცილაციას აჩვენებს Exp 4-ის გლუვ მრუდთან შედარებით. LR ყველაზე კრიტიკული ჰიპერპარამეტრია — Adam-ისთვის lr ∈ [1e-4, 3e-3] რეკომენდებული range-ია.

---

### ექსპერიმენტი 7 — DeepCNN Residual Block-ებით

**არქიტექტურა:**
```
Input(1×48×48)
  Stem:   Conv2d(1→64, 3×3) + BN + ReLU                                  # 64×48×48
  Stage1: ResBlock(64) + MaxPool                                           # 64×24×24
  Stage2: Conv(64→128) + BN + ReLU + ResBlock(128) + MaxPool              # 128×12×12
  Stage3: Conv(128→256) + BN + ReLU + ResBlock(256) + MaxPool             # 256×6×6
  Stage4: Conv(256→512) + BN + ReLU + MaxPool                             # 512×3×3
  → GlobalAvgPool → Flatten                                               # 512
  → Dropout(0.5) → FC(512→256) → ReLU → Dropout(0.25) → FC(256→7)

ResBlock(c): output = ReLU(x + [Conv(c→c)+BN+ReLU+Conv(c→c)+BN])
```
**Trainable პარამეტრები:** 6,573,127

**კონფიგურაცია და დასაბუთება:**
- `optimizer=AdamW, weight_decay=1e-3` — AdamW L2 regularization-ს სწორად ასრულებს (Adam-ში weight_decay gradient-ის შემდეგ ვრიცხავ, AdamW-ში — დამოუკიდებლად; უფრო სწორი regularization)
- `dropout=0.5` პირველ FC-ს წინ, `0.25` მეორეს წინ — progressive dropout; მეტი regularization feature extraction-ის გამოსასვლელთან
- `batch=32` — 64-დან შემცირება; უფრო ხშირი gradient update, მცირედ noisy — generalization-ს ეხმარება
- `epochs=70` — ყველაზე ღრმა ქსელი, კონვერგენციისთვის მეტ iteration-ს საჭიროებს

**ჰიპოთეზა:** Residual (skip) კავშირები gradient-ს პირდაპირ ადრეულ layer-ებამდე გადასცემს — vanishing gradient-ი ქრება. GAP დიდ FC layer-ებს ანაცვლებს, overfitting-ს ამცირებს.

**მოსალოდნელი შედეგი:** ყველა pure-CNN ექსპერიმენტს შორის საუკეთესო.

**რეალური შედეგი:**
```
საუკეთესო Val : 68.99%
Test           : 69.63%
```

**ანალიზი:** DeepCNN ყველა ექსპერიმენტს შორის საუკეთესო შედეგს (68.99%) აღწევს. Skip კავშირებმა შეძლო გაცილებით ღრმა ქსელის ვარჯიში gradient-ის გაქრობის გარეშე. GAP-მა FC პარამეტრები 512-ამდე შეამცირა (MediumCNN-ში 2304 იყო), overfitting-ი შემცირდა. AdamW + weight_decay=1e-3 ძლიერი regularization-ია.

**გადაწყვეტილება:** Custom CNN residual კავშირებით საუკეთესოა. შევამოწმოთ transfer learning.

---

### ექსპერიმენტი 8 — ResNet18 Frozen Backbone

**არქიტექტურა:**
```
ResNet18 (ImageNet pretrained)
  conv1 შეცვლილია: Conv2d(3→64, 7×7) → Conv2d(1→64, 7×7)  [grayscale ადაპტაცია]
  layer1–layer4: frozen (trainable=False)
  fc შეცვლილია: Dropout(0.5) → FC(512→256) → ReLU → Dropout(0.3) → FC(256→7)
```
**Trainable პარამეტრები:** ~530,000 (მხოლოდ conv1 და head)

**კონფიგურაცია და დასაბუთება:**
- `freeze_backbone=True` — backbone-ი frozen; მხოლოდ conv1 და custom head ტრენინგდება
- `epochs=20` — head-ი randomly initialized, ამიტომ ნაკლები epoch საკმარისია
- `optimizer=Adam, lr=0.001` — head-ის fast training-ისთვის
- `scheduler=ReduceLROnPlateau` — val loss-ი 5 epoch-ს არ გაუმჯობესდეს → LR ×0.5

**ჰიპოთეზა:** ImageNet-ზე წინასწარ სწავლებული ნიშნები (კიდეები, ტექსტურები, ფორმები) სახის ემოციებისთვისაც სასარგებლო იქნება.

**მოსალოდნელი შედეგი:** გონივრული სიზუსტე სწრაფად, მაგრამ frozen ნიშნებით შეზღუდული.

**რეალური შედეგი:**
```
საუკეთესო Val : 57.15%
Test           : 58.21%
```

**ანალიზი:** Frozen ResNet18-მა SmallCNN + regularization-ზე **უარესი შედეგი** გამოიტანა (57.15% vs 64.17%). ImageNet-ის frozen ნიშნები 48×48 grayscale სახის ემოციებისთვის კარგად არ გადაიტანება — domain gap ძალიან დიდია. Head-ს ძალიან ცოტა პარამეტრი აქვს frozen ნიშნების კომპენსაციისთვის.

**გადაწყვეტილება:** Backbone-ი უნდა გაიხსნას — სრული fine-tune.

---

### ექსპერიმენტი 9 — ResNet18 სრული Fine-Tune

**არქიტექტურა:** იდენტური Exp 8-ის, მაგრამ `freeze_backbone=False`
**Trainable პარამეტრები:** ~11,400,000

**კონფიგურაცია და დასაბუთება:**
- `lr=0.0001` — Exp 8-ზე 10× დაბალი; catastrophic forgetting-ის თავიდან ასაცილებლად pretrained weights-ი ნელა ადაპტირდება
- `optimizer=AdamW, weight_decay=0.01` — ძლიერი L2; fine-tune-ში overfitting-ის რისკი მაღალია
- `epochs=50` — სრული fine-tune low LR-ზე მეტ iteration-ს საჭიროებს
- `scheduler=CosineAnnealing`

**ჰიპოთეზა:** მცირე LR-ით მთელი ქსელის fine-tune-ი ImageNet-ის ნიშნებს ემოციების ამოცნობასთან ადაპტირებს. DeepCNN-ს სჯობს.

**მოსალოდნელი შედეგი:** ყველა ექსპერიმენტს შორის საუკეთესო.

**რეალური შედეგი:**
```
საუკეთესო Val : 64.06%
Test           : 63.81%
```

**ანალიზი:** Fine-tune frozen-ზე +7%-ით უკეთესია, მაგრამ **DeepCNN-ს 5%-ით ჩამოუვარდება** (64.06% vs 68.99%). სამი მიზეზი:
1. `conv1` შეიცვალა შემთხვევითი layer-ით → პირველ layer-ში ყველა pretrained ნიშანი განადგურდა
2. 48×48 input ResNet-ის 224×224-ზე გათვლილ არქიტექტურასთან შეუთავსებელია → სივრცული ინფორმაცია ადრე იკარგება
3. 50 epoch LR=0.0001-ით შესაძლოა საკმარისი არ იყოს სრული ადაპტაციისთვის

**გადაწყვეტილება:** გამოვასწოროთ ორივე არქიტექტურული პრობლემა.

---

### ექსპერიმენტი 10 — ResNet18 გაუმჯობესებული (Exp 9-ის გამოსწორება)

**ძირითადი გამოსწორებები:**
- Input **224×224**-მდე გაზრდილია ResNet-ის სწორი გარჩევადობისთვის
- Grayscale 3 არხზე მრავლდება `x.repeat(1,3,1,1)` — conv1 **შენარჩუნებულია** pretrained წონებით
- Conv1-ის pretrained წონები სრულად დაცულია

**კონფიგურაცია:** AdamW, lr=0.0001, weight_decay=0.01, batch=32, augmentation, CosineAnnealing, 60 epoch

**ჰიპოთეზა:** ორი არქიტექტურული გამოსწორება პირდაპირ Exp 9-ის ჩავარდნის მიზეზებს აგვარებს. DeepCNN-ს სჯობს (>69%).

**შედეგი:** *(დასრულების პროცესშია)*

---

## საბოლოო შედეგების ცხრილი

| # | ექსპერიმენტი | არქიტექტურა | Opt | LR | Epochs | Batch | Aug | Val | Test | დიაგნოზი |
|---|-------------|------------|-----|----|--------|-------|-----|-----|------|---------|
| 1 | exp1_tinynet_baseline | TinyNet | Adam | 1e-3 | 30 | 64 | ❌ | 50.63% | 49.60% | **Underfitting** |
| 2 | exp2_smallcnn_no_dropout | SmallCNN d=0 | Adam | 1e-3 | 50 | 64 | ❌ | 60.24% | 60.32% | **Overfitting** |
| 3 | exp3_smallcnn_regularised | SmallCNN d=0.4 | Adam | 1e-3 | 50 | 64 | ✅ | 64.17% | 65.37% | Regularized |
| 4 | exp4_mediumcnn_adam | MediumCNN | Adam | 1e-3 | 60 | 64 | ✅ | 66.51% | 69.41% | Balanced |
| 5 | exp5_mediumcnn_sgd | MediumCNN | SGD | 1e-2 | 60 | 64 | ✅ | 63.39% | 66.09% | Adam > SGD |
| 6 | exp6_mediumcnn_highLR | MediumCNN | Adam | 1e-2 | 60 | 64 | ✅ | 52.66% | 53.44% | LR ძალიან მაღალი |
| 7 | **exp7_deepcnn_residual** | DeepCNN | AdamW | 1e-3 | 70 | 32 | ✅ | **68.99%** | **69.63%** | **🏆 საუკეთესო** |
| 8 | exp8_resnet18_frozen | ResNet18 frozen | Adam | 1e-3 | 20 | 32 | ✅ | 57.15% | 58.21% | Transfer შეზღუდული |
| 9 | exp9_resnet18_finetune | ResNet18 full | AdamW | 1e-4 | 50 | 32 | ✅ | 64.06% | 63.81% | მოულოდნელად ნაკლები |
| 10 | exp10_resnet18_v2 | ResNet18 fixed | AdamW | 1e-4 | 60 | 32 | ✅ | TBD | TBD | მიმდინარე |

---

## ძირითადი დასკვნები

### 1. Underfitting vs Overfitting (Exp 1 vs 2)
TinyNet (50.63%) — train და val ორივე დაბალი, მცირე სხვაობა = underfitting. SmallCNN regularization-ის გარეშე — train ~80% vs val 60.24% = overfitting. ეს ორი ექსპერიმენტი bias-variance tradeoff-ს ნათლად აჩვენებს.

### 2. Regularization ეფექტურია (Exp 2 vs 3)
იდენტურ არქიტექტურაზე Dropout + augmentation-ის დამატებამ **+4% val სიზუსტე** მოიტანა. Augmentation განსაკუთრებით ღირებულია FER2013-ის შეზღუდული dataset-ისთვის.

### 3. სიღრმე ეხმარება (Exp 3 → 4 → 7)
Val სიზუსტის პროგრესი: 64.17% → 66.51% → 68.99%. სიღრმის ყოველი ზრდა (სწორი regularization-ით) სიზუსტეს სტაბილურად აუმჯობესებდა.

### 4. Learning Rate კრიტიკულია (Exp 4 vs 6)
LR-ის ცვლილებამ 0.001-დან 0.01-მდე **14%-იანი ვარდნა** გამოიწვია (66.51% → 52.66%). ეს ყველა ექსპერიმენტს შორის ყველაზე დრამატული ერთ-ცვლადიანი პოვნაა. Adam-ისთვის lr ∈ [1e-4, 3e-3] რეკომენდებული range-ია.

### 5. Adam SGD-ს სჯობს (Exp 4 vs 5)
ამ ამოცანაში Adam SGD+Nesterov-ს 3.12%-ით სჯობს (66.51% vs 63.39%). მცირე dataset-ებზე Adam-ის ადაპტური learning rate-ები მნიშვნელოვან უპირატესობას იძლევა.

### 6. Residual კავშირები გადამწყვეტია (Exp 4 vs 7)
Skip კავშირებმა MediumCNN-ზე +2.5% გაუმჯობესება გამოიწვია. GAP overfitting-ს ამცირებს, AdamW + ძლიერი weight decay კი კარგი კომბინაციაა ღრმა ქსელებისთვის.

### 7. Transfer Learning ამ ამოცანაზე შეზღუდულია (Exp 7 vs 9)
Custom DeepCNN (68.99%) pretrained ResNet18-ს (64.06%) სჯობს. ძირეული მიზეზები: conv1-ის შეცვლამ pretrained ნიშნები გაანადგურა; 48×48 input ResNet-ის 224×224-ზე გათვლილ არქიტექტურასთან შეუთავსებელია. Exp 10 ამ პრობლემებს აგვარებს.

---

## WandB ლოგირება

ყველა ექსპერიმენტი დალოგილია: **[Facial-Expression-Recognition-Challenge](https://wandb.ai/akave23-free-university-of-tbilisi-/Facial-Expression-Recognition-Challenge)**

**ყოველ epoch-ზე:**
- `train/loss`, `train/accuracy`
- `val/loss`, `val/accuracy`
- `learning_rate`

**სწავლების ბოლოს:**
- `test/accuracy`
- `best_val_accuracy`

**დამატებით:**
- სრული ჰიპერპარამეტრების კონფიგურაცია `wandb.init(config=...)`-ში
- Gradient ჰისტოგრამები `wandb.watch(model, log="gradients", log_freq=200)`-ით
- ყოველი ექსპერიმენტისთვის ცალკე WandB run (სულ 10 run)
