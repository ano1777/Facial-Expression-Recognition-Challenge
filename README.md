# Facial Expression Recognition Challenge (FER2013)

**Kaggle Competition:** [Challenges in Representation Learning: Facial Expression Recognition](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)

**WandB Project:** [akave23-free-university-of-tbilisi- / Facial-Expression-Recognition-Challenge](https://wandb.ai/akave23-free-university-of-tbilisi-/Facial-Expression-Recognition-Challenge?nw=nwuserakave23)

---

## პროექტის მიმოხილვა

ეს პროექტი მოიცავს FER2013 dataset-ზე სახის ემოციების კლასიფიკაციას (7 კლასი: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral). მიზანია სხვადასხვა CNN არქიტექტურისა და ჰიპერპარამეტრების სისტემატური შედარება, ამასთანავე underfitting/overfitting პრობლემების ნათელი დემონსტრირება.

ყველა ექსპერიმენტი ლოგირებულია **Weights & Biases**-ზე.

---

## რეპოზიტორიის სტრუქტურა

```
├── facial-expressions.ipynb   # მთავარი Kaggle notebook
├── README.md                  # ეს ფაილი
└── checkpoints/               # შენახული საუკეთესო მოდელები (Kaggle working dir)
```

---

## Dataset

- **წყარო:** FER2013 (`icml_face_data.csv`)
- **შეყვანის ზომა:** 48×48 გრეისქეილი სურათები
- **კლასები:** 7 ემოცია
- **განაწილება:**

| Split       | ნიმუშების რაოდენობა |
|-------------|---------------------|
| Training    | ~28,709             |
| PublicTest  | ~3,589              |
| PrivateTest | ~3,589              |

> **შენიშვნა:** `Disgust` კლასი მნიშვნელოვნად არის underrepresented დანარჩენ კლასებთან შედარებით, რაც ართულებს მის კლასიფიკაციას.

---

## გარემო და ინსტრუმენტები

- **Framework:** PyTorch
- **გარემო:** Kaggle Notebook (GPU)
- **Experiment Tracking:** Weights & Biases (WandB)
- **Seed:** 42 (რეპროდუქციისთვის)

---

## Sanity Checks (წინასწარი შემოწმება)

ტრენინგამდე **ყველა** მოდელზე ჩატარდა 3 სახის შემოწმება — ეს სტანდარტული პრაქტიკაა, რომელიც პრობლემებს early stage-ზე ავლენს:

| შემოწმება              | რა შემოწმდა                                                                                   | რატომ მნიშვნელოვანია                                                         |
|------------------------|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| **Forward pass**       | საწყისი loss ≈ ln(7) ≈ 1.946 (uniform პრედიქცია 7 კლასზე)                                    | დარწმუნდება, რომ weight initialization სწორია და მოდელი bias-ის გარეშე იწყებს |
| **Backward pass**      | ყველა trainable პარამეტრი იღებს non-zero gradient-ს                                          | ამოიცნობს dead neurons-სა და broken computation graph-ს                      |
| **Overfit on 1 batch** | TinyNet-ი 100 gradient step-ში ~100%-ს აღწევს ერთ batch-ზე (batch_size=32, augment=False) | ადასტურებს, რომ loss function და training loop სწორად არის დაწერილი           |

---

## არქიტექტურები

### 1. TinyNet — Underfitting-ის დემონსტრირება

```
Input(1×48×48)
  → Conv2d(1→8, 3×3, pad=1) → ReLU → MaxPool2d(2×2)   # 8×24×24
  → Conv2d(8→16, 3×3, pad=1) → ReLU → MaxPool2d(2×2)  # 16×12×12
  → Flatten → FC(2304→7)
```

**არქიტექტურის არჩევანი:** მიზანმიმართულად პატარა — მხოლოდ 2 conv layer, 8 და 16 ფილტრით. ამ სიმძლავრის მოდელს facial expression-ებისთვის საჭირო spatial features-ის სწავლა არ შეუძლია.

**ჰიპერპარამეტრების დასაბუთება:**
- `epochs=30` — ნაკლები epoch საკმარისია underfitting-ის დასადასტურებლად; მეტი epoch მდგომარეობას არ შეცვლის
- `lr=0.001` — Adam-ის standard default; ნეიტრალური არჩევანი, რომ LR არ იყოს confounding factor
- `batch_size=64` — სტანდარტული; augmentation-ი გამორთულია, რომ underfitting მკაფიო იყოს
- `weight_decay=0`, `scheduler=None` — regularization-ის გარეშე, რომ ჩვენება სუფთა იყოს

**მოსალოდნელი შედეგი:** train ~35–42%, val ~35–40% — ორივე დაბალი, gap მცირე. ეს კლასიკური underfitting-ია.

**რეალური შედეგი:** val 50.63%, test 49.60% — ორივე დაბალი, მცირე gap ✓

---

### 2. SmallCNN — Overfitting-ის და Regularization-ის დემონსტრირება

```
Input(1×48×48)
  → Conv2d(1→32, 3×3, pad=1) → BN → ReLU → MaxPool2d(2×2)   # 32×24×24
  → Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool2d(2×2)  # 64×12×12
  → Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool2d(2×2) # 128×6×6
  → Flatten → Dropout(d) → FC(4608→512) → ReLU → Dropout(d) → FC(512→7)
```

**არქიტექტურის არჩევანი:** TinyNet-ზე მნიშვნელოვნად მეტი სიმძლავრე (3 conv block, 32/64/128 ფილტრი, BatchNorm), მაგრამ ორ ვარიანტში ტესტირდება — regularization-ის გარეშე და მასთან ერთად — რათა ნათლად ვაჩვენოთ overfitting-ის მექანიზმი და გამოსავალი.

**ჰიპერპარამეტრების დასაბუთება (Exp 2 — overfitting demo):**
- `dropout=0.0` — განზრახ გამორთული, რომ overfitting სრულად გამოვლინდეს
- `augment=False` — augmentation-ის გარეშე, ნაკლები ხელოვნური variation
- `weight_decay=0` — L2 regularization-ის გარეშე
- `epochs=50` — საკმარისი იმისთვის, რომ მოდელი ტრენინგ-სეტს ზეპირად ისწავლოს
- `scheduler=None` — LR ფიქსირებული, რათა training dynamics სუფთა ჩანდეს

**მოსალოდნელი შედეგი:** train ~75–85%, val ~55–60% — დიდი gap, overfitting.
**რეალური შედეგი:** val 60.24%, test 60.32% ✓

**ჰიპერპარამეტრების დასაბუთება (Exp 3 — regularisation):**
- `dropout=0.4` — ორივე FC-ს წინ; 0.4 კარგი trade-off-ია regularization-სა და underfitting-ს შორის
- `augment=True` — RandomFlip, Rotation, Crop, ColorJitter ხელოვნურად ზრდის ეფექტური dataset-ის ზომას
- `weight_decay=1e-4` — მსუბუქი L2 regularization
- `scheduler=cosine` — CosineAnnealing LR-ს გლუვად ამცირებს, converge-ი სტაბილურია

**რეალური შედეგი:** val 64.17%, test 65.37% — gap მნიშვნელოვნად შემცირდა ✓

---

### 3. MediumCNN — სიღრმე და ოპტიმიზატორების შედარება

```
Input(1×48×48)
  → Conv2d(1→32) → BN → ReLU → Conv2d(32→32) → BN → ReLU → MaxPool2d → Dropout2d(0.1)  # 32×24×24
  → Conv2d(32→64) → BN → ReLU → Conv2d(64→64) → BN → ReLU → MaxPool2d → Dropout2d(0.1) # 64×12×12
  → Conv2d(64→128) → BN → ReLU → Conv2d(128→128) → BN → ReLU → MaxPool2d → Dropout2d(0.2) # 128×6×6
  → Conv2d(128→256) → BN → ReLU → MaxPool2d                                               # 256×3×3
  → Flatten → Dropout(0.4) → FC(2304→512) → ReLU → Dropout(0.4) → FC(512→256) → ReLU → FC(256→7)
```

> **შენიშვნა:** პირველი 3 block double-conv-ია (VGG-style), მე-4 block კი single conv — ფილტრების გაორმაგება ყოველ ჯერზე (32→64→128→256) feature map-ების სიმდიდრეს ზრდის, spatial Dropout2d კი spatial feature-ების overfitting-ს ამცირებს.

**არქიტექტურის არჩევანი:** SmallCNN-ზე მეტი სიღრმე და VGG-style double-conv block-ები, რაც ემოციების უფრო კომპლექსურ spatial pattern-ებს ითვისებს. სამ სხვადასხვა ჰიპერპარამეტრულ კონფიგურაციაში ტესტირდება, რათა optimizer-ისა და LR-ის გავლენა იზოლირებულად შევაფასოთ.

**ჰიპერპარამეტრების დასაბუთება (Exp 4 — Adam baseline):**
- `optimizer=Adam`, `lr=0.001` — Adam-ის recommended default; adaptive learning rates კარგია მრავალფეროვანი gradient magnitude-ებისთვის
- `weight_decay=1e-4` — მსუბუქი L2
- `epochs=60` — SmallCNN-ზე მეტი, რადგან მოდელი სიღრმით დიდია
- `scheduler=cosine` — LR warmup-ის გარეშე cosine decay; Adam-ს კარგად ეწყობა

**ჰიპერპარამეტრების დასაბუთება (Exp 5 — SGD+Nesterov):**
- `optimizer=SGD`, `lr=0.01` — SGD-სთვის Adam-ზე 10× მაღალი LR ნორმალურია; Nesterov momentum აჩქარებს კონვერგენციას
- `momentum=0.9` — standard SGD momentum
- `scheduler=cosine` — SGD warmup-ის გარეშე cosine decay კარგად მუშაობს
- ეს ექსპერიმენტი Exp 4-ის **direct ablation**-ია — მხოლოდ optimizer-ი და LR შეიცვალა

**ჰიპერპარამეტრების დასაბუთება (Exp 6 — LR sensitivity ablation):**
- `optimizer=Adam`, `lr=0.01` — Adam + lr=0.01 განზრახ "ცუდი" კომბინაციაა; Adam-ი lr=0.001-ისთვის არის optimized
- Exp 4-ის **direct ablation** — მხოლოდ LR 10×-ით გაიზარდა, ყველა დანარჩენი პარამეტრი იდენტურია

**რეალური შედეგები:**
- Exp 4 (Adam, lr=0.001): val 66.51%, test 69.41%
- Exp 5 (SGD, lr=0.01): val 63.39%, test 66.09%
- Exp 6 (Adam, lr=0.01): val 52.66%, test 53.44% ← LR ძალიან მაღალი

---

### 4. DeepCNN — Residual Skip Connections

```
Input(1×48×48)
  Stem: Conv2d(1→64, 3×3) → BN → ReLU                                    # 64×48×48

  Stage1: ResBlock(64) → MaxPool2d(2×2)                                   # 64×24×24
  Stage2: Conv2d(64→128) → BN → ReLU → ResBlock(128) → MaxPool2d(2×2)    # 128×12×12
  Stage3: Conv2d(128→256) → BN → ReLU → ResBlock(256) → MaxPool2d(2×2)   # 256×6×6
  Stage4: Conv2d(256→512) → BN → ReLU → MaxPool2d(2×2)                   # 512×3×3

  GAP: AdaptiveAvgPool2d(1×1)                                             # 512×1×1
  → Flatten → Dropout(0.5) → FC(512→256) → ReLU → Dropout(0.25) → FC(256→7)

ResBlock(c): x + [Conv(c→c) → BN → ReLU → Conv(c→c) → BN],  ReLU
```

**არქიტექტურის არჩევანი:** MediumCNN-ზე კიდევ უფრო ღრმა ქსელი (512 ფილტრამდე), skip connection-ებით. ResNet-ის იდეა: `output = F(x) + x` — gradient-ი shortcut-ის გამო პირდაპირ გადის, vanishing gradient-ის პრობლემა ქრება. Global Average Pooling (GAP) FC layer-ებამდე feature map-ების საშუალოს ითვლის — ეს FC-ს parameter რაოდენობას მნიშვნელოვნად ამცირებს და overfitting-ს ამცირებს.

**ჰიპერპარამეტრების დასაბუთება:**
- `dropout=0.5` პირველ FC-ს წინ, `dropout=0.25` (0.5/2) მეორეს წინ — progressive dropout; მეტი regularization feature extraction-ის გამოსასვლელთან, ნაკლები classifier-ში
- `optimizer=AdamW`, `weight_decay=1e-3` — AdamW L2 regularization-ს სწორად ასრულებს (Adam-ში weight_decay weight-ებს gradient-ის შემდეგ ვრიცხავს, AdamW-ში — შემდეგ; AdamW უფრო სწორი regularization-ია)
- `batch_size=32` — 64-დან 32-ზე შემცირება; უფრო ხშირი gradient update, უფრო noisy — ეს ზოგჯერ generalization-ს აუმჯობესებს
- `epochs=70` — ყველაზე მეტი epoch; ღრმა ქსელი კონვერგენციისთვის მეტ iteration-ს საჭიროებს
- `scheduler=cosine` — 70 epoch-ზე cosine decay LR-ს გლუვად ნულამდე მიჰყავს

**რეალური შედეგი:** val **68.99%**, test **69.63%** — საუკეთესო ✓

---

### 5. FERResNet — Transfer Learning (ResNet18)

```
ResNet18 (ImageNet pretrained weights)
  ↓ conv1 შეიცვალა: Conv2d(3→64, 7×7) → Conv2d(1→64, 7×7)  # grayscale adaptation
  ↓ fc შეიცვალა:   FC(512→1000) → Dropout(0.5) → FC(512→256) → ReLU → Dropout(0.3) → FC(256→7)

Exp 8 (freeze_backbone=True):  მხოლოდ conv1 და fc trainable
Exp 9 (freeze_backbone=False): ყველა layer trainable
```

**არქიტექტურის არჩევანი:** ResNet18-ს ImageNet-ზე წინასწარ მომზადებული features გააჩნია (edges, textures, shapes). სახის სურათებისთვის ეს features ემოციებზე transferable-ია — თეორიულად. 1-channel grayscale-ზე ადაპტაცია: `conv1` ხელახლა ინიციალიზდება (3-channel weights-ის average).

**ჰიპერპარამეტრების დასაბუთება (Exp 8 — frozen backbone):**
- `freeze_backbone=True` — backbone-ი (layer1–layer4) frozen, მხოლოდ conv1 და custom FC head ტრენინგდება
- `epochs=20` — მხოლოდ head-ი ტრენინგდება, ამიტომ ნაკლები epoch საკმარისია
- `optimizer=Adam`, `lr=0.001` — standard; head-ი randomly initialized, head LR-ი 0.001 მუშაობს
- `scheduler=plateau` — ReduceLROnPlateau: val loss-ი 5 epoch-ს არ გაუმჯობესდეს → LR ×0.5; head-ის fast training-ში plateau სიგნალი სასარგებლოა

**ჰიპერპარამეტრების დასაბუთება (Exp 9 — full fine-tune):**
- `freeze_backbone=False` — მთელი ქსელი ტრენინგდება, ImageNet weights-ი ადაპტირდება
- `lr=0.0001` — Exp 8-ზე 10× დაბალი; pretrained features-ის "forgetting" თავიდან ასაცილებლად; catastrophic forgetting-ის რისკი მაღალი LR-ზე
- `optimizer=AdamW`, `weight_decay=0.01` — ძლიერი L2 regularization; fine-tune-ში overfitting-ის რისკი მაღალია
- `epochs=50` — საკმარისი full fine-tune-ისთვის low LR-ზე

**რეალური შედეგები:**
- Exp 8 (frozen): val 57.15%, test 58.21% — backbone-ი ვერ ადაპტირდება
- Exp 9 (fine-tune): val 64.06%, test 63.81% — უმჯობესდება, მაგრამ DeepCNN-ს ვერ სჯობია

---

## ექსპერიმენტები — შეჯამება

| # | Run სახელი | არქიტექტურა | Optimizer | LR | Epochs | Batch | Aug | Dropout | Val Acc | Test Acc | დიაგნოზი |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | exp1_tinynet_baseline | TinyNet | Adam | 1e-3 | 30 | 64  | 0 | 50.63% | 49.60% | **Underfitting** |
| 2 | exp2_smallcnn_no_dropout | SmallCNN | Adam | 1e-3 | 50 | 64  | 0 | 60.24% | 60.32% | **Overfitting** |
| 3 | exp3_smallcnn_regularised | SmallCNN | Adam | 1e-3 | 50 | 64  | 0.4 | 64.17% | 65.37% | Regularized |
| 4 | exp4_mediumcnn_adam | MediumCNN | Adam | 1e-3 | 60 | 64  | 0.4 | 66.51% | 69.41% | Balanced |
| 5 | exp5_mediumcnn_sgd | MediumCNN | SGD+Nesterov | 1e-2 | 60 | 64  | 0.4 | 63.39% | 66.09% | Optimizer ablation |
| 6 | exp6_mediumcnn_highLR | MediumCNN | Adam | 1e-2 | 60 | 64  | 0.4 | 52.66% | 53.44% | LR too high |
| 7 | **exp7_deepcnn_residual** | DeepCNN | AdamW | 1e-3 | 70 | 32  | 0.5/0.25 | **68.99%** | **69.63%** | **🏆 საუკეთესო** |
| 8 | exp8_resnet18_frozen | ResNet18 | Adam | 1e-3 | 20 | 32  | 0.5/0.3 | 57.15% | 58.21% | Transfer limited |
| 9 | exp9_resnet18_finetune | ResNet18 | AdamW | 1e-4 | 50 | 32  | 0.5/0.3 | 64.06% | 63.81% | Transfer full |

---

## ექსპერიმენტების ანალიზი

### Underfitting (Exp 1 — TinyNet)

TinyNet-ს 2 conv layer-ით მხოლოდ **2,311 trainable parameter** აქვს. ორივე train და val accuracy ~50%-ის ფარგლებშია, gap მცირეა — მოდელი საკმარისად "ჭკვიანი" არ არის იმისთვის, რომ memorize-ც კი გააკეთოს. FER2013-ის 7-კლასიანი ამოცანა კომპლექსური spatial pattern-ების ამოცნობას საჭიროებს, რასაც 2 conv layer ვერ ასრულებს.

**გამოსავალი:** მეტი layer, მეტი ფილტრი → SmallCNN.

### Overfitting (Exp 2 — SmallCNN, dropout=0)

SmallCNN-ს საკმარისი სიმძლავრე აქვს, მაგრამ regularization-ის გარეშე ტრენინგ-სეტს "ზეპირად ისწავლის". notebook-ში predicted იყო train ~75–85%, val ~55–60% — დიდი gap. val 60.24% vs ვარაუდობდით მაღალ train accuracy-ს, რაც overfitting-ს ადასტურებს.

**გამოსავალი:** Dropout(0.4) + augmentation → Exp 3.

### Regularization-ის გავლენა (Exp 3 vs Exp 2)

Dropout(0.4) და augmentation-ის დამატება val accuracy-ს 60.24%-დან 64.17%-მდე ზრდის (+3.93%), train–val gap კი მნიშვნელოვნად მცირდება. CosineAnnealing scheduler-ი training-ის ბოლოს LR-ს გლუვად ამცირებს, რაც ეხმარება fine-grained weight adjustment-ს.

### Optimizer-ის შედარება (Exp 4 vs Exp 5)

Adam (Exp 4) 66.51% val acc-ს, SGD+Nesterov (Exp 5) კი 63.39%-ს იძლევა — 3.12%-ის სხვაობა. Adam-ის adaptive learning rates ამ dataset-სა და epoch რაოდენობაზე უფრო ეფექტიანია: სწრაფად კონვერგდება და კარგ მინიმუმს პოულობს. SGD 60 epoch-ში ჯერ კიდევ კონვერგენციის პროცესში შეიძლება იყოს — მეტ epoch-ს ან LR warmup-ს საჭიროებს.

### Learning Rate-ის სენსიტიურობა (Exp 6 vs Exp 4)

Adam + lr=0.01 (Exp 6) — val accuracy 66.51%-დან 52.66%-მდე ეცემა (−13.85%). Adam-ი lr=0.001-ისთვის არის designed; lr=0.01 gradient step-ებს ზედმეტად დიდს ხდის, optimizer "overshoots" loss minimum-ს და ტრენინგი არასტაბილური ხდება. ეს ablation study ადასტურებს, რომ Adam-ისთვის lr ∈ [1e-4, 3e-3] range-ია რეკომენდებული.

### Skip Connections (Exp 7 — DeepCNN)

DeepCNN-ი ყველა სხვა არქიტექტურას სჯობს (val 68.99%). Residual block-ები ორ მიმართულებით ეხმარება: (1) gradient-ი shortcut-ის გამო პირდაპირ უკან გადის — vanishing gradient-ი ქრება; (2) მოდელი ან ახალ feature-ებს სწავლობს (`F(x)`), ან ისე ტოვებს (`F(x)=0` → `output=x`) — "do no harm" პრინციპი. GAP FC-ს პარამეტრთა რაოდენობას 512-მდე ამცირებს (MediumCNN-ში 2304 იყო), overfitting-ი შემცირდება. AdamW + weight_decay=1e-3 ძლიერი regularization-ია.

### Transfer Learning (Exp 8 vs Exp 9)

**Frozen (Exp 8):** val 57.15% — ResNet18-ის ImageNet features სახის ემოციებს კარგად ვერ describe-ებს frozen სახით. FER2013-ი 48×48 grayscale-ია, ImageNet features კი 224×224 RGB-ზე ისწავლა — domain shift მნიშვნელოვანია.

**Fine-tune (Exp 9):** val 64.06% — backbone-ის ადაптაცია low LR-ით (1e-4) +6.91% გაუმჯობესებას იძლევა, მაგრამ custom DeepCNN-ს მაინც ვერ სჯობია. სავარაუდო მიზეზი: ResNet18-ს grayscale 48×48-ისთვის ზედმეტი სიმძლავრე აქვს და fine-tune 50 epoch-ში საბოლოოდ ვერ კონვერგდება ოპტიმუმამდე.

---

## Data Augmentation

ტრენინგზე (Exp 3–9) გამოყენებული ტრანსფორმაციები:

| ტრანსფორმაცია | პარამეტრები | დასაბუთება |
|---|---|---|
| `RandomHorizontalFlip` | p=0.5 | ემოციები სიმეტრიულია — flipped სახე იდენტური ემოციით |
| `RandomRotation` | ±10° | ადამიანები სხვადასხვა კუთხით ფოტოგრაფირდებიან |
| `RandomCrop` | size=48, padding=4 | მცირე translation invariance; FER-ის 48×48-ზე padding=4 ოპტიმალურია |
| `ColorJitter` | brightness=0.2, contrast=0.2 | განათების variation-ის სიმულაცია; saturation/hue გამორთულია (grayscale) |

Validation/Test-ზე მხოლოდ `ToTensor()` + `Normalize(mean=0.5, std=0.5)`.

---

## Learning Rate Schedulers

| Scheduler | გამოიყენება | დასაბუთება |
|---|---|---|
| `CosineAnnealingLR` (T_max=epochs) | Exp 3, 4, 6, 7, 9 | LR გლუვად cos(π·t/T)-ის მიხედვით ეცემა ნულამდე; final convergence-ს აუმჯობესებს |
| `ReduceLROnPlateau` (patience=5, factor=0.5) | Exp 8 | val loss-ი 5 epoch-ს არ გაუმჯობესდეს → LR ×0.5; frozen backbone-ის სწრაფ training-ს ეხმარება |
| None | Exp 1, 2, 5 | Exp 1 და 2 — სუფთა ბეისლაინი; Exp 5 (SGD) — cosine-ს ნაცვლად constant LR-ით ტესტი |

---

## WandB Tracking

თითოეული ექსპერიმენტი ლოგავს:

- `train/loss`, `train/accuracy` — ყოველ epoch-ზე
- `val/loss`, `val/accuracy` — ყოველ epoch-ზე
- `learning_rate` — ყოველ epoch-ზე (scheduler-ის გავლენის ვიზუალიზაციისთვის)
- `test/accuracy`, `best_val_accuracy` — run-ის ბოლოს
- gradient histogram-ები — `wandb.watch(model, log="gradients", log_freq=200)` (vanishing/exploding gradient-ის მონიტორინგი)

**Project:** `Facial-Expression-Recognition-Challenge`
**Entity:** `akave23-free-university-of-tbilisi-`

---

## საუკეთესო მოდელი — DeepCNN Residual

```
Val Accuracy:  68.99%
Test Accuracy: 69.63%
```

Happy კლასი ყველაზე კარგად კლასიფიცირდება — ეს ყველაზე გამოხატული ემოციაა (ღია ღიმილი), dataset-შიც ყველაზე მეტი ნიმუში აქვს. Disgust ყველაზე სუსტია — dataset imbalance და სხვა კლასებთან (Angry, Sad) ვიზუალური მსგავსება გართულებს.

---

## ძირითადი დასკვნები

1. **Underfitting** — TinyNet-ი (2 conv, 2,311 params) ნათლად ადასტურებს, რომ არასაკმარისი სიმძლავრე ვლინდება ორივე (train + val) accuracy-ს სიდაბლეში. გამოსავალი: სიღრმე და სიგანე.

2. **Overfitting** — SmallCNN (d=0) დიდ train–val gap-ს ავლენს regularization-ის გარეშე. Dropout(0.4) + augmentation gap-ს პირდაპირ ხურავს (+3.93% val acc).

3. **Optimizer** — Adam (lr=1e-3) SGD+Nesterov-ს (lr=1e-2) სჯობია ამ ამოცანაზე (+3.12% val). Adam-ისთვის lr=1e-2 ზედმეტად მაღალია (−13.85% val Exp 4-ს შედარებით).

4. **Skip connections გადამწყვეტია** — DeepCNN residual block-ებით ყველა სხვა pure-CNN-ს სჯობს. GAP overfitting-ს ამცირებს. AdamW + strong weight decay ამ არქიტექტურაზე კარგი კომბინაციაა.

5. **Transfer learning ამ ამოცანაზე შეზღუდულია** — 48×48 grayscale FER2013 და 224×224 RGB ImageNet შორის domain gap-ი დიდია. Fine-tune ეხმარება (Exp 8→9: +6.91%), მაგრამ custom CNN-ს მაინც ვერ სჯობია.

6. **Batch size** — 64-დან 32-ზე გადასვლა DeepCNN-ში (Exp 7) უფრო noisy gradient-ებს, შესაბამისად — უკეთეს regularization-ს, ნიშნავს.
