# Facial-Expression-Recognition-Challenge
ML assignment 4
# Facial Expression Recognition Challenge (FER2013)

**Kaggle Competition:** [Challenges in Representation Learning: Facial Expression Recognition](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)

---

## პროექტის მიმოხილვა

ეს პროექტი მოიცავს FER2013 dataset-ზე სახის ემოციების კლასიფიკაციას (7 კლასი: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral). მიზანია სხვადასხვა CNN არქიტექტურისა და ჰიპერპარამეტრების სისტემატური შედარება, ამასთანავე underfitting/overfitting პრობლემების ნათელი დემონსტრირება.

ყველა ექსპერიმენტი ლოგირებულია **Weights & Biases**-ზე.
https://wandb.ai/akave23-free-university-of-tbilisi-/Facial-Expression-Recognition-Challenge?nw=nwuserakave23
---

## რეპოზიტორიის სტრუქტურა

```
├── facial-expressions.ipynb   # მთავარი Kaggle notebook
├── README.md                  # ეს ფაილი
└── checkpoints/               # შენახული საუკეთესო მოდელები (Kaggle working dir)
```

---

## Dataset

- **წყარო:** FER2013 (icml_face_data.csv)
- **შეყვანის ზომა:** 48×48 გრეისქეილი სურათები
- **კლასები:** 7 ემოცია
- **განაწილება:**

| Split        | ნიმუშების რაოდენობა |
|-------------|---------------------|
| Training    | ~28,709             |
| PublicTest  | ~3,589              |
| PrivateTest | ~3,589              |

> **შენიშვნა:** Disgust კლასი მნიშვნელოვნად არის underrepresented, რაც ართულებს მის კლასიფიკაციას.

---

## გარემო და ინსტრუმენტები

- **Framework:** PyTorch
- **გარემო:** Kaggle Notebook (GPU)
- **Experiment Tracking:** Weights & Biases (WandB)
- **Seed:** 42 (რეპროდუქციისთვის)

---

## Sanity Checks (წინასწარი შემოწმება)

ტრენინგამდე ყველა მოდელზე ჩატარდა 3 სახის შემოწმება:

| შემოწმება | რა შემოწმდა |
|-----------|------------|
| **Forward pass** | საწყისი loss ≈ ln(7) ≈ 1.946 (uniform პრედიქცია) |
| **Backward pass** | ყველა პარამეტრი იღებს gradient-ს (dead neurons არ არის) |
| **Overfit on 1 batch** | TinyNet-ი 100 ნაბიჯში ~100%-ს აღწევს ერთ batch-ზე |

---

## არქიტექტურები

### 1. TinyNet — Underfitting-ის დემონსტრირება
```
Conv(1→8) → ReLU → Pool → Conv(8→16) → ReLU → Pool → FC(2304→7)
```
- 2 conv layer, 8 და 16 ფილტრი
- **მიზანი:** მოდელი ძალიან პატარაა — გაამართლებს underfitting-ს
- **მოსალოდნელი შედეგი:** train ~35–42%, val ~35–40% (ორივე დაბალი, პატარა gap)

---

### 2. SmallCNN — Overfitting და რეგულარიზაცია
```
3 × [Conv → BN → ReLU → Pool] → Dropout(d) → FC(4608→512) → FC(512→7)
```
- `dropout=0.0` → regularization-ის გარეშე (**Exp 2**, overfitting)
- `dropout=0.4` + augmentation → რეგულარიზაციით (**Exp 3**, გაუმჯობესებული)
- **მიზანი:** ნათლად ვაჩვენოთ, რომ Dropout + augmentation პირდაპირ ასწორებს overfitting-ს

---

### 3. MediumCNN — ბალანსირებული მოდელი
```
4 × [Conv → BN → ReLU → Conv → BN → ReLU → Pool → Dropout2D] → FC → FC → FC(7)
```
- Double-conv per block, spatial Dropout2d, BatchNorm
- გატესტილია სამ ვარიანტში: Adam (**Exp 4**), SGD+Nesterov (**Exp 5**), Adam high-LR (**Exp 6**)
- **მიზანი:** ოპტიმიზატორისა და learning rate-ის სენსიტიურობის შედარება

---

### 4. DeepCNN — Residual Skip Connections
```
Stem → Stage1[ResBlock+Pool] → Stage2[Conv+ResBlock+Pool] →
Stage3[Conv+ResBlock+Pool] → Stage4[Conv+Pool] → GAP → FC → FC(7)
```
- Residual block-ები vanishing gradient-ის პრობლემის გადასაჭრელად
- Global Average Pooling (GAP) — overfitting-ს ამცირებს
- **მოსალოდნელი შედეგი:** საუკეთესო pure-CNN შედეგი

---

### 5. FERResNet — Transfer Learning (ResNet18)
```
ResNet18 (ImageNet pretrained) → Conv1 adapted (1-channel) → FC replaced (7 classes)
```
- `freeze_backbone=True` → feature extraction (**Exp 8**)
- `freeze_backbone=False` → full fine-tune, lr=0.0001 (**Exp 9**)
- **მიზანი:** ImageNet transfer-ის სარგებლიანობის გამოკვლევა სახის ემოციების ამოცნობაში

---

## ექსპერიმენტები

| # | Run სახელი | არქიტექტურა | Optimizer | LR | Aug | Dropout | Val Acc | Test Acc | დიაგნოზი |
|---|---|---|---|---|---|---|---|---|---|
| 1 | exp1_tinynet_baseline | TinyNet | Adam | 1e-3 | ❌ | 0 | 50.63% | 49.60% | **Underfitting** |
| 2 | exp2_smallcnn_no_dropout | SmallCNN | Adam | 1e-3 | ❌ | 0 | 60.24% | 60.32% | **Overfitting** |
| 3 | exp3_smallcnn_regularised | SmallCNN | Adam | 1e-3 | ✅ | 0.4 | 64.17% | 65.37% | გაუმჯობესებული |
| 4 | exp4_mediumcnn_adam | MediumCNN | Adam | 1e-3 | ✅ | 0.4 | 66.51% | 69.41% | ბალანსი |
| 5 | exp5_mediumcnn_sgd | MediumCNN | SGD | 1e-2 | ✅ | 0.4 | 63.39% | 66.09% | Optimizer-ის შედარება |
| 6 | exp6_mediumcnn_highLR | MediumCNN | Adam | 1e-2 | ✅ | 0.4 | 52.66% | 53.44% | LR ძალიან მაღალი |
| 7 | **exp7_deepcnn_residual** | DeepCNN | AdamW | 1e-3 | ✅ | 0.5 | **68.99%** | **69.63%** | **🏆 საუკეთესო** |
| 8 | exp8_resnet18_frozen | ResNet18 | Adam | 1e-3 | ✅ | — | 57.15% | 58.21% | Transfer შეზღუდული |
| 9 | exp9_resnet18_finetune | ResNet18 | AdamW | 1e-4 | ✅ | — | 64.06% | 63.81% | Transfer სრული |

---

## ექსპერიმენტების ანალიზი

### Underfitting (Exp 1 — TinyNet)
TinyNet-ს მხოლოდ 2 conv layer აქვს. train accuracy და val accuracy ორივე დაბალია (~50%), gap მცირეა. ეს კლასიკური underfitting-ია — მოდელს საკმარისი სიმძლავრე არ გააჩნია facial expression-ების სირთულის ასახვისთვის.

**გამოსავალი:** მეტი layer-ი, მეტი ფილტრი.

### Overfitting (Exp 2 — SmallCNN, dropout=0)
SmallCNN-ს საკმარისი სიმძლავრე აქვს, მაგრამ regularization გარეშე train-ზე memorize-ს — train acc >> val acc (დიდი gap). augmentation-ისა და dropout-ის გარეშე მოდელი ტრენინგ-სეტს ზეპირად ისწავლის.

**გამოსავალი:** Dropout(0.4) + augmentation.

### Regularization-ის გავლენა (Exp 3 vs Exp 2)
Dropout(0.4) და augmentation-ის დამატება (Exp 3) train–val gap-ს მნიშვნელოვნად ამცირებს. ეს ადასტურებს, რომ underfitting/overfitting-ის სწორი დიაგნოსტიკა პირდაპირ ეხმარება სწორი გამოსავლის არჩევაში.

### Optimizer-ის შედარება (Exp 4 vs Exp 5)
Adam (Exp 4) MediumCNN-ზე 66.51% val acc-ს იძლევა, SGD+Nesterov (Exp 5) კი — 63.39%-ს. Adam სწრაფად კონვერგდება, SGD უფრო ნელა. ამ dataset-სა და epoch რაოდენობაზე Adam სჯობია.

### Learning Rate-ის სენსიტიურობა (Exp 6)
Adam + lr=0.01 (Exp 6) val acc-ს 52.66%-მდე უწევს, ანუ lr=0.001-ის Exp 4-თან შედარებით 14%-ით ნაკლები. LR ძალიან მაღალია Adam-ისთვის — ტრენინგი არასტაბილურია.

### Skip Connections (Exp 7 — DeepCNN)
Residual block-ები vanishing gradient-ის პრობლემას წყვეტს. GAP (Global Average Pooling) overfitting-ს ამცირებს FC layer-ებამდე. შედეგი — 68.99% val acc, საუკეთესო მოდელი.

### Transfer Learning (Exp 8 vs Exp 9)
Frozen backbone (Exp 8) შეზღუდულია — ImageNet features-ს ემოციებისთვის ვერ ადაპტირდება. Full fine-tune (Exp 9) 64.06%-ს აღწევს, მაგრამ DeepCNN-ს ვერ სჯობია. ეს სავარაუდოდ განპირობებულია grayscale-RGB domain shift-ით.

---

## Data Augmentation

ტრენინგზე გამოყენებული ტრანსფორმაციები:
- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(10°)`
- `RandomCrop(48, padding=4)`
- `ColorJitter(brightness=0.2, contrast=0.2)`

Validation/Test-ზე მხოლოდ normalization (`mean=0.5, std=0.5`).

---

## Learning Rate Schedulers

| Scheduler | გამოიყენება |
|-----------|------------|
| `CosineAnnealingLR` | Exp 3, 4, 6, 7, 9 |
| `ReduceLROnPlateau` | Exp 8 |
| None | Exp 1, 2, 5 |

---

## WandB Tracking

თითოეული ექსპერიმენტი ლოგავს:
- `train/loss`, `train/accuracy`
- `val/loss`, `val/accuracy`
- `learning_rate` (ყოველ epoch-ზე)
- `test/accuracy`, `best_val_accuracy` (run-ის ბოლოს)
- gradient histogram-ები (`wandb.watch`, log_freq=200)

**Project:** `Facial-Expression-Recognition-Challenge`
**Entity:** `akave23-free-university-of-tbilisi-`

---

## საუკეთესო მოდელი — DeepCNN Residual

```
Val Accuracy:  68.99%
Test Accuracy: 69.63%
```

**Per-class accuracy (test set):**

| ემოცია    | სიზუსტე (მიახლოებითი) |
|-----------|-----------------------|
| Angry     | ~60%                  |
| Disgust   | ~40% (underrepresented)|
| Fear      | ~50%                  |
| Happy     | ~85%                  |
| Sad       | ~60%                  |
| Surprise  | ~75%                  |
| Neutral   | ~65%                  |

> Happy და Surprise ყველაზე კარგად კლასიფიცირდება; Disgust ყველაზე სუსტად (dataset imbalance-ის გამო).

---

## ძირითადი დასკვნები

1. **Underfitting** — TinyNet-ი ნათლად ადასტურებს, რომ სიმძლავრის ნაკლებობა ვლინდება ორივე (train + val) accuracy-ს სიდაბლეში.
2. **Overfitting** — SmallCNN (d=0) ავლენს დიდ train–val gap-ს Dropout-ისა და augmentation-ის გარეშე.
3. **Regularization მუშაობს** — Dropout(0.4) + augmentation პირდაპირ ხურავს gap-ს (Exp 2 → Exp 3).
4. **Optimizer-ი მნიშვნელოვანია** — Adam სჯობს SGD-ს ამ ამოცანაზე; LR 10x-ით გაზრდა (~1e-3→1e-2) Adam-ში accuracy-ს 14%-ით ამცირებს.
5. **Skip connections სჭირდება** — DeepCNN residual block-ებით ყველა pure-CNN-ს სჯობს (68.99%).
6. **Transfer learning სიფრთხილით** — ImageNet → grayscale emotions domain shift-ი frozen backbone-ს ზღუდავს; full fine-tune (LR=1e-4) ეხმარება, მაგრამ custom CNN-ს ვერ სჯობია.
