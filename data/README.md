# Dataset Directory

Place both datasets here.

## 1. Kvasir-Capsule (imbalanced — main experiments)

1. Go to [https://osf.io/dv2ag/](https://osf.io/dv2ag/)
2. Download the **labeled images** archive
3. Extract so the structure looks like:

```
data/
└── labeled-images/
    ├── Angioectasia/
    ├── Blood - Fresh/
    ├── Blood - Hematin/
    ├── Erosion/
    ├── Erythema/
    ├── Foreign Body/
    ├── Ileocecal Valve/
    ├── Lymphangiectasia/
    ├── Normal clean mucosa/
    ├── Polyp/
    ├── Pylorus/
    ├── Reduced Mucosal View/
    └── Ulcer/
```

## 2. KVASIR v2 (balanced — gold standard comparison)

1. Go to [https://datasets.simula.no/kvasir/](https://datasets.simula.no/kvasir/)
2. Download the **kvasir-dataset-v2** archive
3. Extract so the structure looks like:

```
data/
└── kvasir-dataset-v2/
    ├── dyed-lifted-polyps/
    ├── dyed-resection-margins/
    ├── esophagitis/
    ├── normal-cecum/
    ├── normal-pylorus/
    ├── normal-z-line/
    ├── polyps/
    └── ulcerative-colitis/
```

> Both dataset folders are git-ignored since they're too large for version control.
