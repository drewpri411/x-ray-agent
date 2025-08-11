# Manual Download Instructions for Chest X-Ray Images

## 📁 Folder Structure
```
data/samples/
├── normal/           (5-10 normal chest X-rays)
├── pneumonia/        (5-10 pneumonia cases)
├── pleural_effusion/ (5-10 pleural effusion cases)
├── atelectasis/      (5-10 atelectasis cases)
└── cardiomegaly/     (5-10 cardiomegaly cases)
```

## 🩺 Download Sources

### **Option 1: Kaggle Dataset (Recommended - Easiest)**

#### **Step 1: Download from Kaggle**
1. **Go to**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. **Sign up**: Create free Kaggle account if needed
3. **Download**: Click "Download" button (1.2GB ZIP file)
4. **Extract**: Extract the ZIP file to a temporary folder

#### **Step 2: Copy Images**
After extraction, you'll have this structure:
```
chest_xray/
├── train/
│   ├── NORMAL/          (1,349 normal X-rays)
│   └── PNEUMONIA/       (3,875 pneumonia X-rays)
└── test/
    ├── NORMAL/          (234 normal X-rays)
    └── PNEUMONIA/       (390 pneumonia X-rays)
```

**Copy 5-10 images from each category:**
```bash
# Copy normal images
copy "chest_xray\train\NORMAL\*.jpeg" "data\samples\normal\"

# Copy pneumonia images
copy "chest_xray\train\PNEUMONIA\*.jpeg" "data\samples\pneumonia\"
```

### **Option 2: NIH Chest X-Ray Dataset**

#### **Step 1: Download from Kaggle**
1. **Go to**: https://www.kaggle.com/datasets/nih-chest-xrays/data
2. **Download**: Click "Download" (112,000 images - download subset)
3. **Extract**: Extract the ZIP file

#### **Step 2: Organize by Pathology**
The NIH dataset has multiple pathologies. Copy images to appropriate folders:
- **Normal**: No significant findings
- **Pneumonia**: Lung infection
- **Pleural Effusion**: Fluid around lungs
- **Atelectasis**: Collapsed lung tissue
- **Cardiomegaly**: Enlarged heart

### **Option 3: Manual Download from Medical Sites**

#### **Step 1: Download Individual Images**

**For each pathology, search and download 5-10 images:**

1. **Normal Chest X-rays**:
   - Search: "normal chest X-ray PA view"
   - Sources: Radiopaedia.org, Radiographics.rsna.org
   - Save as: `normal_1.jpg`, `normal_2.jpg`, etc.

2. **Pneumonia Cases**:
   - Search: "pneumonia chest X-ray consolidation"
   - Sources: Medical image repositories
   - Save as: `pneumonia_1.jpg`, `pneumonia_2.jpg`, etc.

3. **Pleural Effusion**:
   - Search: "pleural effusion chest X-ray"
   - Sources: Medical databases
   - Save as: `pleural_1.jpg`, `pleural_2.jpg`, etc.

4. **Atelectasis**:
   - Search: "atelectasis chest X-ray"
   - Sources: Medical image sites
   - Save as: `atelectasis_1.jpg`, `atelectasis_2.jpg`, etc.

5. **Cardiomegaly**:
   - Search: "cardiomegaly chest X-ray"
   - Sources: Medical repositories
   - Save as: `cardiomegaly_1.jpg`, `cardiomegaly_2.jpg`, etc.

#### **Step 2: Recommended Sources**
- **Radiopaedia.org** - Free medical image database
- **Radiographics.rsna.org** - Professional radiology images
- **PubMed Central** - Medical research images
- **Medical image repositories** - Various medical databases

## 📋 File Requirements

### **Image Format**
- **Format**: JPG, JPEG, or PNG
- **Size**: Any size (system will resize automatically)
- **Quality**: Clear, readable chest X-rays
- **Orientation**: Standard PA (posteroanterior) view preferred

### **Naming Convention**
```
normal/
├── normal_1.jpg
├── normal_2.jpg
└── ...

pneumonia/
├── pneumonia_1.jpg
├── pneumonia_2.jpg
└── ...

pleural_effusion/
├── pleural_1.jpg
├── pleural_2.jpg
└── ...

atelectasis/
├── atelectasis_1.jpg
├── atelectasis_2.jpg
└── ...

cardiomegaly/
├── cardiomegaly_1.jpg
├── cardiomegaly_2.jpg
└── ...
```

## 🧪 Testing Commands

### **After downloading images, test with:**

```bash
# Test normal case
python main.py --image data/samples/normal/normal_1.jpg --symptoms "Patient has no symptoms"

# Test pneumonia case
python main.py --image data/samples/pneumonia/pneumonia_1.jpg --symptoms "Patient has cough and fever"

# Test pleural effusion case
python main.py --image data/samples/pleural_effusion/pleural_1.jpg --symptoms "Patient has shortness of breath"

# Test atelectasis case
python main.py --image data/samples/atelectasis/atelectasis_1.jpg --symptoms "Patient has chest pain"

# Test cardiomegaly case
python main.py --image data/samples/cardiomegaly/cardiomegaly_1.jpg --symptoms "Patient has fatigue and swelling"
```

### **Web Interface Testing**
```bash
# Start web interface
streamlit run ui/app_streamlit.py

# Then go to: http://localhost:8501
# Upload images and test interactively
```

## 📊 Expected Results

### **What You'll Get**
- **Pathology detection** for 6 key conditions
- **Confidence scores** (0.0 to 1.0)
- **Triage assessment** (urgent, routine, etc.)
- **Medical recommendations**
- **Comprehensive report** (with API key)

### **Sample Output**
```
📊 Analysis Results:
- Pneumonia: 0.85 (HIGH confidence)
- Pleural Effusion: 0.12 (LOW confidence)
- Normal: 0.03 (LOW confidence)

Triage: URGENT - High probability of pneumonia
Recommendations: Immediate medical attention required
```

## ⚠️ Important Notes

### **Privacy & Ethics**
- ✅ **Public datasets** are safe to use
- ✅ **No patient identifiers** in public datasets
- ✅ **Research use** is permitted
- ⚠️ **Don't use** private patient data without consent
- ⚠️ **Follow** dataset terms of use

### **Testing Tips**
- **Start with 5 images** per category for initial testing
- **Use clear, high-quality** chest X-rays
- **Test both normal and abnormal** cases
- **Verify image format** (JPG/PNG)
- **Check file permissions** (readable by system)

## 🎯 Quick Start

1. **Download Kaggle dataset** (Option 1 - easiest)
2. **Copy 5 images** to each folder
3. **Test with web interface**: http://localhost:8501
4. **Verify results** and adjust as needed

The AI Radiology Assistant will analyze each image and provide comprehensive medical insights! 