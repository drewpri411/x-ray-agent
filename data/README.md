# Data Directory

This directory is for storing sample chest X-ray images and reference data for the AI Radiology Assistant.

## Sample Images

Place chest X-ray images in this directory for testing and demonstration purposes. Supported formats:
- JPG/JPEG
- PNG
- BMP
- TIFF/TIF

## Important Notes

1. **Privacy and Ethics**: Only use de-identified, publicly available chest X-ray images for testing
2. **Sample Sources**: Consider using images from:
   - NIH ChestX-ray14 dataset (public domain)
   - CheXpert dataset (with proper attribution)
   - MIMIC-CXR dataset (requires access approval)
   - Other publicly available medical imaging datasets

3. **File Naming**: Use descriptive names like:
   - `normal_chest_xray.jpg`
   - `pneumonia_case_1.png`
   - `effusion_sample.tiff`

## Example Usage

```bash
# Copy a sample image to this directory
cp /path/to/sample_xray.jpg data/

# Run analysis on the sample
python main.py --image data/sample_xray.jpg --symptoms "Patient has cough and fever"
```

## Disclaimer

This is a prototype system for educational purposes only. All images should be properly de-identified and used in accordance with relevant privacy regulations and dataset terms of use. 