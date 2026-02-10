# 🎭 Celebrity Lookalike Finder

An AI-powered application that analyzes your facial features and matches you with your celebrity doppelgänger.

## Overview

Upload a photo of yourself and discover which celebrity you resemble the most! This project uses computer vision and deep learning to extract facial features and compare them against a database of celebrity faces.

## Current Status

**🚧 In Development**

### Completed
- [x] Face landmark detection using MediaPipe
- [x] 478 3D facial landmark extraction
- [x] Face blendshape analysis (52 facial expressions)
- [x] Facial transformation matrix computation
- [x] Visualization tools for landmarks and blendshapes

### Roadmap
- [ ] Build celebrity face embedding database
- [ ] Implement face embedding extraction (FaceNet/ArcFace)
- [ ] Create similarity matching algorithm
- [ ] Build web interface for photo uploads
- [ ] Deploy as web application

## Tech Stack

- **Python 3.9+**
- **MediaPipe** - Face landmark detection & mesh generation
- **OpenCV** - Image processing
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/face-recognition-ai.git
   cd face-recognition-ai
   ```

2. **Install dependencies**
   ```bash
   pip install mediapipe opencv-python numpy matplotlib
   ```

3. **Download the Face Landmarker model**
   
   Download `face_landmarker.task` from [MediaPipe Models](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) and place it in the project directory.

## Usage

### Face Landmark Detection

```python
python face_landmarker.py
```

This will:
1. Load a test image
2. Detect 478 facial landmarks
3. Display the annotated face mesh
4. Show a blendshapes analysis chart
5. Output facial transformation matrices

### Example Output

The face landmarker extracts:
- **478 3D landmarks** - Precise facial geometry points
- **52 blendshapes** - Facial expression coefficients (eye blinks, smiles, etc.)
- **Transformation matrices** - 3D face orientation data

## Project Structure

```
face-recognition-ai/
├── README.md
├── face_landmarker.py          # Main landmark detection script
├── Face Landmarker Task.task   # MediaPipe model file
├── test_image.jpg              # Sample test image
└── (future files)
    ├── embeddings/             # Celebrity face embeddings
    ├── models/                 # Trained models
    ├── app.py                  # Web application
    └── utils/                  # Helper functions
```

## How It Will Work

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Upload Photo   │────▶│  Extract Face    │────▶│  Generate       │
│                 │     │  Landmarks       │     │  Embedding      │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Display Top    │◀────│  Rank by         │◀────│  Compare with   │
│  Matches        │     │  Similarity      │     │  Celebrity DB   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Key Features (Planned)

| Feature | Description |
|---------|-------------|
| **Multi-angle support** | Works with various face angles |
| **Expression invariant** | Matches regardless of facial expression |
| **Top-N matches** | Returns multiple celebrity matches with confidence scores |
| **Fast inference** | Real-time matching capability |

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## License

MIT License - feel free to use this project for personal or commercial purposes.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) by Google for face landmark detection
- Future: Celebrity dataset sources

---
