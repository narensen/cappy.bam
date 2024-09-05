# Image Captioning
This repository contains an implementation of an Image Captioning model that utilizes attention mechanisms for processing captions for images. The model is built using PyTorch and incorporates an EfficientNet encoder and a Transformer-based decoder.

Image captioning is the process of generating textual descriptions for images. It combines techniques from computer vision and natural language processing to provide meaningful captions. This project uses a combination of an EfficientNet encoder for feature extraction and a Transformer-based decoder for generating captions, with an attention mechanism to focus on different parts of the image during caption generation.

## Features

- EfficientNet-based encoder for image feature extraction.
- Transformer-based decoder with attention mechanisms.
- Supports custom datasets for training.
- Uses COCO dataset for training and evaluation.
- CUDA support for efficient training on GPUs.

## Installation

### Prerequisites

- Python 
- PyTorch
- torchvision
- transformers
- CUDA-enabled GPU (optional, but recommended)

### Clone the Repository

```bash
git clone https://github.com/narensen/cappy.bam.git
cd cappy.bam
```

## Usage

### Inference
To generate captions for an image:
```
from attentions import ImageCaptioningModel, tokenizer

# Load the model
model = ImageCaptioningModel()
model.load_state_dict(torch.load('path/to/model_weights.pth'))

# Load and preprocess the image
image = load_image('path/to/image.jpg')

# Generate caption
caption = model.generate_caption(image)
print("Generated Caption:", caption)
```

