import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertModel
from efficientnet_pytorch import EfficientNet

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_rate):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_rate, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_rate, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class AttentionEfficientNet(nn.Module):
    def __init__(self, pretrained_model, channel_attention, spatial_attention):
        super(AttentionEfficientNet, self).__init__()
        self.backbone = pretrained_model
        self.channel_attention = channel_attention
        self.spatial_attention = spatial_attention
    
    def forward(self, x):
        x = self.backbone.extract_features(x)
        x = self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        self.bert = BertModel(config)
        
    def forward(self, input_ids, attention_mask, encoder_hidden_states):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states
        )
        return outputs

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder, tokenizer, max_length):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.max_length = max_length  # Define max_length here
        self.fc = nn.Linear(1280, self.decoder.bert.config.hidden_size)

    def forward(self, images, captions=None):
        features = self.encoder(images)
        features = self.fc(features.mean([-2, -1]))  # Global average pooling and projection
        
        if captions is not None:
            tokenized_captions = self.tokenizer(captions, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length)
            tokenized_captions = {k: v.to(device) for k, v in tokenized_captions.items()}
            outputs = self.decoder(input_ids=tokenized_captions['input_ids'], 
                                   attention_mask=tokenized_captions['attention_mask'], 
                                   encoder_hidden_states=features.unsqueeze(1))
            return outputs.last_hidden_state
        else:
            return self.generate(features)

    def generate(self, features):
        batch_size = features.size(0)
        input_ids = torch.full((batch_size, 1), self.tokenizer.cls_token_id, dtype=torch.long, device=device)
        
        for _ in range(self.max_length):
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            outputs = self.decoder(input_ids=input_ids, 
                                   attention_mask=attention_mask, 
                                   encoder_hidden_states=features.unsqueeze(1))
            next_token_logits = outputs.last_hidden_state[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            
            if (next_token == self.tokenizer.sep_token_id).all():
                break
        
        return input_ids

 # Set your desired max length for captions

# Initialize components
pretrained_effnet = EfficientNet.from_pretrained('efficientnet-b0')
encoder = AttentionEfficientNet(pretrained_effnet, ChannelAttention(1280, 16), SpatialAttention()).to(device)

bert_config = BertConfig.from_pretrained('bert-base-uncased')
decoder = TransformerDecoder(bert_config).to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Set your desired max length for captions
max_caption_length = 50

# Instantiate the ImageCaptioningModel
model = ImageCaptioningModel(encoder, decoder, tokenizer, max_length=max_caption_length).to(device)

# If you have multiple GPUs and want to use DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# Example of how to use the model
def train_step(images, captions):
    images = images.to(device)
    outputs = model(images, captions)
    # Compute loss, backpropagate, etc.

def generate_captions(images):
    images = images.to(device)
    with torch.no_grad():
        caption_ids = model(images)
    captions = tokenizer.batch_decode(caption_ids, skip_special_tokens=True)
    return captions

# Don't forget to move your optimizer to CUDA if necessary
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)