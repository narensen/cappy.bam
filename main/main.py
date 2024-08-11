import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from torchvision import transforms
from attentions import ImageCaptioningModel, model, tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CocoCaptions(
    root='/home/naren/Downloads/COCO 2017/train2017/train2017/',
    annFile='/home/naren/Downloads/COCO 2017/annotations_trainval2017/annotations/captions_train2017.json',
    transform=transform
)

val_dataset = CocoCaptions(
    root='/home/naren/Downloads/COCO 2017/val2017/val2017',
    annFile='/home/naren/Downloads/COCO 2017/annotations_trainval2017/annotations/captions_val2017.json',
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

model = model.to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id).to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, (images, captions) in enumerate(train_loader):
        images = images.to(device)
        captions = [cap[0] for cap in captions]
        
        optimizer.zero_grad()
        outputs = model(images, captions)
        
        target_captions = tokenizer(captions, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
        
        loss = criterion(outputs.view(-1, outputs.size(-1)), target_captions.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        val_loss = 0
        for images, captions in val_loader:
            images = images.to(device)
            captions = [cap[0] for cap in captions]
            
            outputs = model(images, captions)
            target_captions = tokenizer(captions, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
            
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_captions.view(-1))
            val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

    model.eval()
    with torch.no_grad():
        sample_image, _ = next(iter(val_loader))
        sample_image = sample_image[0].unsqueeze(0).to(device)
        generated_caption_ids = model.generate(sample_image)
        generated_caption = tokenizer.decode(generated_caption_ids[0], skip_special_tokens=True)
        print(f"Sample generated caption: {generated_caption}")

torch.save(model.state_dict(), 'image_captioning_model.pth')
print("Model saved successfully.")