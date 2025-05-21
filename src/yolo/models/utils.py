import torch
import torch.nn as nn


def train_step(model, optimizer, images, targets, device):
    """
    Single training step
    
    Args:
        model: YOLOv4 model
        optimizer: Optimizer
        images: Batch of images
        targets: Batch of targets
        device: Device to run on
    
    Returns:
        Loss dictionary
    """
    # Move data to device
    images = images.to(device)
    targets = [t.to(device) for t in targets]
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    predictions = model(images)
    
    # Calculate loss
    loss, loss_dict = model.compute_loss(predictions, targets)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    return loss_dict


def inference(model, image, device, conf_threshold=0.5):
    """
    Run inference on a single image
    
    Args:
        model: YOLOv4 model
        image: Input image tensor [C, H, W]
        device: Device to run on
        conf_threshold: Confidence threshold
    
    Returns:
        List of detections
    """
    model.eval()
    input_size = image.shape[1:3]
    
    # Add batch dimension and move to device
    image = image.unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(image)
    
    # Transform predictions
    detections = model.transform_predictions(predictions, input_size, conf_threshold)
    
    return detections


def example_usage():
    """
    Example of how to use the YOLOv4 model
    """
    from model import YOLOv4
    
    # Create model
    model = YOLOv4(num_classes=80)
    
    # Sample input
    x = torch.randn(1, 3, 416, 416)
    
    # Forward pass
    predictions = model(x)
    
    # Print output shapes
    for i, p in enumerate(predictions):
        print(f"Scale {i} output shape: {p.shape}")
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    example_usage()