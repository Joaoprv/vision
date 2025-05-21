import torch
import torch.optim as optim
from model import YOLOv4
from utils import train_step, inference

def main():
    """
    Example of how to initialize and use the YOLOv4 model
    """
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 80  # COCO dataset has 80 classes
    input_size = (416, 416)
    learning_rate = 1e-4
    
    # Initialize model
    print("Initializing YOLOv4 model...")
    model = YOLOv4(num_classes=num_classes)
    model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Example input
    print("Running example inference...")
    example_input = torch.randn(1, 3, 416, 416).to(device)
    with torch.no_grad():
        outputs = model(example_input)
    
    # Check output shapes
    for i, output in enumerate(outputs):
        print(f"Output {i} shape: {output.shape}")
    
    # Example optimizer setup
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("YOLOv4 model initialized successfully!")

    
if __name__ == "__main__":
    main()