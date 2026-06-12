import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import time
import urllib.request

# ─────────────────────────────────────────────
# 1. MobileNetV2
# ─────────────────────────────────────────────
print("Initializing pre-trained MobileNetV2...")
model = models.mobilenet_v2(pretrained=True)
model.eval()

# ─────────────────────────────────────────────
# 2. ImageNet top-1000
# ─────────────────────────────────────────────
print("Downloading ImageNet labels...")
labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
try:
    imagenet_labels = urllib.request.urlopen(labels_url).read().decode('utf-8').splitlines()
    print(f"Successfully loaded {len(imagenet_labels)} classes!")
except Exception as e:
    print(f"Failed to download labels: {e}. Using fallback indices.")
    imagenet_labels = [f"Class {i}" for i in range(1000)]

def get_label(class_idx):
    if 0 <= class_idx < len(imagenet_labels):
        return imagenet_labels[class_idx]
    return f"Object ID: {class_idx}"

# ─────────────────────────────────────────────
# 3. preprocess pipeline: Resize +
#    CenterCrop
# ─────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),         
    transforms.CenterCrop(224),    
    transforms.ToTensor(),
])

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def get_crop_rect(frame):
    """
    calculate CenterCrop 
    pipeline: Resize 256 → CenterCrop 224
    return (x1, y1, x2, y2)
    """
    h, w = frame.shape[:2]
    # Step1: Resize 256
    if w < h:
        new_w, new_h = 256, int(h * 256 / w)
    else:
        new_w, new_h = int(w * 256 / h), 256
    # Step2: CenterCrop 224
    cx, cy = new_w // 2, new_h // 2
    crop_x1 = cx - 112  # 224/2
    crop_y1 = cy - 112
    crop_x2 = cx + 112
    crop_y2 = cy + 112

    scale_x = w / new_w
    scale_y = h / new_h
    x1 = int(crop_x1 * scale_x)
    y1 = int(crop_y1 * scale_y)
    x2 = int(crop_x2 * scale_x)
    y2 = int(crop_y2 * scale_y)
    return x1, y1, x2, y2

# ─────────────────────────────────────────────
# 4. FGSM 
# ─────────────────────────────────────────────
def fgsm_attack(norm_tensor, epsilon, grad):
    perturbed = norm_tensor + epsilon * grad.sign()
    return perturbed.detach()

def norm_to_display(norm_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (norm_tensor.squeeze(0).detach() * std + mean)
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def draw_top3(frame, probs, label_indices, ui_color, x=20, y_start=80):
    softmax_probs = torch.softmax(probs, dim=1)[0]
    for rank, idx in enumerate(label_indices):
        confidence = softmax_probs[idx].item() * 100
        label = get_label(idx)
        prefix = "▶" if rank == 0 else f" {rank+1}"
        text = f"{prefix}  {label:<28s} {confidence:5.1f}%"
        alpha = 1.0 - rank * 0.25     
        color = tuple(int(c * alpha) for c in ui_color)
        font_scale = 0.70 if rank == 0 else 0.58
        thickness  = 2    if rank == 0 else 1
        cv2.putText(frame, text, (x, y_start + rank * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# ─────────────────────────────────────────────
# 5. camera loop + UI
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

attack_enabled = False
epsilon = 0.05

print("\n=== Research System Active ===")
print("Commands: [A] Toggle Attack | [+/-] Epsilon | [Q] Quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(rgb_frame).unsqueeze(0)        # (1,3,224,224)
    norm_tensor  = normalize(input_tensor.squeeze(0)).unsqueeze(0)
    norm_tensor.requires_grad_(True)

    start_time = time.time()
    output = model(norm_tensor)
    latency = (time.time() - start_time) * 1000

    top3_values, top3_indices = output.topk(3, dim=1)
    top3_indices = top3_indices[0].tolist()  # [best, 2nd, 3rd]


    rx1, ry1, rx2, ry2 = get_crop_rect(frame)

    if attack_enabled:
        # ── FGSM  ──
        loss = torch.nn.CrossEntropyLoss()(output, torch.tensor([top3_indices[0]]))
        model.zero_grad()
        loss.backward()
        data_grad = norm_tensor.grad.data
        perturbed_norm = fgsm_attack(norm_tensor, epsilon, data_grad)

        with torch.no_grad():
            output_adv = model(perturbed_norm)

        adv_top3_values, adv_top3_indices = output_adv.topk(3, dim=1)
        adv_top3_indices = adv_top3_indices[0].tolist()

        display_frame = frame.copy()
        adv_crop = norm_to_display(perturbed_norm)
        adv_crop_resized = cv2.resize(adv_crop, (rx2 - rx1, ry2 - ry1))
        display_frame[ry1:ry2, rx1:rx2] = adv_crop_resized

        ui_color  = (0, 0, 255)
        status    = f"MODE: ADVERSARIAL (eps={epsilon:.3f}) | {latency:.1f}ms"
        draw_top3(display_frame, output_adv, adv_top3_indices, ui_color)

        orig_label = get_label(top3_indices[0])
        cv2.putText(display_frame, f"  [orig: {orig_label}]", (20, 80 + 3 * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 255), 1)
    else:
        display_frame = frame.copy()
        ui_color  = (0, 255, 0)
        status    = f"MODE: NORMAL INFERENCE | {latency:.1f}ms"
        draw_top3(display_frame, output, top3_indices, ui_color)

    # ── scene ──
    cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), ui_color, 2)
    cv2.putText(display_frame, "ROI", (rx1 + 6, ry1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_color, 2)

    # ── state  ──
    cv2.putText(display_frame, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, ui_color, 2)
    cv2.putText(display_frame, f"Epsilon: {epsilon:.3f}  (+/- adjust)",
                (20, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)

    cv2.imshow("MRP: Edge AI Robustness Testing", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        attack_enabled = not attack_enabled
        print(f"Attack Mode: {'ON' if attack_enabled else 'OFF'}")
    elif key in (ord('+'), ord('=')):
        epsilon = min(epsilon + 0.01, 0.5)
        print(f"Epsilon: {epsilon:.3f}")
    elif key == ord('-'):
        epsilon = max(epsilon - 0.01, 0.001)
        print(f"Epsilon: {epsilon:.3f}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()