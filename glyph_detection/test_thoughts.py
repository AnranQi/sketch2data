from PIL import Image
from ultralytics import YOLO
import os

design_size_x = 1000
design_size_y = 700
glyph_size_x = 130
glyph_size_y = 130

# Load a trained YOLOv8n model
model = YOLO(".\\runs\\detect\\thoughts_yolov8\\weights\\best.pt")
save_results_dir = '.\\thoughts_detection_results'
if not os.path.exists(save_results_dir):
    os.makedirs(save_results_dir)
image = '.\\thoughts.jpg'
output_imag = '.\\thoughts_resized.jpg'
target_size = (design_size_x, design_size_y)  # Example: (800, 600)

# Open an image file
with Image.open(image) as img:
    # Resize the image
    resized_img = img.resize(target_size)   
    # Save the resized image
    resized_img.save(output_imag)

results = model([output_imag])  # results list

bg_width, bg_height = background_size= (glyph_size_x,glyph_size_y)
image = Image.open(output_imag)
big_background = Image.new('RGBA', (design_size_x,design_size_y), 'white')
results = results[0]
boxes_lists = []
for i, box in enumerate(results.boxes):
    x1, y1, x2, y2 = box.xyxy.tolist()[0]
    boxes_lists.append((x1,y1,x2,y2))
boxes_lists = sorted(boxes_lists, key=lambda bbox:(bbox[1]*20 + bbox[0]))

# Save bounding boxes to a text file
bounding_boxes_file = os.path.join(save_results_dir, 'bounding_boxes.txt')
with open(bounding_boxes_file, 'w') as file:
    file.write("Bounding Boxes Information:\n")
for i, box in enumerate(boxes_lists):
    x1, y1, x2, y2 = box
    with open(bounding_boxes_file, 'a') as file:
        file.write(f"Image: {i}_test.jpg, Box: ({box[0]:.2f}, {box[1]:.2f}), ({box[2]:.2f}, {box[3]:.2f})\n")
    
    cropped_image = image.crop((x1, y1, x2, y2)).convert("RGBA")
    big_background.paste(cropped_image, (round(x1), round(y1)), cropped_image)
    overlay_width, overlay_height = cropped_image.size
    top_left_x = (bg_width - overlay_width) // 2
    top_left_y = (bg_height - overlay_height) // 2

    # Paste the overlay image onto the background
    background = Image.new('RGBA', background_size, 'white')
    background.paste(cropped_image, (top_left_x, top_left_y), cropped_image)
    background.save(os.path.join(save_results_dir, str(i)+"_test.png"))
    # Save JPG version
    background.convert('RGB').save(os.path.join(save_results_dir, f"{i}_test.jpg"), "JPEG")

big_background.save(os.path.join(save_results_dir,"big_test.png"))

