import os
import cv2
import imutils
import pytz
import shutil
import gc

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend SEBELUM import pyplot
import matplotlib.pyplot as plt

from PIL import Image, ImageEnhance
from datetime import datetime 

# Setup direktori dan waktu
if not os.path.exists('./Data'):
    os.mkdir('./Data')

datasets_dir = "./Citra BISINDO"

UTC = pytz.utc
timeJKT = pytz.timezone('Asia/Jakarta') 
start_time = datetime.now(timeJKT).replace(microsecond=0)

final_size = (64, 64)

# Fungsi untuk membersihkan folder temporary
def clean_temp_folders():
    temp_folders = ['./resize', './brightness', './translate', './zoom', './rotate']
    for folder in temp_folders:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except Exception as e:
                print(f"Warning: Could not remove {folder}: {e}")

# Counter global untuk tracking
total_augmented_images = 0

print("=" * 70)
print("BISINDO IMAGE AUGMENTATION - MEMORY EFFICIENT VERSION")
print("=" * 70)
print(f"Start Time: {start_time}\n")

# OPSI: Force re-augment semua folder (hapus data lama)
FORCE_REAUGMENT = False  # Set True jika ingin re-augment semua

if FORCE_REAUGMENT:
    print("⚠️  FORCE RE-AUGMENT MODE: All existing data will be deleted!")
    confirm = input("Are you sure? (yes/no): ").strip().lower()
    if confirm == 'yes':
        if os.path.exists('./Data'):
            shutil.rmtree('./Data')
            os.mkdir('./Data')
        print("✓ Existing data deleted. Starting fresh augmentation...\n")
    else:
        print("❌ Cancelled. Exiting...")
        exit()
else:
    print("ℹ️  Skip Mode: Folders with existing data will be skipped")
    print("   To re-augment all, set FORCE_REAUGMENT = True\n")

# Main augmentation loop
for folder in sorted(os.listdir("%s" % datasets_dir)):
    folder_path = os.path.join(datasets_dir, folder)
    
    if not os.path.isdir(folder_path):
        continue
    
    # Buat folder output
    output_folder = os.path.join('./Data', folder)
    
    # CHECK: Skip jika folder sudah ada dan berisi data
    if os.path.exists(output_folder):
        existing_files = [f for f in os.listdir(output_folder) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
        if len(existing_files) > 0:
            print(f"⏭️  SKIP folder {folder} - Already augmented ({len(existing_files)} images found)")
            total_augmented_images += len(existing_files)
            print("=" * 70)
            continue  # SKIP to next folder
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    raw_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"▶️  Start with folder {folder} (with {len(raw_images)} raw image)")
    
    folder_start_time = datetime.now(timeJKT)

    sort_number = 1
    folder_count = 0
    
    for img_idx, image_data in enumerate(sorted(raw_images), 1):
        # Bersihkan dan buat folder temporary
        clean_temp_folders()
        os.mkdir('./resize')
        os.mkdir('./brightness')
        os.mkdir('./translate')
        os.mkdir('./zoom')
        os.mkdir('./rotate')

        print(f"  Processing image {img_idx}/{len(raw_images)}: {image_data[:30]}...")  # Show progress per image

        # ====================================================================
        # RESIZE ORIGINAL IMAGE
        # ====================================================================
        size_img = (256, 256)
        original = cv2.imread(os.path.join(datasets_dir, folder, image_data))
        
        if original is None:
            print(f"  Warning: Cannot read image {image_data}")
            continue
            
        original = cv2.resize(original, size_img)
        cv2.imwrite(os.path.join("./resize/original.jpg"), original)

        # ====================================================================
        # BRIGHTNESS AUGMENTATION
        # ====================================================================
        original_pil = Image.open("./resize/original.jpg")
        enhancer_original = ImageEnhance.Brightness(original_pil)

        brightness_factors = [0.75, 1.00, 1.25, 1.5]
        brightness_labels = ['075', '100', '125', '150']
        
        for factor, label in zip(brightness_factors, brightness_labels):
            im_output_original = enhancer_original.enhance(factor)
            im_output_original.save("./brightness/%s_brightness_original_%s.jpg" % (folder, label))

        # ====================================================================
        # TRANSLATION AUGMENTATION
        # ====================================================================
        original = cv2.imread("./resize/original.jpg")
        scaling = 16
        height_update = -16
        
        for translate_x in range(1, 6):
            width_update = -16
            for translate_y in range(1, 6):
                T = np.float32([[1, 0, width_update], [0, 1, height_update]]) 
                translate = cv2.warpAffine(original, T, (256, 256))
                translate = translate[0+scaling:256-scaling, 0+scaling:256-scaling]
                translate = cv2.resize(translate, size_img)
                cv2.imwrite(os.path.join("./translate/%s_translate_original_%d-%d.jpg" % (folder, translate_x, translate_y)), translate)
                width_update += 8
            height_update += 8

        # Translation on brightness images
        for image_data_bright in sorted(os.listdir("./brightness/")):
            brightness = cv2.imread("./brightness/" + image_data_bright)
            title = image_data_bright[0:len(image_data_bright)-4]
            scaling = 16
            height_update = -16
            
            for translate_x in range(1, 6):
                width_update = -16
                for translate_y in range(1, 6):
                    T = np.float32([[1, 0, width_update], [0, 1, height_update]]) 
                    translate = cv2.warpAffine(brightness, T, (256, 256))
                    translate = translate[0+scaling:256-scaling, 0+scaling:256-scaling]
                    translate = cv2.resize(translate, size_img)
                    cv2.imwrite(os.path.join("./translate/%s_translate_brightness_%d-%d.jpg" % (title, translate_x, translate_y)), translate)
                    width_update += 8
                height_update += 8
        
        # ====================================================================
        # ZOOM AUGMENTATION
        # ====================================================================
        original = cv2.imread("./resize/original.jpg")
        scaling = 12
        
        for j in range(1, 4):
            zoom_crop = original[0+scaling:256-scaling, 0+scaling:256-scaling]
            zoom_crop = cv2.resize(zoom_crop, size_img)
            cv2.imwrite(os.path.join("./zoom/%s_zoom_original_%03d.jpg" % (folder, j)), zoom_crop)
            scaling += 12

        # Zoom on brightness images
        for image_data_bright in sorted(os.listdir("./brightness/")):
            brightness = cv2.imread("./brightness/" + image_data_bright)
            title = image_data_bright[0:len(image_data_bright)-4]
            scaling = 12
            
            for j in range(1, 4):
                zoom_crop = brightness[0+scaling:256-scaling, 0+scaling:256-scaling]
                zoom_crop = cv2.resize(zoom_crop, size_img)
                cv2.imwrite(os.path.join("./zoom/%s_zoom_brightness_%03d.jpg" % (title, j)), zoom_crop)
                scaling += 12

        # Zoom on translated images
        for image_data_trans in sorted(os.listdir("./translate/")):
            translate_img = cv2.imread("./translate/" + image_data_trans)
            title = image_data_trans[0:len(image_data_trans)-4]
            scaling = 12
            
            for j in range(1, 4):
                zoom_crop = translate_img[0+scaling:256-scaling, 0+scaling:256-scaling]
                zoom_crop = cv2.resize(zoom_crop, size_img)
                cv2.imwrite(os.path.join("./zoom/%s_zoom_translate_%03d.jpg" % (title, j)), zoom_crop)
                scaling += 12
    
        # ====================================================================
        # ROTATION AUGMENTATION
        # ====================================================================
        scaling = 12
        index = 1
        original = cv2.imread("./resize/original.jpg") 
        
        for i in range(-5, 6):
            rotate = imutils.rotate(original, angle=10*i)
            rotate = rotate[0+scaling:256-scaling, 0+scaling:256-scaling]
            rotate = cv2.resize(rotate, size_img)
            cv2.imwrite(os.path.join("./rotate/%s_rotated_original_%03d.jpg" % (folder, index)), rotate)
            index += 1

        # Rotation on brightness images
        for image_data_bright in sorted(os.listdir("./brightness/")):
            brightness_img = cv2.imread("./brightness/" + image_data_bright)
            title = image_data_bright[0:len(image_data_bright)-4]
            index = 1
            
            for i in range(-5, 6):
                rotate = imutils.rotate(brightness_img, angle=10*i)
                rotate = rotate[0+scaling:256-scaling, 0+scaling:256-scaling]
                rotate = cv2.resize(rotate, size_img)
                cv2.imwrite(os.path.join("./rotate/%s_rotated_brightness_%03d.jpg" % (title, index)), rotate)
                index += 1

        # Rotation on translated images
        for image_data_trans in sorted(os.listdir("./translate/")):
            translate_img = cv2.imread("./translate/" + image_data_trans)
            title = image_data_trans[0:len(image_data_trans)-4]
            index = 1
            
            for i in range(-5, 6):
                rotate = imutils.rotate(translate_img, angle=10*i)
                rotate = rotate[0+scaling:256-scaling, 0+scaling:256-scaling]
                rotate = cv2.resize(rotate, size_img)
                cv2.imwrite(os.path.join("./rotate/%s_rotated_translate_%03d.jpg" % (title, index)), rotate)
                index += 1

        # Rotation on zoomed images
        for image_data_zoom in sorted(os.listdir("./zoom/")):
            zoom_img = cv2.imread("./zoom/" + image_data_zoom)
            title = image_data_zoom[0:len(image_data_zoom)-4]
            index = 1
            
            for i in range(-5, 6):
                rotate = imutils.rotate(zoom_img, angle=10*i)
                rotate = rotate[0+scaling:256-scaling, 0+scaling:256-scaling]
                rotate = cv2.resize(rotate, size_img)
                cv2.imwrite(os.path.join("./rotate/%s_rotated_zoom_%03d.jpg" % (title, index)), rotate)
                index += 1

        # ====================================================================
        # SAVE ALL AUGMENTED IMAGES (LANGSUNG KE DISK, TIDAK KE MEMORY)
        # ====================================================================
        temp_folders = ['./resize/', './brightness/', './translate/', './zoom/', './rotate/']
        
        for temp_folder in temp_folders:
            for image_file in sorted(os.listdir(temp_folder)):
                temp_img = cv2.imread(temp_folder + image_file)
                if temp_img is not None:
                    temp_img = cv2.resize(temp_img, final_size)
                    cv2.imwrite(os.path.join("./Data/%s/%s_augmented_%07d.jpg" % (folder, folder, sort_number)), temp_img)
                    sort_number += 1
                    folder_count += 1
        
        # Bersihkan memory setiap selesai 1 gambar original
        gc.collect()

    # Clean temporary folders setelah selesai 1 folder
    clean_temp_folders()
    
    # Hitung total file di folder output (tidak load ke memory!)
    folder_total = len([f for f in os.listdir(output_folder) if f.endswith('.jpg')])
    total_augmented_images += folder_total
    
    folder_end_time = datetime.now(timeJKT)
    folder_duration = folder_end_time - folder_start_time
    
    print(f"✓ Done with folder {folder}")
    print(f"  Total augmented: {folder_total:,} images")
    print(f"  Duration: {folder_duration}")
    print(f"  Avg: {folder_duration.total_seconds()/len(raw_images):.1f} sec/image")
    
    # Save progress checkpoint
    with open('.augmentation_progress.txt', 'a') as f:
        f.write(f"{folder},{folder_total},{datetime.now(timeJKT)}\n")
    
    print("=" * 70)
    
    # Force garbage collection
    gc.collect()

# ============================================================================
# FINAL STATISTICS
# ============================================================================
end_time = datetime.now(timeJKT).replace(microsecond=0)

# Count actual files per folder
print("\n" + "=" * 70)
print("DETAILED STATISTICS PER FOLDER")
print("=" * 70)

base_dir = "./Data"
if os.path.exists(base_dir):
    folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])
    total_files = 0
    
    print(f"{'Folder':<15} {'Images':>10} {'Status':<15}")
    print("-" * 70)
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        num_images = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
        total_files += num_images
        status = "✓ Complete" if num_images > 0 else "✗ Empty"
        print(f"{folder:<15} {num_images:>10,} {status:<15}")
    
    print("-" * 70)
    print(f"{'TOTAL':<15} {total_files:>10,}")

print("\n" + "=" * 70)
print("AUGMENTATION COMPLETED!")
print("=" * 70)
print(f"Total augmented images: {total_files:,}")
print(f"Total folders processed: {len(folders)}")
print(f"Start Time    : {start_time}")
print(f"End Time      : {end_time}")
print(f"Total Duration: {end_time - start_time}")
print("=" * 70)

# ============================================================================
# VISUALIZE SAMPLE IMAGES (ONLY LOAD 10 SAMPLES, NOT ALL!)
# ============================================================================
print("\nGenerating sample visualization...")
try:
    base_dir = "./Data"
    sample_images = []

    folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])

    for folder in folders[:10]:  # Ambil max 10 folder
        folder_path = os.path.join(base_dir, folder)
        images_in_folder = os.listdir(folder_path)
        
        if len(images_in_folder) > 0:
            first_image = images_in_folder[0]
            temporary_img = cv2.imread(os.path.join(folder_path, first_image))
            if temporary_img is not None:
                temporary_img = cv2.cvtColor(temporary_img, cv2.COLOR_BGR2RGB)
                sample_images.append(temporary_img)

    if len(sample_images) > 0:
        print(f"Visualizing {len(sample_images)} sample images...")
        row = 2
        col = 5
        fig = plt.figure(figsize=(col*2, row*2))

        for load_samples in range(min(10, len(sample_images))):
            fig.add_subplot(row, col, load_samples+1)
            plt.axis('off')
            plt.imshow(sample_images[load_samples])

        plt.tight_layout()
        plt.savefig('augmentation_samples.png', dpi=150, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        print("✓ Sample visualization saved to 'augmentation_samples.png'")
    else:
        print("⚠️  No sample images found to visualize")

except Exception as e:
    print(f"⚠️  Warning: Could not generate visualization: {e}")
    print("   Augmentation was successful, but visualization failed.")

print("\n✓ Augmentation completed successfully!")
print("📁 Data saved in './Data/' folder")
