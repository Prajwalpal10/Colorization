import argparse
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import torch
from colorizers import *


def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img_path_entry.delete(0, tk.END)
        img_path_entry.insert(tk.END, file_path)
        image = Image.open(file_path)
        image = image.resize((240, 360))  # Resize image to 240x240
        photo = ImageTk.PhotoImage(image)
        canvas.image = photo
        canvas.create_image(50, 50, anchor="nw", image=photo)



def process_image(img_path):
    # Load colorizers
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()

    if opt.use_gpu:
        colorizer_eccv16.cuda()
        colorizer_siggraph17.cuda()

    # Default size to process images is 256x256
    # Grab L channel in both original ("orig") and resized ("rs") resolutions
    img = load_img(img_path)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))

    if opt.use_gpu:
        tens_l_rs = tens_l_rs.cuda()

    # Colorizer outputs 256x256 ab map
    # Resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    # Save output image
    save_prefix = save_prefix_entry.get()
    if save_prefix:
        # plt.imsave('%s_eccv16.png' % save_prefix, out_img_eccv16)
        plt.imsave('%s_siggraph17.png' % save_prefix, out_img_siggraph17)

    # Display images
    plt.figure(figsize=(5, 5))
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(out_img_siggraph17)
    plt.title('Output')
    plt.axis('off')
    plt.show()


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
opt = parser.parse_args()

# Create the main window
window = tk.Tk()
window.title("Colorizer")

# Create the image canvas
canvas = tk.Canvas(window, width=500, height=400)
canvas.pack()

# Create the "Open Image" button
open_btn = tk.Button(window, text="Open Image", command=open_image)
open_btn.pack()

# Create the image path entry field
img_path_entry = tk.Entry(window, width=50)
img_path_entry.pack()

# Create the save prefix entry field
save_prefix_entry = tk.Entry(window, width=50)
save_prefix_entry.pack()


def process_button_click():
    img_path = img_path_entry.get()
    if img_path:
        process_image(img_path)


# Create the "Process Image" button
process_button = tk.Button(window, text='Process Image', command=process_button_click)
process_button.pack()

# Start the Tkinter event loop
window.mainloop()
