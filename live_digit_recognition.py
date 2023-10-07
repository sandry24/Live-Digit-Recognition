import tkinter as tk
from PIL import Image, ImageGrab, ImageDraw
import numpy as np
import tensorflow as tf
import os
import json

model = tf.keras.models.load_model("Model(ResNet50)_digits_handdrawn_plus.h5")
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

canvas_width = 512
canvas_height = 512
line_smoothness = 10


def on_mouse_down(event):
    global prev_x, prev_y
    prev_x = event.x
    prev_y = event.y


def on_mouse_drag(event):
    global prev_x, prev_y, canvas, eraser_mode
    if not eraser_mode:
        canvas.create_line(prev_x, prev_y, event.x, event.y, width=3, fill=draw_color, smooth=True)
    else:
        canvas.create_line(prev_x, prev_y, event.x, event.y, width=35, fill=erase_color, smooth=True)
    prev_x = event.x
    prev_y = event.y


def on_mouse_release(event):
    recognize_digit()


def switch_to_drawing():
    global eraser_mode
    eraser_mode = False


def switch_to_eraser():
    global eraser_mode
    eraser_mode = True


def reset_canvas():
    global eraser_mode
    canvas.delete("all")
    eraser_mode = False


def save_image():
    global counter
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas_width
    y1 = y + canvas_height
    image = ImageGrab.grab((x, y, x1, y1)).convert('RGB')

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    filename = f"{base_filename}{counter}.png"
    file_path = os.path.join(folder_path, filename)

    image.save(file_path)
    counter += 1

    data = {"counter": counter}
    with open(json_file_path, "w") as json_file:
        json.dump(data, json_file)


def recognize_digit():
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas_width
    y1 = y + canvas_height
    image = ImageGrab.grab((x, y, x1, y1)).convert('RGB')
    image = image.resize((224, 224))

    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)

    predictions = model.predict(image_array)
    predicted_digit = label_predicted = labels[np.argmax(predictions[0])]

    result_label.config(text="Predicted Digit: {}".format(predicted_digit))
    print(predicted_digit)


# Test predict so that it doesnt lag during drawing
frst_arr = np.zeros((1, 224, 224, 3))
model.predict(frst_arr)
print("Setup Complete!")

root = tk.Tk()
root.title("Real-time Digit Recognition")

draw_color = "black"
erase_color = "white"
eraser_mode = False
prev_x, prev_y = None, None
folder_path = 'extra_digit_handdrawn'
json_file_path = 'live_digit_recognition_counter.json'
base_filename = 'train'
counter = 0

if os.path.exists(json_file_path):
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)
        counter = data["counter"]

canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

canvas.bind("<Button-1>", on_mouse_down)
canvas.bind("<B1-Motion>", on_mouse_drag)
canvas.bind("<ButtonRelease-1>", on_mouse_release)

draw_button = tk.Button(root, text="Drawing", command=switch_to_drawing)
draw_button.pack(side="left")

erase_button = tk.Button(root, text="Eraser", command=switch_to_eraser)
erase_button.pack(side="left")

reset_button = tk.Button(root, text="Reset", command=reset_canvas)
reset_button.pack(side="left")

save_button = tk.Button(root, text="Save", command=save_image)
save_button.pack(side="left")

# recognize_button = tk.Button(root, text="Recognize", command=recognize_digit)
# recognize_button.pack()

result_label = tk.Label(root, text="Predicted Digit: ", font=60)
result_label.pack()

root.mainloop()
