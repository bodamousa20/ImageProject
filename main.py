import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk
import os
import importlib
import threading

import image_opration

# Initialize the image label globally
image_label = None
function_var = None  # Variable to hold the selected function
processed_image_global = None  # Global variable to hold the processed image
Kernel_Mask = [
    0.0751, 0.1238, 0.0751,
    0.1238, 0.2042, 0.1238,
    0.0751, 0.1238, 0.0751
]

# Function to open the file dialog and choose an image
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((250, 250))  # Resize the image for display
        img = ImageTk.PhotoImage(img)

        global image_label
        if image_label is None:
            image_label = tk.Label(image_frame, image=img)
            image_label.grid(row=1, column=0)
        else:
            image_label.config(image=img)

        image_label.image = img
        app.image_path = file_path

def update_functions():
    """Update the functions dropdown based on the selected algorithm."""
    if algorithm_var.get() == "":
        messagebox.showerror("Error", "Please select an algorithm!")
        return

    algorithm_name = algorithm_var.get()
    try:
        # Dynamically load the selected algorithm module
        module = importlib.import_module(f"ImageAlgoritm.{algorithm_name}")
        functions = [func for func in dir(module) if callable(getattr(module, func)) and not func.startswith("__")]

        # Update the functions dropdown
        function_menu["values"] = functions
        function_var.set("")  # Reset the function selection
    except ModuleNotFoundError as e:
        messagebox.showerror("Error", f"Module not found: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def process_image():
    """Process the image using the selected function."""
    global processed_image_global  # Use the global variable

    if not hasattr(app, 'image_path') or not app.image_path:
        messagebox.showerror("Error", "Please upload an image first!")
        return

    if algorithm_var.get() == "":
        messagebox.showerror("Error", "Please select an algorithm!")
        return

    if function_var.get() == "":
        messagebox.showerror("Error", "Please select a function!")
        return

    algorithm_name = algorithm_var.get()
    function_name = function_var.get()
    img_path = app.image_path

    try:
        # Load the image and convert it to grayscale
        pil_image = Image.open(img_path).convert("L")  # Grayscale
        image_array = np.array(pil_image)  # Convert to numpy array

        # Dynamically call the function based on user selection
        # Dynamically call the function based on user selection
        if function_name == "sobel_filter":
            from ImageAlgoritm.simple_edge_detection import sobel_filter
            processed_array = sobel_filter(image_array)

        elif function_name == "prewitt_filter":
            from ImageAlgoritm.simple_edge_detection import prewitt_filter
            processed_array = prewitt_filter(image_array)

        elif function_name == "apply_kirsch_filter":#er
            from ImageAlgoritm.simple_edge_detection import apply_kirsch_filter
            processed_array = apply_kirsch_filter(image_array, Kernel_Mask) #

        elif function_name == "kirsch_operator":
            from ImageAlgoritm.simple_edge_detection import kirsch_operator
            processed_array = kirsch_operator(image_array)

        elif function_name == "convert_to_grayscale":
            from ImageAlgoritm.grayscale import convert_to_grayscale
            processed_array = convert_to_grayscale(image_array)

        elif function_name == "manual_thresholding":
            from ImageAlgoritm.histogram_based_segmentation import manual_thresholding
            processed_array = manual_thresholding(image_array, 20)

        elif function_name == "histogram_peak_thresholding":
            from ImageAlgoritm.histogram_based_segmentation import histogram_peak_thresholding
            processed_array = histogram_peak_thresholding(image_array)

        elif function_name == "histogram_valley_thresholding":
            from ImageAlgoritm.histogram_based_segmentation import histogram_valley_thresholding
            processed_array = histogram_valley_thresholding(image_array)

        elif function_name == "adaptive_histogram_thresholding":
            from ImageAlgoritm.histogram_based_segmentation import adaptive_histogram_thresholding
            processed_array = adaptive_histogram_thresholding(image_array)

        elif function_name == "homogeneity_operator":
            from ImageAlgoritm.advanced_edge_detection import homogeneity_operator
            processed_array = homogeneity_operator(image_array)

        elif function_name == "difference_operator":
            from ImageAlgoritm.advanced_edge_detection import difference_operator
            processed_array = difference_operator(image_array)

        elif function_name == "difference_of_gaussians":
            from ImageAlgoritm.advanced_edge_detection import difference_of_gaussians
            processed_array = difference_of_gaussians(image_array)

        elif function_name == "contrast_based_edge_detection":
            from ImageAlgoritm.advanced_edge_detection import contrast_based_edge_detection
            processed_array = contrast_based_edge_detection(image_array)

        elif function_name == "variance_operator":
            from ImageAlgoritm.advanced_edge_detection import variance_operator
            processed_array = variance_operator(image_array)

        elif function_name == "range_operator":
            from ImageAlgoritm.advanced_edge_detection import range_operator
            processed_array = range_operator(image_array)

        elif function_name == "grayscale":
            from ImageAlgoritm.grayscale import convert_to_grayscale
            processed_array = convert_to_grayscale(image_array)

        elif function_name == "equalize_histogram":
            from ImageAlgoritm.histogram import equalize_histogram
            if isinstance(image_array, np.ndarray):
                image = Image.fromarray(image_array)
            else:
                image = image_array  # Already a PIL Image
            processed_image = equalize_histogram(image)

        elif function_name == "high_pass_filter":
            from ImageAlgoritm.filters import high_pass_filter
            processed_image = high_pass_filter(Image.fromarray(image_array))  # تحويل numpy array إلى PIL Image
            processed_array = np.array(processed_image)  # إرجاع الصورة إلى numpy array إذا لزم الأمر

        elif function_name == "low_pass_filter":
            from ImageAlgoritm.filters import low_pass_filter
            processed_image = low_pass_filter(Image.fromarray(image_array))  # تحويل numpy array إلى PIL Image
            processed_array = np.array(processed_image)

        elif function_name == "apply_median_filter":
            from ImageAlgoritm.filters import apply_median_filter
            processed_image = apply_median_filter(Image.fromarray(image_array))  # تحويل numpy array إلى PIL Image
            processed_array = np.array(processed_image)

        elif function_name == "apply_threshold":
            from ImageAlgoritm.threshold import apply_threshold
            processed_image = apply_threshold(Image.fromarray(image_array))  # تحويل numpy array إلى PIL Image
            processed_array = np.array(processed_image)

        elif function_name =="apply_advanced_halftone":
            from ImageAlgoritm.halftone import apply_advanced_halftone
            processed_image = apply_advanced_halftone(Image.fromarray(image_array))
            processed_array = np.array(processed_image)

        elif function_name == "apply_simple_halftone":
            from ImageAlgoritm.halftone import apply_simple_halftone
            processed_array = apply_simple_halftone(image_array)









        else:
            raise ValueError(f"Function {function_name} is not recognized.")

        # Convert processed array back to an image
        processed_image = Image.fromarray(processed_array)

        # Display the processed image
        processed_image.show()

    except ModuleNotFoundError as e:
        messagebox.showerror("Error", f"Module not found: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


def process_image_with_loading():
    # Run the image processing function in a separate thread
    threading.Thread(target=process_image, daemon=True).start()

def twoImageOperation():
    two_image_window = tk.Toplevel(app)
    two_image_window.title("Two Image Operations")
    two_image_window.geometry("1000x850")

    # Frames for layout
    frame = tk.Frame(two_image_window)
    frame.pack(pady=20, padx=20)

    # Global variables for images
    global image1_path, image2_path
    image1_path, image2_path = None, None

    def upload_image(label, is_first=True):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            img = Image.open(file_path).resize((250, 250))  # Resize for display
            img_display = ImageTk.PhotoImage(img)
            label.config(image=img_display)
            label.image = img_display  # Keep reference

            # Assign path
            global image1_path, image2_path
            if is_first:
                image1_path = file_path
            else:
                image2_path = file_path

    def process_images():
        if not image1_path or not image2_path:
            messagebox.showerror("Error", "Please upload both images!")
            return

        operation = operation_var.get()
        if operation == "":
            messagebox.showerror("Error", "Please select an operation!")
            return

        try:
            # Load images
            img1 = np.array(Image.open(image1_path).convert("L"))
            img2 = np.array(Image.open(image2_path).convert("L"))

            # Perform the selected operation
            if operation == "Add":
                result_array = image_opration.add_images(img1, img2)
            elif operation == "Subtract":
                result_array = image_opration.subtract_images(img1, img2)
            elif operation == "Invert":
                result_array = image_opration.invert_image(img1)
            else:
                raise ValueError("Invalid operation selected.")

            # Display the result
            result_img = Image.fromarray(result_array)
            result_img_display = ImageTk.PhotoImage(result_img)

            result_label.config(image=result_img_display)
            result_label.image = result_img_display  # Keep reference
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


    # Upload buttons and image previews
    upload1_btn = tk.Button(frame, text="Upload First Image", command=lambda: upload_image(image1_label, True))
    upload1_btn.grid(row=0, column=0, padx=10, pady=10)

    image1_label = tk.Label(frame)
    image1_label.grid(row=1, column=0, padx=10)

    upload2_btn = tk.Button(frame, text="Upload Second Image", command=lambda: upload_image(image2_label, False))
    upload2_btn.grid(row=0, column=1, padx=10, pady=10)

    image2_label = tk.Label(frame)
    image2_label.grid(row=1, column=1, padx=10)

    # Operation selection
    operation_var = tk.StringVar(value="")
    operations = [("Add", "Add"), ("Subtract", "Subtract"), ("Invert", "Invert")]
    for idx, (text, value) in enumerate(operations):
        tk.Radiobutton(frame, text=text, variable=operation_var, value=value).grid(row=2, column=idx, padx=10, pady=5)

    # Result display
    result_label = tk.Label(frame)
    result_label.grid(row=1, column=2, padx=10)

    # Process button
    process_btn = tk.Button(two_image_window, text="Process Images", command=process_images)
    process_btn.pack(pady=10)



    # Upload first image
    def upload_first_image():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            img = Image.open(file_path).convert("L")  # Convert to grayscale
            img = img.resize((250, 250))  # Resize for consistency
            img_display = ImageTk.PhotoImage(img)

            first_image_label.config(image=img_display)
            first_image_label.image = img_display

            app.first_image = np.array(img)  # Save the image as a NumPy array

    # Upload second image
    def upload_second_image():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            img = Image.open(file_path).convert("L")  # Convert to grayscale
            img = img.resize((250, 250))  # Resize for consistency
            img_display = ImageTk.PhotoImage(img)

            second_image_label.config(image=img_display)
            second_image_label.image = img_display

            app.second_image = np.array(img)  # Save the image as a NumPy array

    # Process images
    def process_images():
        if not hasattr(app, 'first_image') or not hasattr(app, 'second_image'):
            messagebox.showerror("Error", "Please upload both images!")
            return

        operation = operation_var.get()
        if operation == "":
            messagebox.showerror("Error", "Please select an operation!")
            return

        try:
            if operation == "Add":

                result_array = image_opration.add_images(app.first_image, app.second_image)
            elif operation == "Subtract":
                result_array = image_opration.subtract_images(app.first_image, app.second_image)
            elif operation == "Invert":
                result_array = image_opration.subtract_images(app.first_image, app.second_image)

            # Display result
            result_image = Image.fromarray(result_array)
            result_image_display = ImageTk.PhotoImage(result_image)

            result_label.config(image=result_image_display)
            result_label.image = result_image_display

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    # GUI Layout
    upload_first_button = tk.Button(app, text="Upload First Image", command=upload_first_image)
    upload_first_button.grid(row=0, column=0, padx=10, pady=10)

    first_image_label = tk.Label(app)
    first_image_label.grid(row=1, column=0, padx=10, pady=10)

    upload_second_button = tk.Button(app, text="Upload Second Image", command=upload_second_image)
    upload_second_button.grid(row=0, column=1, padx=10, pady=10)

    second_image_label = tk.Label(app)
    second_image_label.grid(row=1, column=1, padx=10, pady=10)

    operation_var = tk.StringVar(value="")

    add_button = tk.Radiobutton(app, text="Add", variable=operation_var, value="Add")
    add_button.grid(row=2, column=0, padx=10, pady=5)

    subtract_button = tk.Radiobutton(app, text="Subtract", variable=operation_var, value="Subtract")
    subtract_button.grid(row=2, column=1, padx=10, pady=5)

    invert_button = tk.Radiobutton(app, text="Invert First Image", variable=operation_var, value="Invert")
    invert_button.grid(row=2, column=2, padx=10, pady=5)

    process_button = tk.Button(app, text="Process Images", command=process_images)
    process_button.grid(row=3, column=0, columnspan=3, pady=10)

    result_label = tk.Label(app)
    result_label.grid(row=4, column=0, columnspan=3, pady=10)

    app.mainloop()



# Welcome Screen
def welcome_screen():
    welcome_label = tk.Label(app, text="Welcome to the Image Processing App", font=("Arial", 24))
    welcome_label.pack(pady=20)

    welcome_img = Image.open("assets/welcome_image.jpg")
    welcome_img = welcome_img.resize((800, 350))
    welcome_img = ImageTk.PhotoImage(welcome_img)
    img_label = tk.Label(app, image=welcome_img)
    img_label.image = welcome_img
    img_label.pack()

    get_started_btn = tk.Button(app, text="One Image Operation", font=("Arial", 21), command=open_algorithm_screen)
    get_started_btn.pack(pady=10)

    TwoImageOperation = tk.Button(app, text="Two Image Operation", font=("Arial", 21), command=twoImageOperation)
    TwoImageOperation.pack(pady=10)
# Algorithm Selection Screen
def open_algorithm_screen():
    for widget in app.winfo_children():
        widget.destroy()

    global image_frame
    image_frame = tk.Frame(app)
    image_frame.pack(pady=20)

    upload_btn = tk.Button(image_frame, text="Upload Image", font=("Arial", 14), command=upload_image)
    upload_btn.grid(row=0, column=0, padx=25, pady=15)

    global image_label
    image_label = None

    algorithm_label = tk.Label(image_frame, text="Choose an Algorithm", font=("Arial", 16))
    algorithm_label.grid(row=2, column=0, padx=10, pady=5)

    algorithms = [f[:-3] for f in os.listdir('ImageAlgoritm') if f.endswith('.py') and f != '__init__.py']
    global algorithm_var
    algorithm_var = tk.StringVar()

    algorithm_menu = ttk.Combobox(image_frame, textvariable=algorithm_var, values=algorithms, font=("Arial", 14),
                                  state="readonly")
    algorithm_menu.grid(row=3, column=0, padx=10, pady=10)
    algorithm_menu.bind("<<ComboboxSelected>>", lambda e: update_functions())

    function_label = tk.Label(image_frame, text="Choose a Function", font=("Arial", 16))
    function_label.grid(row=4, column=0, padx=10, pady=10)

    global function_var
    function_var = tk.StringVar()
    global function_menu
    function_menu = ttk.Combobox(image_frame, textvariable=function_var, font=("Arial", 14), state="readonly")
    function_menu.grid(row=5, column=0, padx=10, pady=10)

    process_btn = tk.Button(image_frame, text="Process Image", font=("Arial", 14), command=process_image_with_loading)
    process_btn.grid(row=6, column=0, padx=10, pady=10)


# Initialize the main window
app = tk.Tk()
app.title("Image Processing Application")
app.geometry("1200x600")

# Start with the welcome screen
welcome_screen()

# Run the application
app.mainloop()
