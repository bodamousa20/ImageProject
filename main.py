import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk
import os
import importlib
import threading

# Initialize the image label globally
image_label = None
function_var = None  # Variable to hold the selected function
processed_image_global = None  # Global variable to hold the processed image
Kernel = [
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
            processed_array = apply_kirsch_filter(image_array,Kernel) #

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

        elif function_name == "equalize_histogram": #err
            from ImageAlgoritm.histogram import equalize_histogram
            if isinstance(image_array, np.ndarray):
                image = Image.fromarray(image_array)
            else:
                image = image_array  # Already a PIL Image
            processed_image = equalize_histogram(image)







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
    """Run image processing in a background thread with a loading bar."""
    # Show loading indicator
    loading_bar.grid(row=7, column=0, padx=10, pady=10)  # Make the loading bar visible
    loading_bar.start()  # Start the loading animation

    # Run the image processing function in a separate thread
    threading.Thread(target=process_image, daemon=True).start()

    # Hide the loading bar once the processing is done
    loading_bar.stop()
    loading_bar.grid_forget()  # Hide the loading bar

# Welcome Screen
def welcome_screen():
    welcome_label = tk.Label(app, text="Welcome to the Image Processing App", font=("Arial", 24))
    welcome_label.pack(pady=20)

    welcome_img = Image.open("assets/welcome_image.jpg")
    welcome_img = welcome_img.resize((600, 450))
    welcome_img = ImageTk.PhotoImage(welcome_img)
    img_label = tk.Label(app, image=welcome_img)
    img_label.image = welcome_img
    img_label.pack()

    get_started_btn = tk.Button(app, text="Get Started", font=("Arial", 21), command=open_algorithm_screen)
    get_started_btn.pack(pady=20)

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

    global loading_bar
    loading_bar = ttk.Progressbar(image_frame, orient="horizontal", length=200, mode="indeterminate")

# Initialize the main window
app = tk.Tk()
app.title("Image Processing Application")
app.geometry("1200x600")

# Start with the welcome screen
welcome_screen()

# Run the application
app.mainloop()
