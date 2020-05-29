# Library imports

import os
import sys
# For building layouts
import tkinter as tk
# Varying text for labels
from tkinter import StringVar
# For choosing test image
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from Parameters import *
from cnn import *
from cnn1 import dl_model


def get_label_from_num(number):

    if number == 0:
        return "Healthy"
    elif number == 1:
        return "Yellow Leaf Curl Virus"
    elif number == 2:
        return "Late Blight"
    elif number == 3:
        return "Bacterial Spot"


def get_remedies(disease):

    if disease == "Late Blight":
        return " Monitor the field, remove and destroy infected leaves.\n  Treat organically with copper spray.\n  Use chemical fungicides, the best of which for tomatoes is chlorothalonil."
    elif disease == "Yellow Leaf Curl Virus":
        return " Monitor the field, handpick diseased plants and bury them.\n  Use sticky yellow plastic traps.\n  Spray insecticides such as organophosphates, carbametes during the seedling stage.\n Use copper fungicites"
    elif disease == "Bacterial Spot":
        return " Discard or destroy any affected plants.\n  Do not compost them.\n  Rotate yoour tomato plants yearly to prevent re-infection next year.\n Use copper fungicites"


def add_remedies(disease):

    remedies_description_string = "The remedies for " + disease + " are:" 
    var_remedies_description.set(remedies_description_string)

    remedies_string = get_remedies(disease)
    var_remedies.set(remedies_string)


def add_status(label):

    if label == 'Healthy':
        var_status.set('Plant is healthy')
        var_remedies_description.set("")
        var_remedies.set("")
    else:
        var_status.set('Disease Name: ' + label)
        add_remedies(label)


def analyse():

    test_image_path = os.path.join(TESTING_DATA, test_image_name)

    test_image = cv2.imread(test_image_path, cv2.IMREAD_COLOR)

    resized_test_image = cv2.resize(test_image, (IMAGE_SIZE, IMAGE_SIZE))

    output_number = np.argmax(dl_model.predict([np.array(resized_test_image)])[0])

    label = get_label_from_num(output_number)

    add_status(label)
    

def load_test_image():

    global test_image_name
    test_image_name = askopenfilename(initialdir=TESTING_DATA, title='Select image for analysis',
                           filetypes=[('image files', '.jpg')])

    test_image = Image.open(test_image_name)

    test_image = ImageTk.PhotoImage(test_image)   

    test_image_label = tk.Label(image=test_image, height="250", width="250")
    test_image_label.image = test_image

    test_image_label.place(x=0, y=0)
    test_image_label.grid(column=0, row=2, padx=10, pady = 10)

    # Add button to analyse image
    analyse_image_button = tk.Button(text="Start checkup", command=analyse)
    analyse_image_button.grid(column=0, row=3, padx=10, pady = 10)


 # Form window
window = tk.Tk()

# Add title
window.title("Dr.Plant")

# Set window size
window.geometry("500x500")

# Give window background
window.configure(background ="lightgreen")

# Give description for choosing image
description = tk.Label(text="Click below to choose the leaf image for testing disease", background = "lightgreen", fg="brown", font=("", "16"))
description.grid(column=0, row=0, padx=10, pady = 10)

# Add button to choose leaf image
choose_test_image_button = tk.Button(text="Choose leaf", command=load_test_image)
choose_test_image_button.grid(column=0, row=1, padx=10, pady = 10)

# Test image name
test_image_name = ""

# String variables
var_remedies_description = StringVar()
var_remedies = StringVar()
var_status = StringVar()

# Status label(healthy/disease)
status = tk.Label(textvariable=var_status, background="lightgreen",
                       fg="black", font=("", 15))
status.grid(column=0, row=4, padx=10, pady=10)

# Description for remedies
remedies_description = tk.Label(textvariable=var_remedies_description, background="lightgreen",
                      fg="brown", font=("", 15))
remedies_description.grid(column=0, row=5, padx=10, pady=10)

# Remedies label
remedies = tk.Label(textvariable=var_remedies, background="lightgreen",
                         fg="black", font=("", 12))
remedies.grid(column=0, row=6, padx=10, pady=10)

# Display window
window.mainloop()




