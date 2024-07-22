import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from treeclassify import convert_to_encoded, tree_classify
import pandas as pd
import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataProcessing')))

try:
    from dataProcessing import undersampling
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

with open('../TrainingTest/utils/label_mappings.json', 'r') as f:
    label_mappings = json.load(f)

event_descriptions = list(label_mappings["description"].keys())

def validate_date(date_str):
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def on_submit():
    date = date_entry.get()
    location = location_entry.get()
    city = city_entry.get()
    description = description_combobox.get()

    if not date or not location or not city or not description:
        messagebox.showwarning("Cancel", "Please fill all fields")
        return

    if not validate_date(date):
        messagebox.showwarning("Invalid Date", "Please enter the date in YYYY-MM-DD format")
        return

    try:
        progress_bar['value'] = 20
        root.update_idletasks()

        outputRF = tree_classify(date, location, city, description, "DecisionTree")
        output_vectorRF = outputRF[0]
        outputAB = tree_classify(date, location, city, description, "Adaboost")
        output_vectorAB = outputAB[0]

        if classifier_var.get() == "DecisionTree":
            clf = "Decision Tree"
            output = outputRF
        elif classifier_var.get() == "Adaboost":
            clf = "Adaboost"
            output = outputAB
        else:
            messagebox.showwarning("Error", "Please select a classifier")
            return
        progress_bar['value'] = 80

        index = output.argmax()
        if(output_vectorRF[index]>output_vectorAB[index] and clf == "Adaboost"):
            messagebox.showinfo("WARNING", "the model is not confident in its prediction,\ninstead the other you didn't select is more accurate, here is the prediction of the other model")
            output = outputRF
            clf = "Decision Tree"
        elif(output_vectorRF[index]<output_vectorAB[index] and clf == "Decision Tree"):
            messagebox.showinfo("WARNING", "the model is not confident in its prediction,\ninstead the other you didn't select is more accurate, here is the prediction of the other model")
            output = outputAB
            clf = "Adaboost"


        output_vector = output[0]
        value = convert_to_encoded(index, 'priority')
        
        progress_bar['value'] = 100
        root.update_idletasks()

        messagebox.showinfo("Results", f"Classifier: {clf}\nOutput: {value}\n % of probability: {int((max(output_vector[0], output_vector[1], output_vector[2]))*100)}\nHelp is on the way")

    except Exception as e:
        messagebox.showerror("ERROR", str(e))
        print(e)
        print("Error occurred at line:", sys.exc_info()[-1].tb_lineno)
        sys.exit(1)

root = tk.Tk()
root.title("911 Call")

# Apply a modern and futuristic look
root.configure(bg='#0D47A1')  # Dark Blue Background

# Create a style object
style = ttk.Style()

# Define styles for different widgets
style.configure('TFrame', background='#0D47A1')
style.configure('TLabel', background='#0D47A1', foreground='#FFFFFF', font=('Arial', 12))
style.configure('TCombobox', background='#1E2A38', foreground='#FFFFFF', font=('Arial', 12))
style.configure('TProgressbar', troughcolor='#0D47A1', background='#1976D2', thickness=20)

# Define a custom style for the button
style.configure('TButton',
                background='#1E2A38',  # Dark button color
                foreground='#000000',  # Black text color
                font=('Arial', 12),
                padding=10)

# Define how the button should look when pressed, etc.
style.map('TButton',
          background=[('pressed', '#1E2A38'), ('active', '#1565C0')],
          foreground=[('pressed', '#000000'), ('active', '#000000')])

# Create the main frame
main_frame = ttk.Frame(root, padding="20")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Apply the style to labels and buttons
ttk.Label(main_frame, text="911, what's your emergency?", font=("Arial", 18, 'bold')).grid(row=0, column=0, columnspan=2, pady=15)

ttk.Label(main_frame, text="Date (YYYY-MM-DD):").grid(row=1, column=0, sticky=tk.W, padx=5)
date_entry = ttk.Entry(main_frame)
date_entry.grid(row=1, column=1, padx=5)

ttk.Label(main_frame, text="Location:").grid(row=2, column=0, sticky=tk.W, padx=5)
location_entry = ttk.Entry(main_frame)
location_entry.grid(row=2, column=1, padx=5)

ttk.Label(main_frame, text="City:").grid(row=3, column=0, sticky=tk.W, padx=5)
city_entry = ttk.Entry(main_frame)
city_entry.grid(row=3, column=1, padx=5)

ttk.Label(main_frame, text="Event Description:").grid(row=4, column=0, sticky=tk.W, padx=5)
description_combobox = ttk.Combobox(main_frame, values=event_descriptions)
description_combobox.grid(row=4, column=1, padx=5)

# Create a StringVar to handle the radio buttons' state
classifier_var = tk.StringVar(value="")

ttk.Radiobutton(main_frame, text="Decision Tree", variable=classifier_var, value="DecisionTree").grid(row=5, column=0, padx=5, pady=5)
ttk.Radiobutton(main_frame, text="Adaboost", variable=classifier_var, value="Adaboost").grid(row=5, column=1, padx=5, pady=5)

submit_button = ttk.Button(main_frame, text="Submit", command=on_submit)
submit_button.grid(row=6, column=0, columnspan=2, pady=15)

progress_bar = ttk.Progressbar(main_frame, orient="horizontal", length=300, mode="determinate")
progress_bar.grid(row=7, column=0, columnspan=2, pady=15)

root.mainloop()
