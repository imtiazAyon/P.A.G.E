from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from subprocess import check_call
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Tk()
root.title("Project")
root.geometry("600x260")

input_entry = Entry(
    root,
    text="",
    fg="black",
    bg="white",
    readonlybackground="white",
    disabledforeground="black",
    state="readonly",
    width=60,
    borderwidth=5)
output_entry = Entry(
    root,
    text="",
    fg="black",
    bg="white",
    readonlybackground="white",
    disabledforeground="black",
    width=60,
    borderwidth=5)
model_entry = Entry(
    root,
    text="",
    fg="black",
    bg="white",
    readonlybackground="white",
    disabledforeground="black",
    width=60,
    borderwidth=5)
input_dir = ''
output_dir = ''
model_dir = ''

if os.path.exists("option.csv"):
    with open('option.csv') as file:
        reader = csv.DictReader(file)
        line = next(reader)
        input_dir = line['input_dir']
        output_dir = line['output_dir']
        model_dir = line['model_dir']


def update():
    global input_dir
    global output_dir
    global model_dir

    data = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "model_dir": model_dir
    }

    with open('option.csv', 'w', newline='') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "input_dir",
                "output_dir",
                "model_dir"])
        writer.writeheader()
        writer.writerow(data)


def choose_dir(option):

    global input_dir
    global output_dir
    global model_dir

    if option == 0:
        input_dir = filedialog.askdirectory(
            parent=root,
            initialdir=sys.path[0],
            title='Please select input directory')
    if option == 1:
        output_dir = filedialog.askdirectory(
            parent=root,
            initialdir=sys.path[0],
            title='Please select output directory')
    if option == 2:
        model_dir = filedialog.askopenfilename(
            initialdir=sys.path[0],
            title='Please select model file',
            filetypes=[('pkl files', '*.pkl')])

    if input_dir:
        input_entry['state'] = 'normal'
        input_entry.delete(0, END)
        input_entry.insert(0, input_dir)
        input_entry['state'] = 'readonly'

    if output_dir:
        output_entry['state'] = 'normal'
        output_entry.delete(0, END)
        output_entry.insert(0, output_dir)
        output_entry['state'] = 'readonly'

    if model_dir:
        model_entry['state'] = 'normal'
        model_entry.delete(0, END)
        model_entry.insert(0, model_dir)
        model_entry['state'] = 'readonly'

    update()


def return_output():

    global output_dir
    return pd.read_csv(output_dir + "/output.csv")


def Person_per_Image():

    df = return_output()
    Person = df.Face
    Images = df.img
    plt.style.use("fivethirtyeight")

    plt.figure(figsize=(20, 10))

    x_indexes = np.arange(len(Images))
    width = 0.25

    plt.bar(
        x_indexes - width,
        Person,
        width=width,
        color="#444444",
        label="Person")

    plt.legend()

    plt.title("Per Image Person Distribution")
    plt.xlabel("Images")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


def Male_and_Female_per_Image():

    df = return_output()
    images = df.img
    Male = df.Male
    Female = df.Female
    plt.style.use("fivethirtyeight")

    plt.figure(figsize=(20, 10))

    x_indexes = np.arange(len(images))
    width = 0.25

    plt.bar(x_indexes, Male, width=width, color="#444444", label="Male")

    plt.bar(
        x_indexes + width,
        Female,
        width=width,
        color="#e5ae38",
        label="Female")

    plt.legend()

    plt.title("Per Image Gender Distribution")
    plt.xlabel("Images")
    plt.ylabel("Count")

    plt.tight_layout()

    plt.show()


def Person_Age_per_Image():

    df = return_output()

    Images = df['img']
    Adst = df['Adst']
    T = df['T']
    Y_Ad = df['Y Ad']
    Ad = df['Ad']
    Old_Ad = df['Old Ad']
    Se = df['Se']

    plt.style.use("fivethirtyeight")

    plt.figure(figsize=(20, 10))

    x_indexes = np.arange(len(Images))
    width = 0.25
    shift1 = 1.75
    shift2 = 2.00
    shift3 = 2.25
    shift4 = 2.50
    shift5 = 2.75
    shift6 = 3.00

    plt.bar(
        x_indexes +
        shift1,
        Adst,
        width=width,
        color="#444444",
        label="Adolescent")
    plt.bar(
        x_indexes +
        shift2,
        T,
        width=width,
        color="#e5ae38",
        label="Teenager")
    plt.bar(
        x_indexes +
        shift3,
        Y_Ad,
        width=width,
        color="#FF5733",
        label="Young Adult")
    plt.bar(
        x_indexes +
        shift4,
        Ad,
        width=width,
        color="#3349FF",
        label="Adult")
    plt.bar(
        x_indexes +
        shift5,
        Old_Ad,
        width=width,
        color="#13F3DE",
        label="Old Adult")
    plt.bar(
        x_indexes +
        shift6,
        Se,
        width=width,
        color="#BC3FB8",
        label="Senior")

    plt.legend()

    plt.title("Per Image Age Distribution")
    plt.xlabel("Images")
    plt.ylabel("Count")

    plt.tight_layout()

    plt.show()


def run_model():

    global input_dir
    global output_dir
    global model_dir
    try:
        check_call(['python', 'pred.py', "--img_dir", input_dir,
                    "--output_dir", output_dir, "--model", model_dir])
    except BaseException:
        messagebox.showerror("Result", "Model did not run successfully!")
    else:
        messagebox.showinfo("Result", "Model ran successfully!")
        graph_btn_1['state'] = 'active'
        graph_btn_2['state'] = 'active'
        graph_btn_3['state'] = 'active'


input_btn = Button(
    root,
    text="Enter Input Directory",
    command=lambda: choose_dir(0),
    width=17)
output_btn = Button(
    root,
    text="Enter Output Directory",
    command=lambda: choose_dir(1),
    width=17)
model_btn = Button(
    root,
    text="Enter Model File",
    command=lambda: choose_dir(2),
    width=17)
submit_btn = Button(root, text="Run Model", command=run_model, width=17)
graph_btn_1 = Button(
    root,
    text="Create Graph 1",
    command=Person_per_Image,
    width=17,
    state="disabled")
graph_btn_2 = Button(
    root,
    text="Create Graph 2",
    command=Male_and_Female_per_Image,
    width=17,
    state="disabled")
graph_btn_3 = Button(
    root,
    text="Create Graph 3",
    command=Person_Age_per_Image,
    width=17,
    state="disabled")

input_btn.grid(row=0, column=0, pady=10, padx=20)
output_btn.grid(row=1, column=0, pady=10, padx=20)
model_btn.grid(row=2, column=0, pady=10, padx=20)
submit_btn.grid(row=3, column=1, pady=10)
graph_btn_1.grid(row=4, column=0, pady=10)
graph_btn_2.grid(row=4, column=1, pady=10)
graph_btn_3.grid(row=4, column=2, pady=10)

input_entry.grid(row=0, column=1, pady=10, padx=20, columnspan=2)
output_entry.grid(row=1, column=1, pady=10, padx=20, columnspan=2)
model_entry.grid(row=2, column=1, pady=10, padx=20, columnspan=2)

input_entry['state'] = 'normal'
input_entry.insert(0, input_dir)
input_entry['state'] = 'readonly'

output_entry['state'] = 'normal'
output_entry.insert(0, output_dir)
output_entry['state'] = 'readonly'

model_entry['state'] = 'normal'
model_entry.insert(0, model_dir)
model_entry['state'] = 'readonly'

root.mainloop()
