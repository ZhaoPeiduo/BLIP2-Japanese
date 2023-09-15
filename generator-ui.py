import tkinter as tk
from tkinter import filedialog, Text, Menu, Scrollbar, ttk
from PIL import Image, ImageTk
from lavis.models import load_model_and_preprocess
import torch
import threading

class ImageDisplayApp:
    def __init__(self, root):
        self.model, self.visual_encoder, self.text_encoder = load_model_and_preprocess('blip2_Japanese', 'finetune')
        self.root = root
        self.root.title("Japanese Caption Generator")
        self.root.resizable(False, False)
        
        menubar = Menu(root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Upload", command=self.upload_image)
        menubar.add_cascade(label="File", menu=filemenu)
        root.config(menu=menubar)
        
        self.canvas = tk.Canvas(root, width=500, height=500, bg='white')
        self.canvas.pack(pady=20)
        
        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=500, mode='determinate')
        self.progress.pack(pady=20)

        self.text_output = Text(root, wrap=tk.WORD, height=10)
        self.text_output.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = Scrollbar(self.text_output)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_output.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.text_output.yview)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")))
        if not file_path:
            return
        
        image = Image.open(file_path)
        image.thumbnail((400, 400)) 
        photo = ImageTk.PhotoImage(image)

        self.canvas.image = photo
        self.canvas.create_image(250, 250, image=photo) 
        thread = threading.Thread(target=self.generate_caption, args=[image])
        thread.start()

    def generate_caption(self, image):
        self.progress['value'] = 1
        root.update_idletasks()
        self.text_output.delete("1.0", tk.END)
        image = torch.unsqueeze(self.visual_encoder['eval'](image), 0)
        data_input  = {"image" : image}
        self.progress['value'] = 25
        root.update_idletasks()
        caption = self.model.generate(data_input)[0]
        self.progress['value'] = 75
        root.update_idletasks()
        self.text_output.insert("1.0", caption)
        self.text_output.mark_set(tk.INSERT, "1.0")
        self.progress['value'] = 100
        root.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageDisplayApp(root)
    root.mainloop()

