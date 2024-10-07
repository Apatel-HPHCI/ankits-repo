import glob 
import tkinter as tk 
import os 
import datetime
import threading
from tkinter import filedialog
from data_processor import DataProcessor

class GUI:
    # Initialize application window and widgets
    def __init__(self,root):
        self.root = root
        self.root.geometry("800x500")
        self.root.rowconfigure(1,weight=1)
        self.root.columnconfigure(0,weight=1)
        #self.root.columnconfigure(1,weight=1)

        self.checkbox_frame = tk.Frame(root, borderwidth=2, relief='groove')
        self.checkbox_frame.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

        self.log_text = tk.Text(root, wrap='word',height=10,width=40)
        self.log_text.grid(row=1, column=1, padx=10, pady=10, rowspan=2, sticky='nsew')

        self.scrollbar = tk.Scrollbar(root, orient='vertical', command=self.log_text.yview)
        self.scrollbar.grid(row=1, column=2, sticky='ns')
        self.log_text.config(yscrollcommand=self.scrollbar.set)

        self.folder_label = tk.Label(self.checkbox_frame, text="Folder Path:")
        self.folder_label.pack(anchor='w')

        self.browse_button= tk.Button(root, text="Select SCDM Folder", command=self.browse_folder)
        self.load_csv_files_button = tk.Button(root, text="Run QA Package", command=self.run_qa_package)
        self.abort_button = tk.Button(root, text="Abort QA Package", command=self.abort_qa_package)
        self.checkboxes = {}

        self.browse_button.grid(row=0, column=0, padx=10, pady=10) 
        self.load_csv_files_button.grid(row=2, column=0, padx=10, pady=10)
        self.abort_button.grid(row=3, column=0, padx=10, pady=10)

        self.data_processor = DataProcessor(self)
        
        self.update_load_button_state()

    # Browse folder method 
    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            for checkbox in self.checkboxes.values():
                checkbox.set(True)
            self.display_csv_files(folder_path)

    # Display CSV files 
    def display_csv_files(self, folder_path):
        for widget in self.checkbox_frame.winfo_children():
            widget.destroy()

        folder_label = tk.Label(self.checkbox_frame, text="Folder Path: " + folder_path)
        folder_label.pack(anchor='w')

        csv_files = glob.glob(os.path.join(folder_path, '*.csv')) 

        if csv_files:
            for file_path in csv_files:
                file_name = os.path.basename(file_path)
                var = tk.BooleanVar(value=True)
                checkbox = tk.Checkbutton(self.checkbox_frame, text=file_name, variable=var, command=self.update_load_button_state)
                checkbox.pack(anchor='w')
                self.checkboxes[file_path] = var 
        else:
            self.checkboxes.clear()

        self.update_load_button_state()

    # Grey out button based on values
    def update_load_button_state(self):
        if any(var.get() for var in self.checkboxes.values()):
            self.load_csv_files_button.config(state=tk.NORMAL)
        else:
            self.load_csv_files_button.config(state=tk.DISABLED)

    # Append log messages to GUI window
    def append_log_message(self, message):
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)
    
    # Run package
    def run_qa_package(self):
        self.load_csv_files_button.config(state=tk.DISABLED)
        # Process data on seperate thread to avoid stalling application
        processing_thread = threading.Thread(target=self.run_data_processing)
        processing_thread.start()

    # Process all SCDM tables
    def run_data_processing(self):
        selected_files = [file for file, var in self.checkboxes.items() if var.get()]
        datasets = self.data_processor.load_csv_files(selected_files)
        # Conditionally call each SCDM module
        if 'ENCOUNTER' in datasets:
            self.data_processor.encounter_module(datasets)
        # TO-DO: Add different modules here
        self.load_csv_files_button.config(state=tk.NORMAL)
    
    # Abort package if button is pressed, write log 
    def abort_qa_package(self):
        self.append_log_message("QA Package aborted.")

        log_content = self.log_text.get("1.0", tk.END)
        log_file_name = "qa_package_log_" + datetime.datetime.now().strftime("%Y-%m-%d") + ".txt"
        with open(log_file_name, 'w') as log_file:
            log_file.write(log_content)
        
        self.append_log_message(f"Log content has been saved to '{log_file_name}'.")

        self.root.destroy()
    