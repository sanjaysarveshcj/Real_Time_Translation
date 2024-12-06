import tkinter as tk
from tkinter import messagebox, scrolledtext
import tensorflow as tf
import pickle
from importnb import Notebook

# Import translation models
with Notebook():
    from English_Hindi import NeuralMachineTranslation as HindiTranslation
    from English_French import NeuralMachineTranslation as FrenchTranslation

class TranslationApp:
    def __init__(self, master):
        self.master = master
        master.title("English to Hindi & French Translator")
        master.geometry("600x500")

        # Input Section
        self.input_label = tk.Label(master, text="Enter English Text:")
        self.input_label.pack(pady=(10, 5))

        self.input_text = scrolledtext.ScrolledText(master, height=5, width=70, wrap=tk.WORD)
        self.input_text.pack(pady=5)

        # Translate Button
        self.translate_button = tk.Button(master, text="Translate", command=self.translate_text)
        self.translate_button.pack(pady=10)

        # Output Sections
        # Hindi Translation
        self.hindi_label = tk.Label(master, text="Hindi Translation:")
        self.hindi_label.pack(pady=(10, 5))

        self.hindi_output = scrolledtext.ScrolledText(master, height=5, width=70, wrap=tk.WORD, state=tk.DISABLED)
        self.hindi_output.pack(pady=5)

        # French Translation
        self.french_label = tk.Label(master, text="French Translation:")
        self.french_label.pack(pady=(10, 5))

        self.french_output = scrolledtext.ScrolledText(master, height=5, width=70, wrap=tk.WORD, state=tk.DISABLED)
        self.french_output.pack(pady=5)

        # Model paths
        self.hindi_model_path = 'en_hi_model.h5'
        self.french_model_path = 'en_fr_model.h5'

        # Tokenizer paths
        self.hindi_tokenizer_paths = ('en_hi_input_tokenizer.pkl', 'en_hi_output_tokenizer.pkl')
        self.french_tokenizer_paths = ('en_fr_input_tokenizer.pkl', 'en_fr_output_tokenizer.pkl')

        # Initialize models and tokenizers
        self.load_models_and_tokenizers()

    def load_models_and_tokenizers(self):
        """Load translation models and their respective tokenizers."""
        try:
            # Load Hindi model and tokenizers
            self.hindi_nmt = HindiTranslation(max_sequence_length=5)
            self.hindi_nmt.model = tf.keras.models.load_model(self.hindi_model_path)
            with open(self.hindi_tokenizer_paths[0], "rb") as f:
                self.hindi_nmt.input_tokenizer = pickle.load(f)
            with open(self.hindi_tokenizer_paths[1], "rb") as f:
                self.hindi_nmt.output_tokenizer = pickle.load(f)

            # Load French model and tokenizers
            self.french_nmt = FrenchTranslation(max_sequence_length=5)
            self.french_nmt.model = tf.keras.models.load_model(self.french_model_path)
            with open(self.french_tokenizer_paths[0], "rb") as f:
                self.french_nmt.input_tokenizer = pickle.load(f)
            with open(self.french_tokenizer_paths[1], "rb") as f:
                self.french_nmt.output_tokenizer = pickle.load(f)

        except Exception as e:
            messagebox.showerror("Model/Tokenizer Loading Error", 
                f"Failed to load models or tokenizers: {str(e)}")

    def translate_text(self):
        """Translate input text to Hindi and French."""
        # Clear previous outputs
        self.hindi_output.config(state=tk.NORMAL)
        self.french_output.config(state=tk.NORMAL)
        self.hindi_output.delete(1.0, tk.END)
        self.french_output.delete(1.0, tk.END)

        # Get input text
        input_text = self.input_text.get(1.0, tk.END).strip()

        # Split input into words
        words = input_text.split()

        # Check if all words have 10 or more letters
        if not all(len(word) >= 10 for word in words):
            messagebox.showwarning("Input Error", 
                "Please upload text where ALL words have 10 or more letters.")
            return

        try:
            # Translate to Hindi
            hindi_translation = self.hindi_nmt.translate(input_text)
            self.hindi_output.insert(tk.END, hindi_translation)

            # Translate to French
            french_translation = self.french_nmt.translate(input_text)
            self.french_output.insert(tk.END, french_translation)

        except Exception as e:
            messagebox.showerror("Translation Error", 
                f"Failed to translate text: {str(e)}")

        # Disable output text areas to prevent editing
        self.hindi_output.config(state=tk.DISABLED)
        self.french_output.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = TranslationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
