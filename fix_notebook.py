import json

notebook_path = 'notebook.ipynb'
with open(notebook_path, 'r') as f:
    content = f.read()

content = content.replace("data_dir = '.'", "data_dir = 'data_set'")

with open(notebook_path, 'w') as f:
    f.write(content)
print("Notebook fixed.")
