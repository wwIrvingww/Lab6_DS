import nbformat
from nbformat import v4 as nbf

def merge_notebooks(file_paths, output_path):
    merged = nbf.new_notebook()
    
    for fp in file_paths:
        with open(fp, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        merged.cells.extend(nb.cells)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(merged, f)

# script hecho para pode juntar archivos jupyter, intentamos con nbmerge, pero da error por el encoding
# para sorpresa de nadie, las tildes seguramente sean el problema que no dejan juntarlos con
# un comando/libreria externa
if __name__ == '__main__':
    notebooks = ['trafico.ipynb', 'usuarios.ipynb']
    merge_notebooks(notebooks, 'analisis_trafico.ipynb')